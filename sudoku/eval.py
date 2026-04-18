"""
sudoku/eval.py
──────────────
Evaluation utilities for IERM-TRM on Sudoku.

Reports, per call:
  - `grid_solved` : fraction of test grids fully completed correctly
  - `px`          : per-cell accuracy across all 81 cells
  - `empty_acc`   : accuracy on cells that were empty in the query
                    (this is the real signal — given cells are trivially copied)
  - `given_acc`   : accuracy on cells already filled in the query
  - `solved_at_k` : fraction of grids with ≤ k wrong cells
  - `violations`  : mean count of Sudoku row/col/block rule violations in preds
  - `confidence`  : mean confidence head output

The main entry-point `evaluate_sudoku` supports the ablation asked for
in the paper: it can produce both TTA and no-TTA numbers side by side.

Expected batch format (consumed by `model.predict`):
    batch = {
        "sx"      : LongTensor [B, S, 9, 9],
        "sy"      : LongTensor [B, S, 9, 9],
        "qx"      : LongTensor [B, 9, 9],          # 0 = empty cell
        "qy"      : LongTensor [B, 9, 9],          # ground truth
        "s_mask"  : BoolTensor [B, S]   (optional; defaults to all-True)
        "task_id" : LongTensor [B]      (optional)
    }
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, Optional

import torch
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════
# Sudoku rule-violation counter (diagnostic only; non-differentiable)
# ══════════════════════════════════════════════════════════════

def sudoku_violation_count(pred: torch.Tensor,
                            H: int = 9, W: int = 9) -> float:
    """
    Count Sudoku rule violations in a batch of predicted grids.

    A violation is a repeated non-zero digit within a row, column, or
    3×3 block. Returns the mean count per grid.

    Vectorized to avoid the per-batch-per-row loop of the notebook version.
    """
    B = pred.shape[0]
    pred = pred.reshape(B, H, W)

    # Stack the 27 "groups" (9 rows + 9 cols + 9 blocks) as [B, 27, 9].
    rows = pred                                           # [B, 9, 9]
    cols = pred.transpose(1, 2)                           # [B, 9, 9]
    blocks = pred.reshape(B, 3, 3, 3, 3).permute(0, 1, 3, 2, 4).reshape(B, 9, 9)

    groups = torch.cat([rows, cols, blocks], dim=1)        # [B, 27, 9]

    # For each group of 9 cells and each digit 1..9, count how many times
    # the digit appears; a valid group has count ≤ 1 for every digit.
    # One-hot over digits 1..9, ignore 0.
    nonzero = (groups > 0)                                 # [B, 27, 9]
    # Build one-hot over digits 1..9 (9 classes).
    # Clamp to [1, 9] before one-hot; zero out contributions from 0s.
    gclamp = groups.clamp(0, 9).long()
    onehot = F.one_hot(gclamp, num_classes=10)[..., 1:]   # [B, 27, 9, 9]
    onehot = onehot * nonzero.unsqueeze(-1)

    counts = onehot.sum(dim=2)                             # [B, 27, 9]
    # A digit appearing k times incurs (k - 1) violations when k ≥ 1.
    excess = (counts - 1).clamp(min=0)                     # [B, 27, 9]
    total = excess.sum(dim=(1, 2)).float()                 # [B]
    return float(total.mean().item())


# ══════════════════════════════════════════════════════════════
# Main evaluation loop
# ══════════════════════════════════════════════════════════════

def _eval_once(
    model,
    loader: Iterable,
    device: str,
    *,
    pred_T: int,
    use_tta: bool,
    n_tta: int,
    iterative_rounds: int,
    residual_alpha: float,
    max_batches: Optional[int],
) -> Dict[str, float]:
    """Run one full pass with a single (use_tta, n_tta) configuration."""
    model.eval()
    H = W = 9
    L = H * W
    EMPTY_TOKEN = 0

    totals: Dict[str, float] = defaultdict(float)
    n_samples = 0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            if max_batches is not None and n_batches >= max_batches:
                break

            sx = batch["sx"].to(device)
            sy = batch["sy"].to(device)
            qx = batch["qx"].to(device)
            qy = batch["qy"].to(device)

            s_mask = batch.get("s_mask")
            if s_mask is not None:
                s_mask = s_mask.to(device).bool()

            task_id = batch.get("task_id")
            if isinstance(task_id, torch.Tensor):
                task_id = task_id.to(device)

            B = qx.shape[0]

            pred, conf, _steps = model.predict(
                sx, sy, qx,
                s_mask=s_mask, task_id=task_id,
                T_max=pred_T,
                residual_alpha=residual_alpha,
                use_tta=use_tta,
                n_aug_passes=n_tta if use_tta else 0,
                iterative_rounds=iterative_rounds,
            )

            pred_flat = pred.reshape(B, L)
            gt_flat = qy.reshape(B, L).long()
            qx_flat = qx.reshape(B, L).long()

            mask_empty = (qx_flat == EMPTY_TOKEN)
            mask_given = ~mask_empty
            correct = (pred_flat == gt_flat)

            # Weighted accumulators — weighted by B so averaging works later.
            totals["px"] += float(correct.float().mean()) * B

            d_empty = mask_empty.float().sum().clamp_min(1)
            d_given = mask_given.float().sum().clamp_min(1)
            totals["empty_acc"] += float(
                (correct & mask_empty).float().sum() / d_empty) * B
            totals["given_acc"] += float(
                (correct & mask_given).float().sum() / d_given) * B

            # Per-grid metrics: sum is already over batch dim.
            solved = correct.all(dim=1)
            totals["grid_solved"] += float(solved.float().sum())

            err = (~correct).sum(dim=1)
            totals["solved_at_2"] += float((err <= 2).float().sum())
            totals["solved_at_5"] += float((err <= 5).float().sum())

            totals["confidence"] += float(conf) * B
            totals["n_empty"] += float(mask_empty.float().sum(1).mean()) * B

            # Violations on the predicted grid (diagnostic).
            totals["violations"] += sudoku_violation_count(pred, H, W) * B

            n_samples += B
            n_batches += 1

    n = max(n_samples, 1)
    avg = {k: v / n for k, v in totals.items()}
    avg["_n_samples"] = float(n_samples)
    avg["_n_batches"] = float(n_batches)
    return avg


def _format_row(tag: str, avg: Dict[str, float]) -> str:
    return (f"  {tag:10s} | "
            f"solved={avg['grid_solved']:.4f}  "
            f"px={avg['px']:.4f}  "
            f"empty={avg['empty_acc']:.4f}  "
            f"given={avg['given_acc']:.4f}  "
            f"solved@2={avg['solved_at_2']:.4f}  "
            f"solved@5={avg['solved_at_5']:.4f}  "
            f"viol={avg['violations']:.2f}  "
            f"conf={avg['confidence']:.3f}")


def evaluate_sudoku(
    model,
    loader: Iterable,
    device: str,
    *,
    pred_T: int = 5,
    residual_alpha: float = 0.4,
    # What to report:
    report_no_tta: bool = True,
    report_tta: bool = True,
    # TTA settings:
    n_tta: int = 8,
    iterative_rounds: int = 1,
    # Scope:
    max_batches: Optional[int] = None,
    # Output:
    print_diag: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate IERM_TRM_Sudoku on a Sudoku dataset.

    By default returns BOTH no-TTA and TTA results, for the paper ablation.

    Args:
        model          : an IERM_TRM_Sudoku (or any model whose `.predict`
                         accepts `use_tta`, `n_aug_passes`, `T_max`,
                         `residual_alpha`, `iterative_rounds`).
        loader         : yields batches with keys sx, sy, qx, qy[, s_mask, task_id].
        pred_T         : number of latent-recursion steps at inference.
        residual_alpha : must match training (0.4 for the paper model).
        report_no_tta  : include a no-TTA pass (fast, single forward).
        report_tta     : include a TTA pass (averages 1 + n_tta digit
                         permutations — this is the headline number).
        n_tta          : number of augmented passes for the TTA variant.
        max_batches    : cap the number of batches (None = full loader).
        print_diag     : print a compact summary table.

    Returns:
        dict with keys among {"no_tta", "tta"}, each mapping to a dict
        of metrics as documented at the top of this file.
    """
    assert report_no_tta or report_tta, \
        "At least one of report_no_tta / report_tta must be True."

    results: Dict[str, Dict[str, float]] = {}

    if report_no_tta:
        results["no_tta"] = _eval_once(
            model, loader, device,
            pred_T=pred_T,
            use_tta=False, n_tta=0,
            iterative_rounds=iterative_rounds,
            residual_alpha=residual_alpha,
            max_batches=max_batches,
        )

    if report_tta:
        results["tta"] = _eval_once(
            model, loader, device,
            pred_T=pred_T,
            use_tta=True, n_tta=n_tta,
            iterative_rounds=iterative_rounds,
            residual_alpha=residual_alpha,
            max_batches=max_batches,
        )

    if print_diag:
        n_samples = int(next(iter(results.values()))["_n_samples"])
        print(f"  ┌─ EVAL SUDOKU ({n_samples} samples, T={pred_T}, "
              f"α={residual_alpha}) ─────")
        if "no_tta" in results:
            print(_format_row("no-TTA", results["no_tta"]))
        if "tta" in results:
            print(_format_row(f"TTA×{n_tta}", results["tta"]))
            if "no_tta" in results:
                delta = (results["tta"]["grid_solved"]
                         - results["no_tta"]["grid_solved"])
                print(f"  TTA lift on grid_solved: {delta:+.4f}")
        print(f"  └─────────────────────────────────────────────────\n")

    return results


# ══════════════════════════════════════════════════════════════
# Convenience: full reproduction of the paper number
# ══════════════════════════════════════════════════════════════

def reproduce_paper_number(
    model,
    loader: Iterable,
    device: str,
    *,
    pred_T: int = 5,
    n_tta: int = 8,
    expected_tta_solved: float = 0.63,
    tolerance: float = 0.01,
) -> float:
    """
    Run the full TTA evaluation and assert the headline number.

    Returns the achieved `grid_solved` with TTA. Raises AssertionError
    if the result is not within `tolerance` of `expected_tta_solved`.
    """
    results = evaluate_sudoku(
        model, loader, device,
        pred_T=pred_T,
        report_no_tta=True,
        report_tta=True,
        n_tta=n_tta,
        max_batches=None,
        print_diag=True,
    )
    achieved = results["tta"]["grid_solved"]
    msg = (f"Expected TTA grid_solved ≈ {expected_tta_solved:.4f} "
           f"(± {tolerance:.3f}), got {achieved:.4f}")
    assert abs(achieved - expected_tta_solved) <= tolerance, msg
    print(f"✓ Paper number reproduced: {achieved:.4f} "
          f"(expected {expected_tta_solved:.4f})")
    return achieved


__all__ = [
    "evaluate_sudoku",
    "reproduce_paper_number",
    "sudoku_violation_count",
]