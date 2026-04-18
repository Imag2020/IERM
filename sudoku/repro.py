"""
sudoku/repro.py
───────────────
Reproduce the Sudoku Extreme numbers from the paper.

Evaluates the shipped checkpoint and reports both no-TTA and TTA
numbers side by side, with a live progress bar and ETA.

Defaults to --max_batches 3000 (~96k puzzles at batch_size=32), which
matches the evaluation protocol used in the paper. Pass --max_batches 0
(or a very large value) to evaluate on the full 1M+ test set.

Usage
─────
    python -m sudoku.repro \\
        --ckpt      model_checkpoints/sudoku_mlp_62pct.pt \\
        --data_dir  data/sudoku \\
        --save_json results/sudoku_eval.json

    # Full-set evaluation (hours):
    python -m sudoku.repro --ckpt ... --max_batches 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Iterable, Optional

import torch

from sudoku.model import (
    build_sudoku_model,
    load_sudoku_checkpoint,
    CFG_SUDOKU,
)
from sudoku.eval import sudoku_violation_count


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Reproduce IERM Sudoku Extreme numbers.")
    p.add_argument("--ckpt",
                   default="model_checkpoints/sudoku_mlp_62pct.pt",
                   help="Path to the Sudoku checkpoint.")
    p.add_argument("--data_dir",
                   default="data/sudoku",
                   help="Directory containing test.npz.")
    p.add_argument("--split", default="test")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--n_supports", type=int, default=4,
                   help="Number of support puzzles per query.")
    p.add_argument("--pred_T", type=int, default=5,
                   help="Latent-recursion steps at inference.")
    p.add_argument("--n_tta", type=int, default=8,
                   help="Augmented passes for TTA.")
    p.add_argument("--residual_alpha", type=float, default=0.4,
                   help="Must match the training value (0.4).")
    p.add_argument("--max_batches", type=int, default=3000,
                   help="Cap number of eval batches (0 = full set).")
    p.add_argument("--save_json", default=None,
                   help="If set, write the results as JSON here.")
    p.add_argument("--device", default=None)
    p.add_argument("--no_tta", action="store_true",
                   help="Skip the TTA pass (no-TTA only, ~9× faster).")
    p.add_argument("--tta_only", action="store_true",
                   help="Skip the no-TTA pass (TTA only).")
    p.add_argument("--log_every", type=int, default=50,
                   help="Print a status line every N batches (fallback if tqdm missing).")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════
# Progress helper (tqdm if available, else text-only)
# ══════════════════════════════════════════════════════════════

def _get_progress_bar(iterable, total: int, desc: str, log_every: int):
    """Return an iterable that shows progress. Uses tqdm when available."""
    try:
        from tqdm.auto import tqdm
        return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True,
                    mininterval=0.5)
    except ImportError:
        # Fallback: plain iterator with periodic logs.
        def _gen():
            t0 = time.time()
            for i, x in enumerate(iterable, 1):
                yield x
                if i % log_every == 0 or i == total:
                    elapsed = time.time() - t0
                    rate = i / max(elapsed, 1e-6)
                    eta = (total - i) / max(rate, 1e-6)
                    print(f"    [{desc}] {i:,}/{total:,}  "
                          f"{rate:.1f} it/s  ETA {eta/60:.1f} min",
                          flush=True)
        return _gen()


# ══════════════════════════════════════════════════════════════
# Evaluation loop (single configuration, with progress)
# ══════════════════════════════════════════════════════════════

def eval_sudoku_progress(
    model,
    loader: Iterable,
    device: str,
    *,
    pred_T: int,
    use_tta: bool,
    n_tta: int,
    residual_alpha: float,
    max_batches: Optional[int],
    desc: str,
    log_every: int = 50,
) -> dict:
    """Run one eval pass with progress reporting."""
    model.eval()
    H = W = 9
    L = H * W
    EMPTY_TOKEN = 0

    totals: dict = defaultdict(float)
    n_samples = 0
    n_batches = 0

    try:
        total = len(loader) if max_batches is None else min(len(loader), max_batches)
    except TypeError:
        total = max_batches or 0

    bar = _get_progress_bar(loader, total, desc, log_every)

    with torch.no_grad():
        for batch in bar:
            if max_batches is not None and n_batches >= max_batches:
                break

            sx = batch["sx"].to(device, non_blocking=True)
            sy = batch["sy"].to(device, non_blocking=True)
            qx = batch["qx"].to(device, non_blocking=True)
            qy = batch["qy"].to(device, non_blocking=True)

            s_mask = batch.get("s_mask")
            if s_mask is not None:
                s_mask = s_mask.to(device).bool()

            task_id = batch.get("task_id")
            if isinstance(task_id, torch.Tensor):
                task_id = task_id.to(device)

            B = qx.shape[0]

            pred, conf, _ = model.predict(
                sx, sy, qx,
                s_mask=s_mask, task_id=task_id,
                T_max=pred_T,
                residual_alpha=residual_alpha,
                use_tta=use_tta,
                n_aug_passes=n_tta if use_tta else 0,
                iterative_rounds=1,
            )

            pred_flat = pred.reshape(B, L)
            gt_flat = qy.reshape(B, L).long()
            qx_flat = qx.reshape(B, L).long()

            mask_empty = (qx_flat == EMPTY_TOKEN)
            mask_given = ~mask_empty
            correct = (pred_flat == gt_flat)

            totals["px"] += float(correct.float().mean()) * B
            d_empty = mask_empty.float().sum().clamp_min(1)
            d_given = mask_given.float().sum().clamp_min(1)
            totals["empty_acc"] += float(
                (correct & mask_empty).float().sum() / d_empty) * B
            totals["given_acc"] += float(
                (correct & mask_given).float().sum() / d_given) * B

            solved = correct.all(dim=1)
            totals["grid_solved"] += float(solved.float().sum())
            err = (~correct).sum(dim=1)
            totals["solved_at_2"] += float((err <= 2).float().sum())
            totals["solved_at_5"] += float((err <= 5).float().sum())
            totals["confidence"] += float(conf) * B
            totals["violations"] += sudoku_violation_count(pred, H, W) * B

            n_samples += B
            n_batches += 1

            # Live progress update for tqdm (running solved rate).
            if hasattr(bar, "set_postfix_str"):
                cur_solved = totals["grid_solved"] / max(n_samples, 1)
                bar.set_postfix_str(f"solved={cur_solved:.4f}")

    # Close tqdm properly.
    if hasattr(bar, "close"):
        bar.close()

    n = max(n_samples, 1)
    avg = {k: v / n for k, v in totals.items()}
    avg["_n_samples"] = float(n_samples)
    avg["_n_batches"] = float(n_batches)
    return avg


# ══════════════════════════════════════════════════════════════
# Summary printing
# ══════════════════════════════════════════════════════════════

def _print_row(tag: str, avg: dict) -> None:
    print(f"  {tag:10s} | solved={avg['grid_solved']:.4f}  "
          f"px={avg['px']:.4f}  empty={avg['empty_acc']:.4f}  "
          f"given={avg['given_acc']:.4f}  "
          f"s@2={avg['solved_at_2']:.4f}  s@5={avg['solved_at_5']:.4f}  "
          f"viol={avg['violations']:.2f}  conf={avg['confidence']:.3f}")


def main() -> int:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"  Device          : {device}")
    print(f"  Checkpoint      : {args.ckpt}")
    print(f"  Data dir        : {args.data_dir}")
    print(f"  pred_T          : {args.pred_T}")
    print(f"  n_tta           : {args.n_tta}")
    print(f"  residual_alpha  : {args.residual_alpha}")
    max_b = None if args.max_batches <= 0 else args.max_batches
    print(f"  max_batches     : {max_b if max_b else 'full set'}")

    if not os.path.exists(args.ckpt):
        print(f"\n✗ Checkpoint not found: {args.ckpt}", file=sys.stderr)
        return 1

    # ── Build model + load checkpoint ─────────────────────────
    print(f"\n  Building model ...")
    model = build_sudoku_model(CFG_SUDOKU).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters      : {n_params:,}")

    print(f"  Loading checkpoint ...")
    ckpt = load_sudoku_checkpoint(model, args.ckpt,
                                   device=device, strict=True)
    print(f"  Checkpoint step : {ckpt.get('step', 'n/a')}")
    m = ckpt.get("metrics", {})
    if isinstance(m, dict) and m:
        saved_eval = m.get("eval", m)
        if isinstance(saved_eval, dict) and saved_eval:
            print(f"  Saved metrics   : "
                  f"solved={saved_eval.get('grid_solved', '?')}")

    # ── Build dataloader ──────────────────────────────────────
    from sudoku.data import build_sudoku_loader
    loader = build_sudoku_loader(
        data_dir=args.data_dir,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        n_supports=args.n_supports,
    )
    n_batches_total = len(loader)
    if max_b:
        print(f"  Data split      : {args.split}  "
              f"(evaluating {max_b:,}/{n_batches_total:,} batches = "
              f"{max_b * args.batch_size:,} puzzles)")
    else:
        print(f"  Data split      : {args.split}  "
              f"({n_batches_total:,} batches of {args.batch_size} = "
              f"{n_batches_total * args.batch_size:,} puzzles)")

    # ── Run evaluation(s) ─────────────────────────────────────
    results: dict = {}

    do_no_tta = not args.tta_only
    do_tta = not args.no_tta

    if do_no_tta:
        print(f"\n{'═' * 68}")
        print(f"  Evaluation: no-TTA")
        print(f"{'═' * 68}")
        t0 = time.time()
        results["no_tta"] = eval_sudoku_progress(
            model, loader, device,
            pred_T=args.pred_T,
            use_tta=False, n_tta=0,
            residual_alpha=args.residual_alpha,
            max_batches=max_b,
            desc="no-TTA",
            log_every=args.log_every,
        )
        print(f"  no-TTA time     : {(time.time() - t0) / 60:.1f} min")
        _print_row("no-TTA", results["no_tta"])

    if do_tta:
        print(f"\n{'═' * 68}")
        print(f"  Evaluation: TTA×{args.n_tta}  "
              f"(~{1 + args.n_tta}× slower than no-TTA)")
        print(f"{'═' * 68}")
        t0 = time.time()
        results["tta"] = eval_sudoku_progress(
            model, loader, device,
            pred_T=args.pred_T,
            use_tta=True, n_tta=args.n_tta,
            residual_alpha=args.residual_alpha,
            max_batches=max_b,
            desc=f"TTA×{args.n_tta}",
            log_every=args.log_every,
        )
        print(f"  TTA time        : {(time.time() - t0) / 60:.1f} min")
        _print_row(f"TTA×{args.n_tta}", results["tta"])

    # ── Headline summary ──────────────────────────────────────
    print("\n" + "=" * 68)
    print("  SUDOKU EXTREME — HEADLINE NUMBERS")
    print("=" * 68)
    if do_no_tta:
        _print_row("no-TTA", results["no_tta"])
    if do_tta:
        _print_row(f"TTA×{args.n_tta}", results["tta"])
    if do_no_tta and do_tta:
        lift = (results["tta"]["grid_solved"]
                - results["no_tta"]["grid_solved"])
        print(f"\n  TTA lift on grid_solved: {lift:+.4f}")
    print("=" * 68)

    # ── Save JSON ─────────────────────────────────────────────
    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)
        clean: dict = {}
        for mode, res in results.items():
            clean[mode] = {k: v for k, v in res.items()
                           if not k.startswith("_")}
            clean[mode]["n_samples"] = int(res["_n_samples"])
        clean["config"] = {
            "ckpt": args.ckpt,
            "pred_T": args.pred_T,
            "n_tta": args.n_tta,
            "residual_alpha": args.residual_alpha,
            "n_supports": args.n_supports,
            "batch_size": args.batch_size,
            "max_batches": max_b,
        }
        with open(args.save_json, "w") as f:
            json.dump(clean, f, indent=2)
        print(f"\n  Saved results to {args.save_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())