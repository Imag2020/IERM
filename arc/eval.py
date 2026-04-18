"""
arc/eval.py
───────────
Official ARC-AGI-1 task-level evaluation.

Metric (from ARC-AGI-1 spec)
────────────────────────────
A task is "solved" if and only if the model produces the correct
output grid for ALL test inputs of that task. With 419 queries
distributed across 400 tasks (19 tasks have 2-3 test inputs), the
aggregation is strict:
    task_solved = all(queries_correct_for_this_task)

Reported numbers
────────────────
Two metrics per run:
    official_task_solved : fraction of tasks where all queries are correct
    query_accuracy       : fraction of test queries correctly predicted

Example usage
─────────────
    >>> results = evaluate_arc_official_from_json(
    ...     model, "data/arc_agi1/evaluation",
    ...     stable_to_task_id=map_from_build_eval_dataset,
    ...     use_tta=True, T_max=6,
    ...     n_color_perms=1, use_geometric=True, voting="confidence",
    ... )
    >>> print(results["official_task_solved"])   # → 0.1225 with TTA
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import torch

from .data import stable_task_id_from_filename
from .tta import tta_predict


# ══════════════════════════════════════════════════════════════
# 1. Grid-level helpers (task-level evaluation operates on
#    content crops, not padded 30×30 tensors)
# ══════════════════════════════════════════════════════════════

def arc_grid_to_padded(grid, H: int = 30, W: int = 30,
                        eos_id: int = 10, pad_id: int = 11,
                        device: str = "cpu") -> torch.Tensor:
    """Encode a small grid into [H, W] with L-shape EOS border."""
    if not isinstance(grid, torch.Tensor):
        grid = torch.tensor(grid, dtype=torch.long, device=device)
    else:
        grid = grid.to(device=device, dtype=torch.long)

    h, w = grid.shape
    out = torch.full((H, W), pad_id, dtype=torch.long, device=device)
    out[:h, :w] = grid
    if w < W:
        out[:h, w] = eos_id
    if h < H:
        out[h, :w] = eos_id
        if w < W:
            out[h, w] = eos_id
    return out


def extract_content_from_padded(grid: torch.Tensor,
                                 eos_id: int = 10,
                                 pad_id: int = 11) -> torch.Tensor:
    """Recover the h×w content from an L-shape padded grid."""
    H, W = grid.shape
    h = H
    for r in range(H):
        if grid[r, 0].item() == eos_id or bool((grid[r] == pad_id).all()):
            h = r
            break
    w = W
    for c in range(W):
        if grid[0, c].item() == eos_id or bool((grid[:, c] == pad_id).all()):
            w = c
            break
    return grid[:h, :w].clone()


def exact_grid_match(pred_content: torch.Tensor, gt_grid) -> bool:
    """True iff predicted content matches ground truth exactly (shape + values)."""
    if not isinstance(gt_grid, torch.Tensor):
        gt_grid = torch.tensor(
            gt_grid, dtype=torch.long, device=pred_content.device)
    else:
        gt_grid = gt_grid.to(
            device=pred_content.device, dtype=torch.long)
    return (pred_content.shape == gt_grid.shape
            and torch.equal(pred_content.long(), gt_grid.long()))


# ══════════════════════════════════════════════════════════════
# 2. Support selection
# ══════════════════════════════════════════════════════════════

def select_supports(train_pairs: List[Dict],
                     support_cap: Optional[int] = None,
                     strategy: str = "first") -> List[Dict]:
    """
    Choose `support_cap` supports from a task's train pairs.

    Strategies:
      * "first"  : the first `support_cap` pairs (default; matches
                    the run that produced 12.25%).
      * "last"   : the last `support_cap` pairs.
      * "spread" : evenly spaced indices over the full range.

    Observation from the paper: 4, 6, and "all" supports produce the
    same 12.25% on this checkpoint, suggesting the model saturates
    around 4 supports. We keep `support_cap=None` (all) as default.
    """
    if support_cap is None or support_cap >= len(train_pairs):
        return train_pairs

    n = len(train_pairs)
    k = int(support_cap)

    if strategy == "first":
        idx = list(range(k))
    elif strategy == "last":
        idx = list(range(n - k, n))
    elif strategy == "spread":
        if k == 1:
            idx = [0]
        else:
            idx = sorted(set(
                round(i * (n - 1) / (k - 1)) for i in range(k)))
            while len(idx) < k:
                for j in range(n):
                    if j not in idx:
                        idx.append(j)
                    if len(idx) == k:
                        break
            idx = sorted(idx[:k])
    else:
        raise ValueError(f"Unknown support selection strategy: {strategy!r}")

    return [train_pairs[i] for i in idx]


# ══════════════════════════════════════════════════════════════
# 3. Build per-task tensors from a raw JSON
# ══════════════════════════════════════════════════════════════

def build_arc_task_tensors(task_json: Dict, *,
                            H: int = 30, W: int = 30,
                            eos_id: int = 10, pad_id: int = 11,
                            device: str = "cpu",
                            support_cap: Optional[int] = None,
                            support_strategy: str = "first"):
    """
    Convert a task JSON into tensors ready for model inference.

    Returns:
        sx            : [1, S, H, W] — supports input (padded)
        sy            : [1, S, H, W] — supports output
        s_mask        : [1, S] bool — all True (we use all selected supports)
        test_queries  : list of [H, W] padded tensors, one per test input
        test_outputs  : list of ground-truth grids (as lists or ndarrays)
        n_supports_raw : number of supports in the original task
        n_supports_used : number of supports after capping
    """
    train_pairs = task_json["train"]
    test_pairs = task_json["test"]
    train_sel = select_supports(
        train_pairs, support_cap=support_cap, strategy=support_strategy)

    S = len(train_sel)
    sx = torch.stack([
        arc_grid_to_padded(p["input"], H=H, W=W,
                           eos_id=eos_id, pad_id=pad_id, device=device)
        for p in train_sel
    ], dim=0).unsqueeze(0)
    sy = torch.stack([
        arc_grid_to_padded(p["output"], H=H, W=W,
                           eos_id=eos_id, pad_id=pad_id, device=device)
        for p in train_sel
    ], dim=0).unsqueeze(0)

    s_mask = torch.ones((1, S), dtype=torch.bool, device=device)

    test_queries = [
        arc_grid_to_padded(p["input"], H=H, W=W,
                           eos_id=eos_id, pad_id=pad_id, device=device)
        for p in test_pairs
    ]
    test_outputs = [p["output"] for p in test_pairs]

    return (sx, sy, s_mask, test_queries, test_outputs,
            len(train_pairs), len(train_sel))


# ══════════════════════════════════════════════════════════════
# 4. Per-query prediction (wraps tta_predict or model.predict)
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_arc_query(
    model, sx, sy, qx, *,
    s_mask=None,
    task_id=None,
    use_tta: bool = False,
    T_max: int = 6,
    n_color_perms: int = 1,
    use_geometric: bool = True,
    voting: str = "confidence",
    pad_id: int = 11,
    eos_id: int = 10,
):
    """
    Predict a single ARC query. `qx` is unbatched [H, W].

    Returns:
        pred_content : torch.LongTensor [h_pred, w_pred] — predicted content
        pred_padded  : torch.LongTensor [H, W]           — padded prediction
        info         : dict with {"conf", "n_aug"/"steps", "agreement"?}
    """
    qx_b = qx.unsqueeze(0)  # [1, H, W]

    if use_tta:
        pred, conf, n_aug, agreement = tta_predict(
            model, sx, sy, qx_b,
            s_mask=s_mask,
            task_id=task_id,
            T_max=T_max,
            n_color_perms=n_color_perms,
            use_geometric=use_geometric,
            voting=voting,
            pad_id=pad_id,
            eos_id=eos_id,
        )
        pred_padded = pred[0]
        info = {
            "conf": float(conf),
            "n_aug": int(n_aug),
            "agreement": float(agreement),
        }
    else:
        pred, conf, steps = model.predict(
            sx, sy, qx_b,
            s_mask=s_mask,
            task_id=task_id,
            T_max=T_max,
        )
        pred_padded = pred[0]
        info = {"conf": float(conf), "steps": int(steps)}

    pred_content = extract_content_from_padded(
        pred_padded, eos_id=eos_id, pad_id=pad_id)
    return pred_content, pred_padded, info


# ══════════════════════════════════════════════════════════════
# 5. Main entry point — official task-level evaluation
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_arc_official_from_json(
    model,
    eval_dir,
    *,
    stable_to_task_id: Optional[Dict[int, int]] = None,
    device: str = "cuda",
    H: int = 30, W: int = 30,
    eos_id: int = 10, pad_id: int = 11,
    use_tta: bool = False,
    T_max: int = 6,
    n_color_perms: int = 1,
    use_geometric: bool = True,
    voting: str = "confidence",
    support_cap: Optional[int] = None,
    support_strategy: str = "first",
    max_tasks: Optional[int] = None,
    progress: bool = True,
    verbose: bool = True,
):
    """
    Official ARC-AGI-1 evaluation loop.

    Required: `stable_to_task_id` mapping. Build it with
    `arc.data.build_eval_dataset()` — this ensures the eval task_ids
    match training (eval_offset=400), which is critical for the
    checkpoint's `emb_task` to be correctly indexed.

    If `stable_to_task_id=None`, falls back to `task_id = stable_id`
    modulo the table size, which will likely give a much lower score.
    You'll get a warning.

    Returns a dict with:
        n_tasks, n_queries
        official_task_solved : main metric (paper: 0.1225 with TTA)
        query_accuracy       : per-query accuracy
        avg_supports_{raw,used}
        per_task_results     : list of per-task dicts
    """
    eval_dir = Path(eval_dir)
    json_files = sorted(eval_dir.glob("*.json"))
    if max_tasks is not None:
        json_files = json_files[:int(max_tasks)]
    if not json_files:
        raise FileNotFoundError(f"No *.json in {eval_dir}")

    if stable_to_task_id is None:
        n_emb = int(getattr(
            getattr(model, "embedder", None), "n_tasks", 2048))
        print(f"  ⚠ WARNING: no stable_to_task_id mapping provided.")
        print(f"             Using `task_id = stable_id % {n_emb}` as fallback.")
        print(f"             This MAY significantly degrade the score.")
        print(f"             Build the mapping with arc.data.build_eval_dataset().")

        def resolve_task_id(sid: int) -> int:
            return int(sid) % n_emb
    else:
        def resolve_task_id(sid: int) -> int:
            if int(sid) not in stable_to_task_id:
                raise KeyError(
                    f"stable_id {sid} not in stable_to_task_id mapping. "
                    f"Did you pass the correct eval split?")
            return stable_to_task_id[int(sid)]

    model.eval()
    totals = defaultdict(float)
    per_task_results = []

    # Optional tqdm.
    if progress:
        try:
            from tqdm.auto import tqdm
            bar = tqdm(json_files, desc="ARC eval", dynamic_ncols=True,
                        mininterval=0.5)
        except ImportError:
            bar = json_files
    else:
        bar = json_files

    for task_idx, jf in enumerate(bar):
        with open(jf, "r") as f:
            task_json = json.load(f)

        sx, sy, s_mask, test_queries, test_outputs, n_sup_raw, n_sup_used = (
            build_arc_task_tensors(
                task_json, H=H, W=W,
                eos_id=eos_id, pad_id=pad_id, device=device,
                support_cap=support_cap,
                support_strategy=support_strategy))

        stable_id = stable_task_id_from_filename(jf.name)
        tid_int = resolve_task_id(stable_id)
        task_id = torch.tensor([tid_int], device=device, dtype=torch.long)

        query_correct_flags = []
        for qx, gt_out in zip(test_queries, test_outputs):
            pred_content, _, _ = predict_arc_query(
                model, sx, sy, qx,
                s_mask=s_mask, task_id=task_id,
                use_tta=use_tta, T_max=T_max,
                n_color_perms=n_color_perms,
                use_geometric=use_geometric,
                voting=voting,
                pad_id=pad_id, eos_id=eos_id)

            ok = exact_grid_match(pred_content, gt_out)
            query_correct_flags.append(ok)
            totals["n_queries"] += 1
            totals["query_correct"] += float(ok)

        task_solved = all(query_correct_flags)
        totals["n_tasks"] += 1
        totals["task_solved"] += float(task_solved)
        totals["supports_raw_sum"] += float(n_sup_raw)
        totals["supports_used_sum"] += float(n_sup_used)

        per_task_results.append({
            "task_file": jf.name,
            "stable_id": int(stable_id),
            "task_id": int(tid_int),
            "n_test_queries": len(test_queries),
            "task_solved": bool(task_solved),
            "n_supports_raw": int(n_sup_raw),
            "n_supports_used": int(n_sup_used),
            "query_correct_flags": [bool(x) for x in query_correct_flags],
        })

        if hasattr(bar, "set_postfix_str"):
            cur = totals["task_solved"] / max(totals["n_tasks"], 1)
            bar.set_postfix_str(f"solved={cur:.4f}")

    if hasattr(bar, "close"):
        bar.close()

    n_tasks = max(int(totals["n_tasks"]), 1)
    n_queries = max(int(totals["n_queries"]), 1)
    results = {
        "n_tasks": int(totals["n_tasks"]),
        "n_queries": int(totals["n_queries"]),
        "official_task_solved": float(totals["task_solved"] / n_tasks),
        "query_accuracy": float(totals["query_correct"] / n_queries),
        "avg_supports_raw": float(totals["supports_raw_sum"] / n_tasks),
        "avg_supports_used": float(totals["supports_used_sum"] / n_tasks),
        "per_task_results": per_task_results,
    }

    # Multi-query diagnostic.
    multi_q = [r for r in per_task_results if r["n_test_queries"] > 1]
    single_q = [r for r in per_task_results if r["n_test_queries"] == 1]
    n_multi_solved = sum(1 for r in multi_q if r["task_solved"])
    n_single_solved = sum(1 for r in single_q if r["task_solved"])
    results["n_multi_query_tasks"] = len(multi_q)
    results["n_multi_query_solved"] = int(n_multi_solved)
    results["n_single_query_tasks"] = len(single_q)
    results["n_single_query_solved"] = int(n_single_solved)

    if verbose:
        print()
        print("═" * 70)
        tag = "TTA" if use_tta else "no-TTA"
        print(f"OFFICIAL ARC-AGI-1 EVAL ({tag}, all test queries)")
        print("═" * 70)
        print(f"  Tasks                : {results['n_tasks']}")
        print(f"  Queries              : {results['n_queries']}")
        print(f"  Official task solved : {results['official_task_solved']:.4f}")
        print(f"  Query accuracy       : {results['query_accuracy']:.4f}")
        if len(multi_q) > 0:
            print(f"  Multi-query tasks    : "
                  f"{n_multi_solved}/{len(multi_q)} "
                  f"({n_multi_solved / len(multi_q):.4f})")
            print(f"  Single-query tasks   : "
                  f"{n_single_solved}/{len(single_q)} "
                  f"({n_single_solved / len(single_q):.4f})")
        print(f"  Avg supports         : {results['avg_supports_raw']:.2f}")
        if use_tta:
            n_views = (8 if use_geometric else 1) * max(n_color_perms, 1)
            print(f"  TTA config           : "
                  f"D4={use_geometric}, color_perms={n_color_perms}, "
                  f"voting={voting}, T={T_max}, n_views={n_views}")
        print("═" * 70)

    return results


__all__ = [
    "arc_grid_to_padded",
    "extract_content_from_padded",
    "exact_grid_match",
    "select_supports",
    "build_arc_task_tensors",
    "predict_arc_query",
    "evaluate_arc_official_from_json",
]