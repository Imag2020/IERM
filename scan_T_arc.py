"""
scripts/scan_T_arc.py
─────────────────────
Scan the number of latent-recursion steps T for ARC no-TTA evaluation.

Runs the official task-level eval for T ∈ {1, 2, 3, 4, 5, 6, 7, 8}
with no-TTA (single forward pass per query), and prints a summary
table to help pick the best T for the paper.

Usage
─────
    python scripts/scan_T_arc.py \\
        --ckpt     model_checkpoints/checkpoints_support_only_ft_lastshot/best_ema.pt \\
        --data_dir data/arc_agi1

Expected runtime: ~10-15 min on GPU (scanning 8 values of T).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

from arc.model import build_arc_model, load_arc_checkpoint, CFG_ARC
from arc.data import build_eval_dataset
from arc.eval import evaluate_arc_official_from_json


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ckpt",
        default="model_checkpoints/checkpoints_support_only_ft_lastshot/best_ema.pt")
    p.add_argument("--data_dir", default="data/arc_agi1")
    p.add_argument("--T_values", type=int, nargs="+",
                   default=[1, 2, 3, 4, 5, 6, 7, 8])
    p.add_argument("--save_json", default="results/arc_T_scan.json")
    p.add_argument("--device", default=None)
    p.add_argument("--max_tasks", type=int, default=None,
                   help="Cap tasks (debug, default: all 400).")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"  Device     : {device}")
    print(f"  Checkpoint : {args.ckpt}")
    print(f"  T values   : {args.T_values}")

    # Build once.
    print(f"\n  Building model + loading checkpoint ...")
    model = build_arc_model(CFG_ARC).to(device)
    _ = load_arc_checkpoint(model, args.ckpt, device=device, strict=True)
    print(f"  Params     : {sum(p.numel() for p in model.parameters()):,}")

    print(f"  Loading eval set (needs training/ for eval_offset) ...")
    ds_eval, stable_to_task_id, _ = build_eval_dataset(
        data_root=args.data_dir,
        eos_id=CFG_ARC["eos_id"], pad_id=CFG_ARC["pad_id"])
    print(f"  Eval tasks : {len(ds_eval)}")

    eval_dir = os.path.join(args.data_dir, "evaluation")

    results_by_T = {}
    print(f"\n{'═' * 60}")
    print(f"  T-SCAN : no-TTA, single forward pass per query")
    print(f"{'═' * 60}\n")

    for T in args.T_values:
        print(f"  ─── T={T} ───────────────────────────────────────────")
        t0 = time.time()
        res = evaluate_arc_official_from_json(
            model, eval_dir,
            stable_to_task_id=stable_to_task_id,
            device=device,
            H=CFG_ARC["Hmax"], W=CFG_ARC["Wmax"],
            eos_id=CFG_ARC["eos_id"], pad_id=CFG_ARC["pad_id"],
            use_tta=False,
            T_max=T,
            support_cap=None,
            support_strategy="first",
            max_tasks=args.max_tasks,
            progress=False, verbose=False,
        )
        elapsed = time.time() - t0
        n_solved = int(res["official_task_solved"] * res["n_tasks"])
        print(f"    solved = {res['official_task_solved']:.4f}  "
              f"({n_solved}/{res['n_tasks']})  "
              f"queries = {res['query_accuracy']:.4f}  "
              f"[{elapsed:.1f}s]")

        results_by_T[T] = {
            "T": T,
            "n_tasks": res["n_tasks"],
            "n_queries": res["n_queries"],
            "official_task_solved": res["official_task_solved"],
            "query_accuracy": res["query_accuracy"],
            "n_tasks_solved": n_solved,
            "time_sec": elapsed,
        }

    # Summary.
    best_T = max(results_by_T.keys(),
                 key=lambda t: results_by_T[t]["official_task_solved"])
    print(f"\n{'═' * 60}")
    print(f"  SUMMARY — no-TTA")
    print(f"{'═' * 60}")
    print(f"  {'T':>3} | {'solved':>8} | {'queries':>8} | {'n_solved':>8} | {'time':>6}")
    print(f"  {'-' * 3}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 6}")
    for T in args.T_values:
        r = results_by_T[T]
        marker = " ←" if T == best_T else ""
        print(f"  {T:>3} | {r['official_task_solved']:>8.4f} | "
              f"{r['query_accuracy']:>8.4f} | "
              f"{r['n_tasks_solved']:>3}/{r['n_tasks']:<4} | "
              f"{r['time_sec']:>5.1f}s{marker}")
    print(f"\n  Best T for no-TTA : {best_T}  "
          f"(solved = {results_by_T[best_T]['official_task_solved']:.4f})")

    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump({"results_by_T": results_by_T,
                        "best_T": best_T,
                        "config": {"ckpt": args.ckpt}}, f, indent=2)
        print(f"\n  Saved to {args.save_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())