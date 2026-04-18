"""
arc/repro.py
────────────
Reproduce the ARC-AGI-1 headline numbers from the paper.

Evaluates the shipped checkpoint on the 400 public-evaluation tasks
using the official task-level metric (all test inputs must be
correctly predicted for the task to count as solved).

Usage
─────
    # Full eval: no-TTA + TTA (8 views, voting=confidence, T=6)
    python -m arc.repro \\
        --ckpt      model_checkpoints/checkpoints_support_only_ft_lastshot/best_ema.pt \\
        --data_dir  data/arc_agi1 \\
        --save_json results/arc_eval.json

    # Only no-TTA (faster, ~1 min):
    python -m arc.repro --ckpt ... --no_tta

    # Only TTA:
    python -m arc.repro --ckpt ... --tta_only

Expected numbers
────────────────
    no-TTA   : 0.1075 task-solved (10.75%)
    TTA×8    : 0.1225 task-solved (12.25%)  ← headline
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


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Reproduce IERM ARC-AGI-1 numbers.")
    p.add_argument(
        "--ckpt",
        default="model_checkpoints/checkpoints_support_only_ft_lastshot/best_ema.pt",
        help="Path to the ARC checkpoint.")
    p.add_argument(
        "--data_dir",
        default="data/arc_agi1",
        help="Root containing `training/` and `evaluation/` (both needed "
             "so eval task_ids get the correct offset, even though we "
             "only evaluate on `evaluation/`).")
    p.add_argument("--pred_T", type=int, default=6,
                   help="Latent-recursion steps (fine-tuned up to T=6).")
    p.add_argument("--n_color_perms", type=int, default=1,
                   help="Color-perm variants per geometric view.")
    p.add_argument("--use_geometric", action="store_true", default=True)
    p.add_argument("--no_geometric", dest="use_geometric",
                   action="store_false")
    p.add_argument("--voting", default="confidence",
                   choices=("confidence", "majority"))
    p.add_argument("--support_cap", type=int, default=None,
                   help="Cap supports to this many (None = all).")
    p.add_argument("--support_strategy", default="first",
                   choices=("first", "last", "spread"))
    p.add_argument("--max_tasks", type=int, default=None,
                   help="Evaluate only first N tasks (debug).")
    p.add_argument("--save_json", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--no_tta", action="store_true",
                   help="Skip the TTA pass (no-TTA only).")
    p.add_argument("--tta_only", action="store_true",
                   help="Skip the no-TTA pass (TTA only).")
    return p.parse_args()


def _print_row(tag: str, r: dict) -> None:
    print(f"  {tag:12s} | solved = {r['official_task_solved']:.4f}  "
          f"queries = {r['query_accuracy']:.4f}  "
          f"({int(r['n_queries'] * r['query_accuracy'])}/"
          f"{r['n_queries']} correct)")


def main() -> int:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"  Device            : {device}")
    print(f"  Checkpoint        : {args.ckpt}")
    print(f"  Data root         : {args.data_dir}")
    print(f"  pred_T            : {args.pred_T}")
    print(f"  n_color_perms     : {args.n_color_perms}")
    print(f"  use_geometric     : {args.use_geometric}")
    print(f"  voting            : {args.voting}")
    print(f"  support_cap       : {args.support_cap}")

    if not os.path.exists(args.ckpt):
        print(f"\n✗ Checkpoint not found: {args.ckpt}", file=sys.stderr)
        return 1

    # ── Build model + load checkpoint ─────────────────────────
    print(f"\n  Building model ...")
    model = build_arc_model(CFG_ARC).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters        : {n_params:,}")

    print(f"  Loading checkpoint ...")
    ckpt = load_arc_checkpoint(
        model, args.ckpt, device=device, strict=True)
    print(f"  Checkpoint step   : {ckpt.get('step', 'n/a')}")

    # ── Build eval dataset with correct task_id offset ─────────
    print(f"\n  Loading eval set ...")
    t0 = time.time()
    ds_eval, stable_to_task_id, n_total = build_eval_dataset(
        data_root=args.data_dir,
        eos_id=CFG_ARC["eos_id"], pad_id=CFG_ARC["pad_id"])
    print(f"  Eval tasks        : {len(ds_eval)}")
    print(f"  task_id range     : "
          f"[{min(stable_to_task_id.values())}, "
          f"{max(stable_to_task_id.values())}]  "
          f"(eval_offset confirmed at "
          f"{min(stable_to_task_id.values())})")
    print(f"  Data load time    : {time.time() - t0:.1f}s")

    # ── Run evaluations ────────────────────────────────────────
    eval_dir = os.path.join(args.data_dir, "evaluation")
    results: dict = {}

    do_no_tta = not args.tta_only
    do_tta = not args.no_tta

    if do_no_tta:
        print(f"\n{'═' * 68}")
        print(f"  Running: no-TTA  (T={args.pred_T})")
        print(f"{'═' * 68}")
        t0 = time.time()
        results["no_tta"] = evaluate_arc_official_from_json(
            model, eval_dir,
            stable_to_task_id=stable_to_task_id,
            device=device,
            H=CFG_ARC["Hmax"], W=CFG_ARC["Wmax"],
            eos_id=CFG_ARC["eos_id"], pad_id=CFG_ARC["pad_id"],
            use_tta=False,
            T_max=args.pred_T,
            support_cap=args.support_cap,
            support_strategy=args.support_strategy,
            max_tasks=args.max_tasks,
            progress=True, verbose=True,
        )
        print(f"  no-TTA time       : {(time.time() - t0) / 60:.1f} min")

    if do_tta:
        n_views = (8 if args.use_geometric else 1) * max(args.n_color_perms, 1)
        print(f"\n{'═' * 68}")
        print(f"  Running: TTA×{n_views}  "
              f"(~{n_views}× slower than no-TTA)")
        print(f"{'═' * 68}")
        t0 = time.time()
        results["tta"] = evaluate_arc_official_from_json(
            model, eval_dir,
            stable_to_task_id=stable_to_task_id,
            device=device,
            H=CFG_ARC["Hmax"], W=CFG_ARC["Wmax"],
            eos_id=CFG_ARC["eos_id"], pad_id=CFG_ARC["pad_id"],
            use_tta=True,
            T_max=args.pred_T,
            n_color_perms=args.n_color_perms,
            use_geometric=args.use_geometric,
            voting=args.voting,
            support_cap=args.support_cap,
            support_strategy=args.support_strategy,
            max_tasks=args.max_tasks,
            progress=True, verbose=True,
        )
        print(f"  TTA time          : {(time.time() - t0) / 60:.1f} min")

    # ── Summary ────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  ARC-AGI-1 PUBLIC EVAL — HEADLINE NUMBERS")
    print("=" * 68)
    if do_no_tta:
        _print_row("no-TTA", results["no_tta"])
    if do_tta:
        n_views = (8 if args.use_geometric else 1) * max(args.n_color_perms, 1)
        _print_row(f"TTA×{n_views}", results["tta"])
    if do_no_tta and do_tta:
        lift = (results["tta"]["official_task_solved"]
                - results["no_tta"]["official_task_solved"])
        print(f"\n  TTA lift on task-solved: {lift:+.4f}")
    print("=" * 68)

    # ── Save JSON ──────────────────────────────────────────────
    if args.save_json:
        os.makedirs(
            os.path.dirname(args.save_json) or ".", exist_ok=True)
        clean: dict = {}
        for mode, res in results.items():
            # Drop the heavy per_task_results unless needed.
            clean[mode] = {
                k: v for k, v in res.items()
                if k != "per_task_results"
            }
            # Keep a compact per-task summary.
            clean[mode]["tasks_solved"] = sorted(
                r["task_file"] for r in res["per_task_results"]
                if r["task_solved"])

        clean["config"] = {
            "ckpt": args.ckpt,
            "pred_T": args.pred_T,
            "n_color_perms": args.n_color_perms,
            "use_geometric": args.use_geometric,
            "voting": args.voting,
            "support_cap": args.support_cap,
            "support_strategy": args.support_strategy,
        }
        with open(args.save_json, "w") as f:
            json.dump(clean, f, indent=2)
        print(f"\n  Saved results to {args.save_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())