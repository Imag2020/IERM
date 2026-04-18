"""
arc/data.py
───────────
ARC-AGI-1 dataset loading — SELF-CONTAINED.

Only what's needed for evaluation: task parsing, 30×30 encoding with
L-shape EOS border, task_id assignment (with eval_offset=400 to match
training).

Directory layout expected
─────────────────────────
    data/arc_agi1/
        training/    (400 JSON files)
        evaluation/  (400 JSON files)

CRITICAL — task_id assignment
─────────────────────────────
The shipped checkpoint was trained with task_id = 0..399 for training
tasks and 400..799 for evaluation tasks. `emb_task(2048)` was learned
with this exact mapping. **Any eval script MUST reproduce this offset,
even when evaluating only on the eval split** — otherwise the model
receives wrong task embeddings and the score drops significantly.

This means `load_split_tasks("training/")` must be called alongside
`load_split_tasks("evaluation/")` so that we can compute
`eval_offset = len(train_raw) = 400` and assign the eval task_ids
accordingly. This module handles that automatically in `build_eval_dataset`.

Public API
──────────
    load_split_tasks(dir_path)              # raw dicts
    assign_split_task_ids_with_eval_offset(train_raw, eval_raw, offset=None)
    encode_bbox_with_eos_to_30(grid, pad_id, eos_id)
    ARCVariableSupportDataset(tasks, cfg, mode)
    build_eval_dataset(data_root)           # returns (ds_eval, stable_to_task_id)
    ArcConfig, ParsedTask                   # lightweight dataclasses
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# ══════════════════════════════════════════════════════════════
# 1. Config and task dataclass
# ══════════════════════════════════════════════════════════════

@dataclass
class ArcConfig:
    eos_id: int = 10
    pad_id: int = 11
    rng_seed: int = 0


@dataclass
class ParsedTask:
    """One ARC task with stable/integer identifiers and all pairs."""
    stable_id: int
    task_id: int
    supports_in: List[np.ndarray]
    supports_out: List[np.ndarray]
    tests_in: List[np.ndarray]
    tests_out: List[np.ndarray]


# ══════════════════════════════════════════════════════════════
# 2. Parse helpers
# ══════════════════════════════════════════════════════════════

def stable_task_id_from_filename(fn: str) -> int:
    """
    Deterministic integer ID from a filename.

    ARC task filenames are 8-char hex strings (e.g., "007bbfb7.json").
    We try hex parsing first, and fall back to SHA1 truncation for
    non-hex names. The result is stable across runs and machines.
    """
    base = os.path.splitext(os.path.basename(fn))[0]
    try:
        return int(base, 16)
    except ValueError:
        return int(hashlib.sha1(base.encode()).hexdigest()[:8], 16)


def _to_np_grid(grid_ll: List[List[int]]) -> np.ndarray:
    return np.array(grid_ll, dtype=np.uint8)


def load_split_tasks(dir_path: str) -> List[Dict]:
    """
    Load all JSON tasks from a directory.

    Returns a list of dicts with keys:
      stable_id : int
      sx, sy    : List[np.ndarray]  (supports input / output)
      qx, qy    : List[np.ndarray]  (test queries input / output)
    """
    files = sorted(
        os.path.join(dir_path, f)
        for f in os.listdir(dir_path)
        if f.endswith(".json"))
    if not files:
        raise FileNotFoundError(
            f"No *.json files in {dir_path!r}. "
            f"Expected ARC-AGI-1 training/evaluation splits.")

    out = []
    for p in files:
        with open(p, "r") as fh:
            d = json.load(fh)
        out.append({
            "stable_id": stable_task_id_from_filename(p),
            "sx": [_to_np_grid(pair["input"]) for pair in d["train"]],
            "sy": [_to_np_grid(pair["output"]) for pair in d["train"]],
            "qx": [_to_np_grid(pair["input"]) for pair in d["test"]],
            "qy": [_to_np_grid(pair["output"]) for pair in d["test"]],
        })
    return out


def assign_split_task_ids_with_eval_offset(
    train_raw: List[Dict],
    eval_raw: List[Dict],
    *,
    eval_offset: Optional[int] = None,
) -> Tuple[List[ParsedTask], List[ParsedTask], int]:
    """
    Assign integer task_ids that MATCH the training-time assignment.

    Training tasks get ids [0 .. len(train_raw) - 1].
    Eval tasks get ids [eval_offset .. eval_offset + len(eval_raw) - 1],
    where eval_offset defaults to len(train_raw) (= 400 for ARC-AGI-1).

    This is critical: the `emb_task(2048)` embedding table in the
    checkpoint was indexed with this exact scheme. Using the wrong
    offset at inference gives the model incorrect task embeddings.

    Returns:
        (train_tasks, eval_tasks, n_total_ids)
    """
    if eval_offset is None:
        eval_offset = len(train_raw)

    def _build(raw: List[Dict], base: int) -> List[ParsedTask]:
        return [
            ParsedTask(
                stable_id=t["stable_id"],
                task_id=base + i,
                supports_in=t["sx"],
                supports_out=t["sy"],
                tests_in=t["qx"],
                tests_out=t["qy"],
            )
            for i, t in enumerate(raw)
        ]

    train_tasks = _build(train_raw, 0)
    eval_tasks = _build(eval_raw, eval_offset)
    return train_tasks, eval_tasks, eval_offset + len(eval_raw)


# ══════════════════════════════════════════════════════════════
# 3. Grid encoding: (h×w content) → 30×30 with L-shape EOS
# ══════════════════════════════════════════════════════════════

def encode_bbox_with_eos_to_30(
    grid: np.ndarray,
    *,
    pad_id: int,
    eos_id: int,
    H: int = 30,
    W: int = 30,
) -> np.ndarray:
    """
    Encode a small grid (h × w, values in 0..9) into H×W (default 30×30).

    Convention
    ──────────
      - Content anchored at (0, 0)
      - L-shape EOS border after the content
          * Column w  (for rows 0..h-1)      if w < W
          * Row h     (for cols 0..w)        if h < H
          * Corner (h, w)                    if both fit
      - Everything else is pad_id
    """
    h, w = grid.shape
    assert h <= H and w <= W, f"Grid {h}×{w} too large for {H}×{W}"
    canvas = np.full((H, W), pad_id, dtype=np.uint8)
    if h == 0 or w == 0:
        return canvas

    canvas[:h, :w] = grid

    if w < W:
        canvas[:h, w] = eos_id          # right EOS column
    if h < H:
        end_c = min(w + (1 if w < W else 0), W)
        if end_c > 0:
            canvas[h, :end_c] = eos_id  # bottom EOS row (incl. corner)

    return canvas


# ══════════════════════════════════════════════════════════════
# 4. Dataset
# ══════════════════════════════════════════════════════════════

class ARCVariableSupportDataset(Dataset):
    """
    One task = one item. Supports have variable count S per task.

    Modes
    ─────
      MODE_TRAIN_OR_EVAL_QUERY (0): query = tests_in[0], tests_out[0]
          Used during training (supervised on test pair) and during
          standard eval (single query per task).
      MODE_EVAL_AS_SUPPORT_QUERY (1): query = a randomly held-out support
          Used for pre-training on train-split tasks, reusing them as
          self-queries.

    Returns a dict with (among others):
      sx, sy  : LongTensor [S, 30, 30]
      qx, qy  : LongTensor [30, 30]
      task_id : LongTensor scalar (matches emb_task index)
    """

    MODE_TRAIN_OR_EVAL_QUERY = 0
    MODE_EVAL_AS_SUPPORT_QUERY = 1

    def __init__(self, tasks: List[ParsedTask], *, cfg: ArcConfig, mode: int):
        self.tasks = tasks
        self.cfg = cfg
        self.mode = int(mode)
        for t in self.tasks:
            assert len(t.supports_in) >= 1, f"Task {t.task_id} has no supports"

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx: int):
        t = self.tasks[idx]
        pad_id, eos_id = self.cfg.pad_id, self.cfg.eos_id
        q_idx = -1

        if self.mode == self.MODE_EVAL_AS_SUPPORT_QUERY:
            S_raw = len(t.supports_in)
            q_idx = random.randrange(S_raw)
            q_in, q_out = t.supports_in[q_idx], t.supports_out[q_idx]
            keep = [i for i in range(S_raw) if i != q_idx]
            s_in = [t.supports_in[i] for i in keep] or [q_in]
            s_out = [t.supports_out[i] for i in keep] or [q_out]
        else:
            assert len(t.tests_in) >= 1, f"Task {t.task_id} has no tests"
            q_in, q_out = t.tests_in[0], t.tests_out[0]
            s_in, s_out = t.supports_in, t.supports_out

        def enc(g):
            return encode_bbox_with_eos_to_30(g, pad_id=pad_id, eos_id=eos_id)

        S = len(s_in)
        return {
            "sx":          torch.from_numpy(np.stack([enc(g) for g in s_in])),
            "sy":          torch.from_numpy(np.stack([enc(g) for g in s_out])),
            "qx":          torch.from_numpy(enc(q_in)),
            "qy":          torch.from_numpy(enc(q_out)),
            "task_id":     torch.tensor(t.task_id, dtype=torch.int64),
            "task_stable": torch.tensor(t.stable_id, dtype=torch.int64),
            "S":           torch.tensor(S, dtype=torch.int64),
            "S_raw":       torch.tensor(len(t.supports_in), dtype=torch.int64),
            "q_idx":       torch.tensor(q_idx, dtype=torch.int64),
            "mode":        torch.tensor(self.mode, dtype=torch.int64),
            "pad_id":      torch.tensor(pad_id, dtype=torch.int64),
            "eos_id":      torch.tensor(eos_id, dtype=torch.int64),
        }


# ══════════════════════════════════════════════════════════════
# 5. High-level builder — this is what eval/repro scripts call
# ══════════════════════════════════════════════════════════════

def build_eval_dataset(
    data_root: str = "data/arc_agi1",
    *,
    eos_id: int = 10,
    pad_id: int = 11,
) -> Tuple[ARCVariableSupportDataset, Dict[int, int], int]:
    """
    Build the eval dataset with the CORRECT task_id offset.

    Why this function exists
    ────────────────────────
    The checkpoint was trained with task_ids [0..399] for training
    tasks and [400..799] for eval tasks. When evaluating, we MUST
    reproduce this offset — otherwise emb_task is misaligned.
    This means loading BOTH training/ and evaluation/ splits, even
    though we only evaluate on eval. This function does that.

    Args:
        data_root : path to `data/arc_agi1` (contains training/, evaluation/)
        eos_id, pad_id : special-token ids (defaults match the paper)

    Returns:
        (ds_eval, stable_to_task_id, n_total_ids)
          ds_eval            : ARCVariableSupportDataset over the 400 eval tasks
          stable_to_task_id  : dict mapping stable_id → task_id (for JSON eval)
          n_total_ids        : total number of distinct ids assigned
    """
    root = Path(data_root).expanduser()
    train_dir = root / "training"
    eval_dir = root / "evaluation"

    if not train_dir.exists():
        raise FileNotFoundError(
            f"Missing training dir: {train_dir}. "
            f"ARC-AGI-1 training set is REQUIRED even for eval, "
            f"so we can compute the correct eval_offset (=400).")
    if not eval_dir.exists():
        raise FileNotFoundError(f"Missing evaluation dir: {eval_dir}")

    train_raw = load_split_tasks(str(train_dir))
    eval_raw = load_split_tasks(str(eval_dir))
    train_tasks, eval_tasks, n_total = (
        assign_split_task_ids_with_eval_offset(train_raw, eval_raw))

    cfg = ArcConfig(eos_id=eos_id, pad_id=pad_id, rng_seed=0)
    ds_eval = ARCVariableSupportDataset(
        eval_tasks, cfg=cfg,
        mode=ARCVariableSupportDataset.MODE_TRAIN_OR_EVAL_QUERY)

    stable_to_task_id = {int(t.stable_id): int(t.task_id) for t in eval_tasks}

    return ds_eval, stable_to_task_id, n_total


__all__ = [
    "ArcConfig",
    "ParsedTask",
    "stable_task_id_from_filename",
    "load_split_tasks",
    "assign_split_task_ids_with_eval_offset",
    "encode_bbox_with_eos_to_30",
    "ARCVariableSupportDataset",
    "build_eval_dataset",
]