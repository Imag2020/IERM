"""
sudoku/data.py
──────────────
Sudoku Extreme dataset, dataloader, and task-level augmentation.

Pipeline
────────
  download_sudoku_data(output_dir, source_repo) -> {train: path, test: path}
      Fetches CSV from HuggingFace `sapientinc/sudoku-extreme` (default).
      Caches as `data/sudoku/{train,test}.npz` with keys:
          inputs : uint8 [N, 9, 9]  (0 = empty, 1..9 = given digit)
          labels : uint8 [N, 9, 9]  (1..9, the complete solution)

  SudokuSupportDataset(inputs, labels, pool_in, pool_out, n_supports=S)
      Each __getitem__ returns a dict with keys sx, sy, qx, qy, task_id.
      Supports are resampled at every access from the pool.

  collate_sudoku(batch, p_task_aug)
      Stacks into [B, S, 9, 9] / [B, 9, 9] tensors and applies
      task-level Sudoku-preserving augmentation with probability
      `p_task_aug` (same transform for all S supports + query).

  build_sudoku_loader(data_dir, split, ...) -> DataLoader
      Thin wrapper used by `sudoku.repro` and training scripts.

  build_all(root, batch_size, ...) -> (datasets..., loaders...)
      Convenience builder for train / eval / trainext loaders.

Augmentations (all Sudoku-preserving)
─────────────────────────────────────
  * digit permutation of 1..9 (0 = empty stays fixed)
  * band permutation (3 row-bands of 3 rows)
  * row permutation within each band
  * stack permutation (3 col-stacks of 3 columns)
  * column permutation within each stack
  * transposition
  * 90°/180°/270° rotations

The same transformation is applied coherently to all (sx_i, sy_i, qx, qy)
of a single task, preserving row/column/block uniqueness.
"""

from __future__ import annotations

import csv
import os
import random
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# ══════════════════════════════════════════════════════════════
# 1. Download / load Sudoku data
# ══════════════════════════════════════════════════════════════

def download_sudoku_data(
    output_dir: str = "data/sudoku",
    source_repo: str = "sapientinc/sudoku-extreme",
    min_difficulty: Optional[int] = None,
) -> Dict[str, str]:
    """
    Download (if needed) and cache the Sudoku Extreme CSVs as .npz.

    Args:
        output_dir     : where to cache `train.npz` / `test.npz`.
        source_repo    : HuggingFace dataset repo.
        min_difficulty : optionally filter out rows with rating < this.

    Returns:
        dict with keys "train" and "test", each mapping to the .npz path.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required to download the Sudoku dataset. "
            "Install with: pip install huggingface_hub"
        ) from e

    os.makedirs(output_dir, exist_ok=True)

    paths: Dict[str, str] = {}
    for split in ("train", "test"):
        npz_path = os.path.join(output_dir, f"{split}.npz")
        if os.path.exists(npz_path):
            paths[split] = npz_path
            continue

        csv_path = hf_hub_download(source_repo, f"{split}.csv",
                                    repo_type="dataset")
        inputs, labels = [], []
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                _source, q, a, rating = row
                if min_difficulty is not None and int(rating) < min_difficulty:
                    continue
                assert len(q) == 81 and len(a) == 81
                inp = np.frombuffer(
                    q.replace(".", "0").encode(), dtype=np.uint8
                ).reshape(9, 9) - ord("0")
                lab = np.frombuffer(
                    a.encode(), dtype=np.uint8
                ).reshape(9, 9) - ord("0")
                inputs.append(inp)
                labels.append(lab)

        inputs_arr = np.stack(inputs).astype(np.uint8)
        labels_arr = np.stack(labels).astype(np.uint8)
        np.savez_compressed(npz_path, inputs=inputs_arr, labels=labels_arr)
        paths[split] = npz_path
        print(f"[Sudoku] {split}: {len(inputs)} puzzles saved -> {npz_path}")

    return paths


def load_sudoku_split(npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a split from disk. Returns (inputs, labels), each [N, 9, 9] uint8."""
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"Missing {npz_path}. Run download_sudoku_data() first, "
            f"or ensure the repo ships pre-generated .npz files "
            f"under data/sudoku/."
        )
    data = np.load(npz_path)
    return data["inputs"], data["labels"]


# ══════════════════════════════════════════════════════════════
# 2. Sudoku-preserving augmentation (task-level)
# ══════════════════════════════════════════════════════════════

def _make_sudoku_aug_params() -> dict:
    """Sample a random set of augmentation parameters for one task."""
    # Digit permutation 1..9 (0 = empty stays fixed).
    digit_perm = np.zeros(10, dtype=np.int64)
    digit_perm[1:] = np.random.permutation(9) + 1

    # Band permutation (3 bands of 3 rows).
    band_perm = np.random.permutation(3)
    row_within = [np.random.permutation(3) for _ in range(3)]
    row_perm = np.concatenate(
        [band_perm[b] * 3 + row_within[b] for b in range(3)])

    # Stack permutation (3 stacks of 3 columns).
    stack_perm = np.random.permutation(3)
    col_within = [np.random.permutation(3) for _ in range(3)]
    col_perm = np.concatenate(
        [stack_perm[s] * 3 + col_within[s] for s in range(3)])

    do_transpose = bool(np.random.rand() < 0.5)
    rot_k = int(np.random.randint(0, 4))

    return {
        "digit_perm": digit_perm,
        "row_perm": row_perm,
        "col_perm": col_perm,
        "do_transpose": do_transpose,
        "rot_k": rot_k,
    }


def _apply_sudoku_aug_single(grid: torch.Tensor, params: dict) -> torch.Tensor:
    """
    Apply augmentation `params` to a single 9×9 grid tensor.

    Order: digit_perm -> transpose -> row_perm -> col_perm -> rotation.
    The order matters; changing it would break coherence with
    `_make_sudoku_aug_params` callers that rely on it implicitly.
    """
    g = grid.clone()

    dp = torch.from_numpy(params["digit_perm"]).to(g.device)
    g = dp[g.long()]

    if params["do_transpose"]:
        g = g.T.contiguous()

    g = g[params["row_perm"], :]
    g = g[:, params["col_perm"]]

    k = params["rot_k"]
    if k > 0:
        g = torch.rot90(g, k=k, dims=(0, 1))

    return g


def augment_sudoku_task(sx: torch.Tensor, sy: torch.Tensor,
                        qx: torch.Tensor, qy: torch.Tensor,
                        *, aug_prob: float = 0.8):
    """
    Augment a SINGLE task coherently.

    Args:
        sx, sy : [S, 9, 9]  (supports input / output)
        qx, qy : [9, 9]     (query input / output)

    Returns:
        sx_aug, sy_aug, qx_aug, qy_aug — same shapes.

    Guarantees
    ----------
      * Same transformation applied to all S supports + the query.
      * Digit 0 (empty) stays 0 after digit permutation.
      * Sudoku structure (rows, columns, 3×3 blocks) preserved.
    """
    if random.random() > aug_prob:
        return sx, sy, qx, qy

    params = _make_sudoku_aug_params()
    S = sx.shape[0]

    sx_aug = torch.stack(
        [_apply_sudoku_aug_single(sx[s], params) for s in range(S)])
    sy_aug = torch.stack(
        [_apply_sudoku_aug_single(sy[s], params) for s in range(S)])
    qx_aug = _apply_sudoku_aug_single(qx, params)
    qy_aug = _apply_sudoku_aug_single(qy, params)
    return sx_aug, sy_aug, qx_aug, qy_aug


def augment_sudoku_batch(sx: torch.Tensor, sy: torch.Tensor,
                          qx: torch.Tensor, qy: torch.Tensor,
                          *, aug_prob: float = 0.8):
    """
    Augment an entire batch — each task gets its own independent transform.

    Args:
        sx, sy : [B, S, 9, 9]
        qx, qy : [B, 9, 9]
    """
    B = sx.shape[0]
    sx_l, sy_l, qx_l, qy_l = [], [], [], []
    for b in range(B):
        a, b2, c, d = augment_sudoku_task(
            sx[b], sy[b], qx[b], qy[b], aug_prob=aug_prob)
        sx_l.append(a)
        sy_l.append(b2)
        qx_l.append(c)
        qy_l.append(d)
    return (torch.stack(sx_l), torch.stack(sy_l),
            torch.stack(qx_l), torch.stack(qy_l))


# ══════════════════════════════════════════════════════════════
# 3. Dataset
# ══════════════════════════════════════════════════════════════

class SudokuSupportDataset(Dataset):
    """
    Few-shot Sudoku dataset.

    One puzzle = one item. Supports (S of them) are drawn uniformly at
    random from `support_pool_{inputs,labels}` at every __getitem__ call,
    so each epoch sees a different random support set for the same query.

    Yields dicts with keys:
      sx      : LongTensor [S, 9, 9]
      sy      : LongTensor [S, 9, 9]
      qx      : LongTensor [9, 9]
      qy      : LongTensor [9, 9]
      task_id : LongTensor scalar (the dataset-index of the query)
    """

    def __init__(
        self,
        inputs: np.ndarray,
        labels: np.ndarray,
        support_pool_inputs: np.ndarray,
        support_pool_labels: np.ndarray,
        *,
        n_supports: int = 4,
    ):
        assert len(inputs) == len(labels)
        assert len(support_pool_inputs) == len(support_pool_labels)
        self.inputs = inputs
        self.labels = labels
        self.pool_in = support_pool_inputs
        self.pool_out = support_pool_labels
        self.n_supports = int(n_supports)
        self.pool_size = len(self.pool_in)

        if self.pool_size < self.n_supports:
            raise ValueError(
                f"Support pool has only {self.pool_size} puzzles, "
                f"but n_supports={self.n_supports}."
            )

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx: int):
        qx = self.inputs[idx]
        qy = self.labels[idx]

        s_idxs = random.sample(range(self.pool_size), self.n_supports)
        sx = np.stack([self.pool_in[i] for i in s_idxs])
        sy = np.stack([self.pool_out[i] for i in s_idxs])

        return {
            "sx": torch.from_numpy(sx).long(),
            "sy": torch.from_numpy(sy).long(),
            "qx": torch.from_numpy(qx.copy()).long(),
            "qy": torch.from_numpy(qy.copy()).long(),
            "task_id": torch.tensor(idx, dtype=torch.int64),
        }


# ══════════════════════════════════════════════════════════════
# 4. Collate
# ══════════════════════════════════════════════════════════════

def collate_sudoku(batch, *, p_task_aug: float = 0.0):
    """
    Collate a list of items into a batched dict.

    Applies task-level augmentation (same transform across supports
    and query) with probability `p_task_aug`.
    """
    sx = torch.stack([b["sx"] for b in batch])
    sy = torch.stack([b["sy"] for b in batch])
    qx = torch.stack([b["qx"] for b in batch])
    qy = torch.stack([b["qy"] for b in batch])
    task_ids = torch.stack([b["task_id"] for b in batch])

    if p_task_aug > 0:
        sx, sy, qx, qy = augment_sudoku_batch(
            sx, sy, qx, qy, aug_prob=p_task_aug)

    return {
        "sx": sx,
        "sy": sy,
        "qx": qx,
        "qy": qy,
        "task_id": task_ids,
    }


# ══════════════════════════════════════════════════════════════
# 5. Single-split loader (used by eval/repro scripts)
# ══════════════════════════════════════════════════════════════

def build_sudoku_loader(
    data_dir: str = "data/sudoku",
    split: str = "test",
    *,
    batch_size: int = 32,
    num_workers: int = 2,
    shuffle: bool = False,
    n_supports: int = 4,
    p_task_aug: float = 0.0,
    pool_split: str = "train",
) -> DataLoader:
    """
    Build a DataLoader for a single split — the interface used by
    `sudoku.repro` and any standalone evaluation script.

    Args:
        data_dir    : directory containing `{split}.npz` and `{pool_split}.npz`.
        split       : which split to iterate over ("test" or "train").
        batch_size  : items per batch.
        num_workers : DataLoader workers.
        shuffle     : whether to shuffle the split.
        n_supports  : number of supports per query. Must match the training
                      configuration for faithful reproduction (we used 4).
        p_task_aug  : task-level augmentation probability (0.0 for eval).
        pool_split  : split used as the support pool. Default "train".

    Returns:
        A torch DataLoader. For eval runs, pass shuffle=False and
        p_task_aug=0.0 to keep the run deterministic w.r.t. augmentation
        (note: support sampling is still stochastic).
    """
    # Validate + load both the query and the pool splits.
    query_path = os.path.join(data_dir, f"{split}.npz")
    pool_path = os.path.join(data_dir, f"{pool_split}.npz")
    q_in, q_out = load_sudoku_split(query_path)
    p_in, p_out = load_sudoku_split(pool_path)

    ds = SudokuSupportDataset(
        q_in, q_out, p_in, p_out, n_supports=n_supports)

    def _collate(b):
        return collate_sudoku(b, p_task_aug=p_task_aug)

    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate,
        pin_memory=torch.cuda.is_available(),
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    return DataLoader(ds, **loader_kwargs)


# ══════════════════════════════════════════════════════════════
# 6. Full build (train / eval / trainext) — used by training scripts
# ══════════════════════════════════════════════════════════════

def build_all(
    root: str = ".",
    *,
    batch_size: int = 16,
    n_supports: int = 4,
    p_task_aug_train: float = 0.8,
    min_difficulty: Optional[int] = None,
    num_workers: int = 4,
):
    """
    Build datasets + loaders for train / eval / trainext.

    `ds_trainext` is the concatenation of train and eval queries, both
    drawing supports from the train pool. Useful to evaluate the model
    on everything it might see in production.

    Returns:
        ds_train, ds_trainext, ds_eval,
        n_total,
        train_loader, eval_loader, trainext_loader
    """
    paths = download_sudoku_data(
        output_dir=os.path.join(root, "data", "sudoku"),
        min_difficulty=min_difficulty)
    train_in, train_out = load_sudoku_split(paths["train"])
    eval_in, eval_out = load_sudoku_split(paths["test"])

    print(f"[Sudoku] train: {len(train_in)}  eval: {len(eval_in)}")

    ds_train = SudokuSupportDataset(
        train_in, train_out, train_in, train_out, n_supports=n_supports)
    ds_eval = SudokuSupportDataset(
        eval_in, eval_out, train_in, train_out, n_supports=n_supports)

    all_in = np.concatenate([train_in, eval_in], axis=0)
    all_out = np.concatenate([train_out, eval_out], axis=0)
    ds_trainext = SudokuSupportDataset(
        all_in, all_out, train_in, train_out, n_supports=n_supports)

    n_total = len(all_in)
    print(f"[Sudoku] ds_train={len(ds_train)}  "
          f"ds_eval={len(ds_eval)}  ds_trainext={len(ds_trainext)}  "
          f"n_total={n_total}")

    loader_kw = dict(pin_memory=torch.cuda.is_available())
    if num_workers > 0:
        loader_kw["persistent_workers"] = True

    train_loader = DataLoader(
        ds_train, batch_size=batch_size, shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda b: collate_sudoku(b, p_task_aug=p_task_aug_train),
        **loader_kw)
    eval_loader = DataLoader(
        ds_eval, batch_size=batch_size, shuffle=False,
        num_workers=max(1, num_workers // 2),
        collate_fn=lambda b: collate_sudoku(b, p_task_aug=0.0),
        **loader_kw)
    trainext_loader = DataLoader(
        ds_trainext, batch_size=batch_size, shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda b: collate_sudoku(b, p_task_aug=p_task_aug_train),
        **loader_kw)

    return (
        ds_train, ds_trainext, ds_eval, n_total,
        train_loader, eval_loader, trainext_loader,
    )


__all__ = [
    # Download / load
    "download_sudoku_data",
    "load_sudoku_split",
    # Augmentation
    "augment_sudoku_task",
    "augment_sudoku_batch",
    # Dataset / collate
    "SudokuSupportDataset",
    "collate_sudoku",
    # Loaders
    "build_sudoku_loader",
    "build_all",
]