import time
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm


import math
import random
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import animation
from typing import List, Tuple, Dict, Optional


import numpy as np, random, math
import torch
import torch.nn.functional as F

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple

from torch.utils.data import DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
device=DEVICE
torch.backends.cuda.matmul.allow_tf32 = True

def count_params(m: nn.Module):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

import torch

def save_checkpoint(path, model):
    torch.save({"model": model.state_dict()}, path)

def load_checkpoint(path, model, device="cuda", strict=True):
    ckpt = torch.load(path, map_location=device)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=strict)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
    if len(missing) > 0: print(" first missing:", missing[:10])
    if len(unexpected) > 0: print(" first unexpected:", unexpected[:10])
    return model

"""
CELLULE 1 — ARC-AGI-1 Dataset + DataLoader (propre, unifié)
============================================================
- ARCVariableSupportDataset : 1 task = 1 item, S variable
- collate_arc_fixedS_v2    : fixe S=4, task-level augmentation cohérente
- 3 loaders : train / eval / trainext

Vocab: 0..9 (couleurs ARC), 10 (EOS), 11 (PAD)  → vocab_size = 12
Grids: 30×30  (bbox ancré en (0,0), EOS en L, PAD ailleurs)

Corrections appliquées vs code original :
  ✅ recolor exclut la couleur 0 (fond) : range(1,10)
  ✅ p_task_aug configurable (défaut 0.5 train, 0.0 eval)
  ✅ persistent_workers + num_workers=4
  ✅ pas de .clone() inutiles quand p_task_aug=0
"""

import os, json, hashlib, random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

# ═══════════════════════════════════════════════════
# 1. Download ARC-AGI-1 JSONs
# ═══════════════════════════════════════════════════
from pathlib import Path

DATA_ROOT = Path("data/arc_agi1").expanduser()
TRAIN_DIR = DATA_ROOT / "training"
EVAL_DIR  = DATA_ROOT / "evaluation"

assert TRAIN_DIR.exists(), f"Missing training dir: {TRAIN_DIR}"
assert EVAL_DIR.exists(), f"Missing evaluation dir: {EVAL_DIR}"



# ═══════════════════════════════════════════════════
# 2. Parse helpers
# ═══════════════════════════════════════════════════
def stable_task_id_from_filename(fn: str) -> int:
    base = os.path.splitext(os.path.basename(fn))[0]
    try:
        return int(base, 16)
    except Exception:
        return int(hashlib.sha1(base.encode()).hexdigest()[:8], 16)


def to_np_grid(grid_ll: List[List[int]]) -> np.ndarray:
    return np.array(grid_ll, dtype=np.uint8)


def load_split_tasks(dir_path: str) -> List[Dict]:
    files = sorted(os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(".json"))
    out = []
    for p in files:
        with open(p, "r") as fh:
            d = json.load(fh)
        sid = stable_task_id_from_filename(p)
        tr, te = d["train"], d["test"]
        out.append({
            "stable_id": sid,
            "sx": [to_np_grid(pair["input"])  for pair in tr],
            "sy": [to_np_grid(pair["output"]) for pair in tr],
            "qx": [to_np_grid(pair["input"])  for pair in te],
            "qy": [to_np_grid(pair["output"]) for pair in te],
        })
    return out


@dataclass
class ParsedTask:
    stable_id: int
    task_id: int
    supports_in:  List[np.ndarray]
    supports_out: List[np.ndarray]
    tests_in:     List[np.ndarray]
    tests_out:    List[np.ndarray]


def assign_split_task_ids_with_eval_offset(
    train_raw: List[Dict], eval_raw: List[Dict], *, eval_offset: Optional[int] = None,
) -> Tuple[List[ParsedTask], List[ParsedTask], int]:
    if eval_offset is None:
        eval_offset = len(train_raw)

    def build(raw, base):
        return [ParsedTask(
            stable_id=t["stable_id"], task_id=base + i,
            supports_in=t["sx"], supports_out=t["sy"],
            tests_in=t["qx"], tests_out=t["qy"],
        ) for i, t in enumerate(raw)]

    train_tasks = build(train_raw, 0)
    eval_tasks  = build(eval_raw, eval_offset)
    return train_tasks, eval_tasks, eval_offset + len(eval_raw)


# ═══════════════════════════════════════════════════
# 3. Encoding: grid → 30×30 with EOS L-shape + PAD
# ═══════════════════════════════════════════════════
def encode_bbox_with_eos_to_30(grid: np.ndarray, *, pad_id: int, eos_id: int) -> np.ndarray:
    """
    grid: (h, w) uint8 values in [0..9]
    → (30, 30) uint8 with bbox anchored at (0,0), EOS L-shape, PAD elsewhere
    """
    h, w = grid.shape
    assert h <= 30 and w <= 30, f"Grid {h}×{w} too large for 30×30"
    if h == 0 or w == 0:
        return np.full((30, 30), pad_id, dtype=np.uint8)

    canvas = np.full((30, 30), pad_id, dtype=np.uint8)
    canvas[:h, :w] = grid

    if w < 30:
        canvas[:h, w] = eos_id                       # EOS colonne droite
    if h < 30:
        end_c = min(w + (1 if w < 30 else 0), 30)    # coin inclus si EOS col existe
        if end_c > 0:
            canvas[h, :end_c] = eos_id                # EOS ligne basse

    return canvas


# ═══════════════════════════════════════════════════
# 4. Config
# ═══════════════════════════════════════════════════
@dataclass
class ArcVarSConfig:
    eos_id: int = 10
    pad_id: int = 11
    rng_seed: int = 0


# ═══════════════════════════════════════════════════
# 5. Dataset: 1 task = 1 item, S variable
# ═══════════════════════════════════════════════════
class ARCVariableSupportDataset(Dataset):
    """
    Retourne:
      sx: (S, 30, 30) uint8       supports input
      sy: (S, 30, 30) uint8       supports output
      qx: (30, 30) uint8          query input
      qy: (30, 30) uint8          query output (GT)
      task_id, task_stable, S, S_raw, q_idx, mode, pad_id, eos_id
    """
    MODE_TRAIN_OR_EVAL_QUERY   = 0   # query = test[0]
    MODE_EVAL_AS_SUPPORT_QUERY = 1   # query = random support, rest = supports

    def __init__(self, tasks: List[ParsedTask], *, cfg: ArcVarSConfig, mode: int):
        self.tasks = tasks
        self.cfg = cfg
        self.mode = int(mode)
        for t in self.tasks:
            assert len(t.supports_in) >= 1, "Task with 0 supports"

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx: int):
        t = self.tasks[idx]
        sup_in, sup_out = t.supports_in, t.supports_out
        S_raw = len(sup_in)
        pad_id, eos_id = self.cfg.pad_id, self.cfg.eos_id
        q_idx = -1

        if self.mode == self.MODE_EVAL_AS_SUPPORT_QUERY:
            q_idx = random.randrange(S_raw)
            q_in, q_out = sup_in[q_idx], sup_out[q_idx]
            keep = [i for i in range(S_raw) if i != q_idx]
            s_in  = [sup_in[i]  for i in keep] or [q_in]
            s_out = [sup_out[i] for i in keep] or [q_out]
        else:
            assert len(t.tests_in) >= 1, "Task with 0 tests"
            q_in, q_out = t.tests_in[0], t.tests_out[0]
            s_in, s_out = sup_in, sup_out

        enc = lambda g: encode_bbox_with_eos_to_30(g, pad_id=pad_id, eos_id=eos_id)
        S = len(s_in)

        return {
            "sx":          torch.from_numpy(np.stack([enc(g) for g in s_in])),   # (S, 30, 30)
            "sy":          torch.from_numpy(np.stack([enc(g) for g in s_out])),
            "qx":          torch.from_numpy(enc(q_in)),
            "qy":          torch.from_numpy(enc(q_out)),
            "task_id":     torch.tensor(t.task_id,   dtype=torch.int64),
            "task_stable": torch.tensor(t.stable_id, dtype=torch.int64),
            "S":           torch.tensor(S,           dtype=torch.int64),
            "S_raw":       torch.tensor(S_raw,       dtype=torch.int64),
            "q_idx":       torch.tensor(q_idx,       dtype=torch.int64),
            "mode":        torch.tensor(self.mode,   dtype=torch.int64),
            "pad_id":      torch.tensor(pad_id,      dtype=torch.int64),
            "eos_id":      torch.tensor(eos_id,      dtype=torch.int64),
        }


# ═══════════════════════════════════════════════════
# 6. Augmentation helpers (task-level, bbox-safe)
# ═══════════════════════════════════════════════════
def _extract_crop(g30: torch.Tensor, *, eos_id=10, pad_id=11):
    """g30 [30,30] long → crop [h,w] long (0..9 only), or None."""
    g = g30.long()
    m = (g != eos_id) & (g != pad_id)
    if not m.any():
        return None
    ys, xs = torch.where(m)
    crop = g[ys.min():ys.max()+1, xs.min():xs.max()+1].clone()
    crop[(crop == eos_id) | (crop == pad_id)] = 0   # safety
    return crop


def _encode_crop_to_30(crop: torch.Tensor, *, eos_id=10, pad_id=11):
    """crop [h,w] long (0..9) → [30,30] long with EOS L-shape + PAD."""
    h, w = crop.shape
    assert h <= 30 and w <= 30
    out = torch.full((30, 30), pad_id, dtype=torch.long, device=crop.device)
    if h == 0 or w == 0:
        return out
    out[:h, :w] = crop
    if w < 30:
        out[:h, w] = eos_id
    if h < 30:
        end_c = min(w + (1 if w < 30 else 0), 30)
        if end_c > 0:
            out[h, :end_c] = eos_id
    return out


def _apply_task_aug(g30, *, k, f, recolor, eos_id, pad_id):
    """
    1) extract crop  2) rot/flip  3) recolor  4) re-encode 30×30
    Returns [30,30] long. If crop empty → clone original.
    """
    crop = _extract_crop(g30, eos_id=eos_id, pad_id=pad_id)
    if crop is None:
        return g30.clone().long()

    # geom
    crop = torch.rot90(crop, k=k, dims=(-2, -1))
    if f == 1:
        crop = torch.flip(crop, dims=(-1,))
    elif f == 2:
        crop = torch.flip(crop, dims=(-2,))

    # recolor (cycle a→b→c→a, couleur 0 jamais touchée)
    if recolor is not None:
        a, b, c = recolor
        out = crop.clone()
        out[crop == a] = b
        out[crop == b] = c
        out[crop == c] = a
        crop = out

    return _encode_crop_to_30(crop, eos_id=eos_id, pad_id=pad_id)




# ═══════════════════════════════════════════════════
# 8. Build datasets + loaders
# ═══════════════════════════════════════════════════
def build_all(root: str = ".", batch_size: int = 16, p_task_aug_train: float = 0.5):
    cfg = ArcVarSConfig(eos_id=10, pad_id=11, rng_seed=0)
    paths =  {"root": "./data/", "training": "./data/training/", "evaluation": "./data/evaluation/"} 

    train_raw = load_split_tasks(paths["training"])
    eval_raw  = load_split_tasks(paths["evaluation"])
    train_tasks, eval_tasks, n_tasks = assign_split_task_ids_with_eval_offset(train_raw, eval_raw)

    DS = ARCVariableSupportDataset
    ds_train = DS(train_tasks, cfg=cfg, mode=DS.MODE_TRAIN_OR_EVAL_QUERY)
    ds_eval  = DS(eval_tasks,  cfg=cfg, mode=DS.MODE_TRAIN_OR_EVAL_QUERY)
    ds_trainext_eval = DS(eval_tasks, cfg=cfg, mode=DS.MODE_EVAL_AS_SUPPORT_QUERY)
    ds_trainext = ConcatDataset([ds_train, ds_trainext_eval])

    print(f"[ARC] train={len(ds_train)}  eval={len(ds_eval)}  "
          f"trainext={len(ds_trainext)}  n_tasks={n_tasks}")

    loader_kw = dict(pin_memory=True, persistent_workers=True)

    train_loader = DataLoader(
        ds_train, batch_size=batch_size, shuffle=True, num_workers=4,
        collate_fn=lambda b: collate_arc_fixedS(b, p_task_aug=p_task_aug_train),
        **loader_kw,
    )
    eval_loader = DataLoader(
        ds_eval, batch_size=batch_size, shuffle=False, num_workers=2,
        collate_fn=lambda b: collate_arc_fixedS(b, p_task_aug=0.0),
        **loader_kw,
    )
    trainext_loader = DataLoader(
        ds_trainext, batch_size=batch_size, shuffle=True, num_workers=4,
        collate_fn=lambda b: collate_arc_fixedS(b, p_task_aug=p_task_aug_train),
        **loader_kw,
    )

    return ds_train, ds_trainext, ds_eval, n_tasks, train_loader, eval_loader, trainext_loader



# ═══════════════════════════════════════════════════════════
# 4. augment_arc_colors — SEULE source de recolor
# ═══════════════════════════════════════════════════════════

"""
ARC Augmentation — Cohérent & Complet
======================================
Règles :
  1. MÊME transformation appliquée à TOUS les supports + query
  2. Ne JAMAIS toucher EOS (10) ni PAD (11)
  3. Color 0 (background) reste 0
  4. Géométrie (rotate, flip) respecte les bornes EOS/PAD
  5. UNE SEULE augmentation : dans le train_step, PAS dans encode_supports
"""

import torch
import torch.nn.functional as F
import random


# ════════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════════

def _find_content_hw(grid, eos_id=10, pad_id=11):
    """
    Trouve la taille réelle (h, w) du contenu dans une grille [H, W].
    Le contenu est tout ce qui est avant le premier EOS dans la colonne 0
    et avant le premier EOS dans la ligne 0.
    Retourne (h, w) — dimensions du contenu sans EOS/PAD.
    """
    H, W = grid.shape

    # Hauteur : premier EOS dans la colonne 0
    col0 = grid[:, 0]
    eos_rows = (col0 == eos_id).nonzero(as_tuple=True)[0]
    h = int(eos_rows[0].item()) if len(eos_rows) > 0 else H

    # Largeur : premier EOS dans la ligne 0
    if h == 0:
        return 0, 0
    row0 = grid[0, :]
    eos_cols = (row0 == eos_id).nonzero(as_tuple=True)[0]
    w = int(eos_cols[0].item()) if len(eos_cols) > 0 else W

    return h, w


def _rebuild_grid(content, H_out, W_out, eos_id=10, pad_id=11):
    """
    Place le contenu [h, w] dans une grille [H_out, W_out]
    avec EOS border et PAD fill.
    """
    h, w = content.shape
    device = content.device
    grid = torch.full((H_out, W_out), pad_id, dtype=content.dtype, device=device)

    # Contenu
    h_fit = min(h, H_out - 1)  # laisser place pour EOS
    w_fit = min(w, W_out - 1)
    grid[:h_fit, :w_fit] = content[:h_fit, :w_fit]

    # EOS border : colonne w_fit pour les lignes 0..h_fit-1
    if w_fit < W_out:
        grid[:h_fit, w_fit] = eos_id

    # EOS border : ligne h_fit pour les colonnes 0..w_fit
    if h_fit < H_out:
        grid[h_fit, :w_fit] = eos_id
        if w_fit < W_out:
            grid[h_fit, w_fit] = eos_id

    return grid


# ════════════════════════════════════════════════════════════════════
#  COLOR PERMUTATION
# ════════════════════════════════════════════════════════════════════

def _make_color_perm(device, n_colors=10):
    """
    Crée une table de remapping [0..11] → [0..11].
    - Color 0 reste 0
    - Colors 1-9 sont permutées aléatoirement
    - EOS (10) reste 10
    - PAD (11) reste 11
    """
    remap = torch.arange(12, device=device, dtype=torch.long)
    perm_1_9 = torch.randperm(9, device=device) + 1  # permutation de 1..9
    for i in range(9):
        remap[i + 1] = perm_1_9[i]
    return remap


def _apply_color_perm(grid, remap):
    """Applique le remapping de couleurs à une grille. PAD et EOS sont préservés."""
    return remap[grid.long()]


# ════════════════════════════════════════════════════════════════════
#  GEOMETRIC TRANSFORMS
# ════════════════════════════════════════════════════════════════════

def _extract_content(grid, eos_id=10, pad_id=11):
    """Extrait le contenu [h, w] d'une grille [H, W]."""
    h, w = _find_content_hw(grid, eos_id=eos_id, pad_id=pad_id)
    if h == 0 or w == 0:
        return grid[:1, :1]  # fallback : au moins 1x1
    return grid[:h, :w].clone()


def _rotate_content(content, k):
    """Rotate content by k*90 degrees counterclockwise."""
    if k == 0:
        return content
    return torch.rot90(content, k=k, dims=(0, 1))


def _flip_content(content, flip_h, flip_v):
    """Flip content horizontally and/or vertically."""
    if flip_h:
        content = content.flip(1)
    if flip_v:
        content = content.flip(0)
    return content


# ════════════════════════════════════════════════════════════════════
#  MAIN AUGMENTATION — TASK LEVEL
# ════════════════════════════════════════════════════════════════════

def augment_arc_task(
    sx, sy, qx, qy, *,
    eos_id=10, pad_id=11, Hmax=30, Wmax=30,
    do_color=True, do_geometry=True, aug_prob=0.8,
):
    """
    Augmente UNE tâche ARC de manière cohérente.

    Args:
        sx: [S, H, W] supports input
        sy: [S, H, W] supports output
        qx: [H, W] query input
        qy: [H, W] query output (GT)

    Returns:
        sx_aug, sy_aug, qx_aug, qy_aug — même shapes

    GARANTIES :
        - Même transformation appliquée à TOUS les grids
        - EOS/PAD jamais corrompus
        - Color 0 jamais touchée
        - Géométrie respecte les bornes de grille
    """
    if random.random() > aug_prob:
        return sx, sy, qx, qy

    device = sx.device
    S, H, W = sx.shape

    # ── Décider des transformations ─────────────────────────
    do_color_now = do_color and random.random() < 0.7
    do_rotate    = do_geometry and random.random() < 0.3
    do_flip_h    = do_geometry and random.random() < 0.5
    do_flip_v    = do_geometry and random.random() < 0.5

    # Color perm (une seule pour toute la tâche)
    color_remap = _make_color_perm(device) if do_color_now else None

    # Rotation (0, 90, 180, 270)
    rot_k = random.randint(1, 3) if do_rotate else 0

    # ── Vérifier que la rotation est faisable ─────────────────
    # Si rotation 90/270, les dimensions h/w sont swappées.
    # Vérifier que toutes les grilles transformées tiennent dans Hmax×Wmax.
    if rot_k in (1, 3):
        all_grids = [qx, qy] + [sx[s] for s in range(S)] + [sy[s] for s in range(S)]
        for g in all_grids:
            h, w = _find_content_hw(g, eos_id=eos_id, pad_id=pad_id)
            # Après rotation 90/270 : new_h=w, new_w=h
            if w >= Hmax or h >= Wmax:  # besoin de +1 pour EOS
                rot_k = 0  # annuler la rotation
                break

    # ── Fonction de transformation d'une grille ───────────────
    def transform_grid(grid):
        content = _extract_content(grid, eos_id=eos_id, pad_id=pad_id)

        # Color
        if color_remap is not None:
            content = _apply_color_perm(content, color_remap)

        # Geometry
        if rot_k > 0:
            content = _rotate_content(content, rot_k)
        content = _flip_content(content, do_flip_h, do_flip_v)

        # Rebuild avec EOS/PAD
        return _rebuild_grid(content, H, W, eos_id=eos_id, pad_id=pad_id)

    # ── Appliquer à tout ──────────────────────────────────────
    sx_aug = torch.stack([transform_grid(sx[s]) for s in range(S)])
    sy_aug = torch.stack([transform_grid(sy[s]) for s in range(S)])
    qx_aug = transform_grid(qx)
    qy_aug = transform_grid(qy)

    return sx_aug, sy_aug, qx_aug, qy_aug

def augment_arc_colors(sx, sy, qx, qy, *, n_colors=10, aug_prob=0.8,
                       eos_id=10, pad_id=11, Hmax=30, Wmax=30):
    """
    Augmente un batch entier. Chaque tâche (item du batch) reçoit
    sa propre transformation aléatoire, appliquée de manière cohérente
    à tous ses supports + query.

    Args:
        sx: [B, S, H, W]
        sy: [B, S, H, W]
        qx: [B, H, W]
        qy: [B, H, W]

    Returns:
        sx_aug, sy_aug, qx_aug, qy_aug — mêmes shapes
    """
    B = sx.shape[0]

    sx_list, sy_list, qx_list, qy_list = [], [], [], []

    for b in range(B):
        sx_b, sy_b, qx_b, qy_b = augment_arc_task(
            sx[b], sy[b], qx[b], qy[b],
            eos_id=eos_id, pad_id=pad_id,
            Hmax=Hmax, Wmax=Wmax,
            do_color=True, do_geometry=True,
            aug_prob=aug_prob,
        )
        sx_list.append(sx_b)
        sy_list.append(sy_b)
        qx_list.append(qx_b)
        qy_list.append(qy_b)

    return (torch.stack(sx_list), torch.stack(sy_list),
            torch.stack(qx_list), torch.stack(qy_list))


# ═══════════════════════════════════════════════════════════
# 2. Augmentation géométrique UNIQUEMENT (dans le collate)
# ═══════════════════════════════════════════════════════════

def _apply_geom_aug(g30, *, k, f, eos_id, pad_id):
    """
    Applique rot90(k) + flip(f) à la grille g30 [30,30] long.
    Extrait le crop → transforme → ré-encode proprement.
    PAS de recolor ici — uniquement géométrie.
    """
    crop = _extract_crop(g30, eos_id=eos_id, pad_id=pad_id)
    if crop is None:
        return g30.clone().long()

    # Rotation (k=0 → identité, k=1..3 → 90°/180°/270°)
    crop = torch.rot90(crop, k=k, dims=(-2, -1))

    # Flip (f=0 → rien, f=1 → horizontal, f=2 → vertical)
    if f == 1:
        crop = torch.flip(crop, dims=(-1,))
    elif f == 2:
        crop = torch.flip(crop, dims=(-2,))

    return _encode_crop_to_30(crop, eos_id=eos_id, pad_id=pad_id)




# ═══════════════════════════════════════════════════════════
# 3. collate_arc_fixedS — géométrie task-level, ZERO recolor
# ═══════════════════════════════════════════════════════════

def collate_arc_fixedS(
    batch,
    *,
    S_fixed: int = 4,
    pad_id: int = 11,
    eos_id: int = 10,
    p_task_aug: float = 0.0,
    # ⚠️ p_recolor SUPPRIMÉ — recolor géré exclusivement par augment_arc_colors
):
    """
    Retourne un batch avec augmentation géométrique cohérente (même k,f pour
    tous les supports ET la query). Le recolor est délégué à augment_arc_colors.

    Appelé dans le training loop :
        batch = collate_arc_fixedS(raw_batch, p_task_aug=0.5)
        sx, sy, qx, qy = augment_arc_colors(batch["sx"], batch["sy"],
                                             batch["qx"], batch["qy"],
                                             aug_prob=1.0)
    """
    B = len(batch)
    sx_out, sy_out, qx_out, qy_out = [], [], [], []
    task_ids, stables = [], []
    s_masks, s_reals, s_idxs = [], [], []

    for b in batch:
        sx = b["sx"].long()   # [S, 30, 30]
        sy = b["sy"].long()
        S  = sx.shape[0]

        # --- sélection / rembourrage des supports ---
        if S >= S_fixed:
            idxs = random.sample(range(S), S_fixed)
            # Copie explicite pour éviter aliasing lors de l'assignment in-place
            xs = sx[idxs].clone()
            ys = sy[idxs].clone()
            is_real = [True] * S_fixed
            idx_dbg = list(idxs)
        else:
            xs_l = [sx[i].clone() for i in range(S)]
            ys_l = [sy[i].clone() for i in range(S)]
            is_real = [True] * S
            idx_dbg = list(range(S))
            while len(xs_l) < S_fixed:
                j = random.randrange(S)
                xs_l.append(sx[j].clone())
                ys_l.append(sy[j].clone())
                is_real.append(False)
                idx_dbg.append(-1)
            xs = torch.stack(xs_l)
            ys = torch.stack(ys_l)

        qx = b["qx"].long().clone()
        qy = b["qy"].long().clone()

        # --- Augmentation géométrique task-level (MÊME transform pour tous) ---
        if p_task_aug > 0 and random.random() < p_task_aug:
            k = random.randint(0, 3)
            f = random.randint(0, 2)
            # Identité pure → skip le traitement coûteux
            if k != 0 or f != 0:
                kw = dict(k=k, f=f, eos_id=eos_id, pad_id=pad_id)
                for s in range(S_fixed):
                    xs[s] = _apply_geom_aug(xs[s], **kw)
                    ys[s] = _apply_geom_aug(ys[s], **kw)
                qx = _apply_geom_aug(qx, **kw)
                qy = _apply_geom_aug(qy, **kw)

        sx_out.append(xs)
        sy_out.append(ys)
        qx_out.append(qx)
        qy_out.append(qy)
        task_ids.append(b["task_id"])
        stables.append(b["task_stable"])
        s_masks.append(torch.ones(S_fixed, dtype=torch.bool))
        s_reals.append(torch.tensor(is_real, dtype=torch.bool))
        s_idxs.append(torch.tensor(idx_dbg, dtype=torch.int64))

    return {
        "sx":           torch.stack(sx_out),
        "sy":           torch.stack(sy_out),
        "qx":           torch.stack(qx_out),
        "qy":           torch.stack(qy_out),
        "s_mask":       torch.stack(s_masks),
        "s_is_real":    torch.stack(s_reals),
        "support_idxs": torch.stack(s_idxs),
        "task_id":      torch.stack(task_ids).view(B),
        "task_stable":  torch.stack(stables).view(B),
        "pad_id":       pad_id,
        "eos_id":       eos_id,
    }


# ═══════════════════════════════════════════════════
# Quick test
# ═══════════════════════════════════════════════════
if __name__ == "__main__":
    ds_train, ds_trainext, ds_eval, n_tasks, tl, el, txl = build_all(
        root=".", batch_size=8, p_task_aug_train=0.5
    )
    batch = next(iter(tl))
    print("sx:", batch["sx"].shape, batch["sx"].dtype)
    print("qy:", batch["qy"].shape, "task_ids:", batch["task_id"])
    print("s_is_real:", batch["s_is_real"])

