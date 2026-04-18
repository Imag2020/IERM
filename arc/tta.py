"""
arc/tta.py
──────────
Test-Time Augmentation for ARC-AGI — SELF-CONTAINED.

Implements D4 geometric symmetries + (optional) color permutations
on the *content* of each grid (before the EOS/PAD border), then
inverse-transforms the prediction back to the original orientation.

Why "on the content"?
─────────────────────
ARC grids are stored as 30×30 tensors with an L-shape EOS border and
PAD fill. Directly rotating the 30×30 tensor would mix EOS/PAD with
real colors. We instead:
  1. Extract the actual content (shape h×w, values 0..9)
  2. Apply the geometric transform to that content
  3. Rebuild a 30×30 grid with a fresh L-shape EOS border

Public API
──────────
    tta_predict(model, sx, sy, qx, ...)
        → (final_pred, mean_conf, N_aug, agreement)

Configuration that produced 12.25% on ARC-AGI-1 eval
────────────────────────────────────────────────────
    use_geometric=True, n_color_perms=1, voting="confidence", T_max=6
    → 8 views (D4 only), voting weighted by model confidence.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════
# 1. Grid structure: content extraction / rebuild
# ══════════════════════════════════════════════════════════════

def _extract_content(grid: torch.Tensor, eos_id: int = 10,
                      pad_id: int = 11):
    """
    Extract the actual content of a padded grid [H_pad, W_pad].

    Content convention:
      * Anchored at (0, 0)
      * Bounded by the first row/col that starts with EOS or is all PAD

    Returns:
        (content [h, w] LongTensor, h, w).  If content is empty, h=w=0.
    """
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

    return grid[:h, :w].clone(), h, w


def _rebuild_padded(content: torch.Tensor, H_pad: int, W_pad: int,
                    eos_id: int = 10, pad_id: int = 11) -> torch.Tensor:
    """
    Rebuild a [H_pad, W_pad] grid from [h, w] content with L-shape EOS.

    If the content exceeds the canvas in either dimension, it is clipped.
    This matters for D4 transforms that swap (h, w) → (w, h): if the
    task had an asymmetric grid close to 30 on one axis, rotating may
    require dropping a border. We clip rather than error so that TTA
    remains robust; such rare cases contribute a degraded view which
    voting will override.
    """
    h, w = content.shape
    device = content.device
    grid = torch.full((H_pad, W_pad), pad_id, dtype=content.dtype,
                      device=device)

    h_fit = min(h, H_pad)
    w_fit = min(w, W_pad)
    grid[:h_fit, :w_fit] = content[:h_fit, :w_fit]

    if w_fit < W_pad:
        grid[:h_fit, w_fit] = eos_id
    if h_fit < H_pad:
        end_c = min(w_fit + (1 if w_fit < W_pad else 0), W_pad)
        if end_c > 0:
            grid[h_fit, :end_c] = eos_id

    return grid


# ══════════════════════════════════════════════════════════════
# 2. D4 group — 8 geometric symmetries on content
# ══════════════════════════════════════════════════════════════

def _d4_content_transforms():
    """
    Return the 8 elements of D4 acting on content (sans EOS/PAD).

    Each item is (name, forward, inverse) where forward/inverse are
    callables taking a 2D tensor and returning a 2D tensor.

    Inverses verified by _verify_d4() (see below).
    """
    return [
        ("id",
         lambda g: g,
         lambda g: g),
        ("rot90",
         lambda g: torch.rot90(g, 1, [-2, -1]),
         lambda g: torch.rot90(g, -1, [-2, -1])),
        ("rot180",
         lambda g: torch.rot90(g, 2, [-2, -1]),
         lambda g: torch.rot90(g, 2, [-2, -1])),
        ("rot270",
         lambda g: torch.rot90(g, 3, [-2, -1]),
         lambda g: torch.rot90(g, 1, [-2, -1])),
        ("flip_h",
         lambda g: torch.flip(g, [-1]),
         lambda g: torch.flip(g, [-1])),
        ("flip_v",
         lambda g: torch.flip(g, [-2]),
         lambda g: torch.flip(g, [-2])),
        ("transp",
         lambda g: g.transpose(-2, -1),
         lambda g: g.transpose(-2, -1)),
        ("anti_t",
         lambda g: torch.rot90(g, 2, [-2, -1]).transpose(-2, -1),
         lambda g: torch.rot90(g, 2, [-2, -1]).transpose(-2, -1)),
    ]


def _verify_d4() -> bool:
    """Sanity-check that every D4 inverse is correct on a non-square grid."""
    g = torch.arange(5 * 7).reshape(5, 7).float()
    for name, fwd, inv in _d4_content_transforms():
        restored = inv(fwd(g))
        assert torch.allclose(restored, g), f"D4 inverse broken: {name}"
    return True


# ══════════════════════════════════════════════════════════════
# 3. Color permutations (task-aware)
# ══════════════════════════════════════════════════════════════

def _get_present_colors(sx: torch.Tensor, sy: torch.Tensor, qx: torch.Tensor,
                         eos_id: int = 10, pad_id: int = 11):
    """
    Colors 1..9 that actually appear anywhere in the task.

    Color 0 (background), EOS (10), PAD (11) are excluded.
    The result drives color-perm construction: we only permute colors
    that are actually used, so the permutation space stays tight.
    """
    all_grids = torch.cat([sx.reshape(-1), sy.reshape(-1), qx.reshape(-1)])
    present = set()
    for v in all_grids.unique().tolist():
        v = int(v)
        if 1 <= v <= 9:
            present.add(v)
    return sorted(present)


def _make_task_color_perms(n: int, present_colors, device):
    """
    Generate `n` color-permutation tables of size 12.

    Rules
    ─────
      * Index 0 (background) : FIXED
      * Indices 1..9 present in task : permuted among themselves
      * Indices 1..9 absent from task : FIXED (identity)
      * Index 10 (EOS), 11 (PAD) : FIXED

    The first entry is always the identity permutation, so the "default"
    view is included.

    Returns:
        List of (fwd_perm [12], inv_perm [12]) tensors.
    """
    identity = torch.arange(12, device=device)
    perms = [(identity, identity.clone())]

    present = list(present_colors)
    k = len(present)

    for _ in range(n - 1):
        fwd = torch.arange(12, device=device)
        if k >= 2:
            shuffled = [present[i] for i in torch.randperm(k).tolist()]
            for old, new in zip(present, shuffled):
                fwd[old] = new
        inv = torch.empty_like(fwd)
        inv[fwd] = torch.arange(12, device=device)
        perms.append((fwd, inv))

    return perms


# ══════════════════════════════════════════════════════════════
# 4. Transform helpers — apply geo+color to padded grids
# ══════════════════════════════════════════════════════════════

def _transform_grid(grid: torch.Tensor, geo_fwd, color_fwd,
                    H_pad: int, W_pad: int,
                    eos_id: int = 10, pad_id: int = 11) -> torch.Tensor:
    """Transform a single padded grid: extract → color → geo → rebuild."""
    content, _, _ = _extract_content(grid, eos_id, pad_id)
    content = color_fwd[content.long()]
    content = geo_fwd(content)
    return _rebuild_padded(content, H_pad, W_pad, eos_id, pad_id)


def _transform_batch_supports(sx: torch.Tensor, sy: torch.Tensor,
                               geo_fwd, color_fwd,
                               eos_id: int = 10, pad_id: int = 11):
    """Apply (geo_fwd, color_fwd) to every grid in [B, S, H, W]."""
    B, S, H, W = sx.shape
    sx_out = torch.empty_like(sx)
    sy_out = torch.empty_like(sy)
    for b in range(B):
        for s in range(S):
            sx_out[b, s] = _transform_grid(
                sx[b, s], geo_fwd, color_fwd, H, W, eos_id, pad_id)
            sy_out[b, s] = _transform_grid(
                sy[b, s], geo_fwd, color_fwd, H, W, eos_id, pad_id)
    return sx_out, sy_out


def _transform_batch_query(qx: torch.Tensor, geo_fwd, color_fwd,
                            eos_id: int = 10, pad_id: int = 11) -> torch.Tensor:
    """Apply (geo_fwd, color_fwd) to [B, H, W]."""
    B, H, W = qx.shape
    qx_out = torch.empty_like(qx)
    for b in range(B):
        qx_out[b] = _transform_grid(
            qx[b], geo_fwd, color_fwd, H, W, eos_id, pad_id)
    return qx_out


def _inverse_transform_pred(pred: torch.Tensor, geo_inv, color_inv,
                             eos_id: int = 10,
                             pad_id: int = 11) -> torch.Tensor:
    """Inverse geo, then inverse color, then re-encode."""
    B, H, W = pred.shape
    out = torch.empty_like(pred)
    for b in range(B):
        content, _, _ = _extract_content(pred[b], eos_id, pad_id)
        content = geo_inv(content)
        content = color_inv[content.long()]
        out[b] = _rebuild_padded(content, H, W, eos_id, pad_id)
    return out


# ══════════════════════════════════════════════════════════════
# 5. Main TTA entry point
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def tta_predict(
    model,
    sx: torch.Tensor,
    sy: torch.Tensor,
    qx: torch.Tensor,
    *,
    s_mask=None,
    task_id=None,
    T_max=None,
    confidence_threshold: float = 0.95,
    n_color_perms: int = 1,
    use_geometric: bool = True,
    voting: str = "confidence",
    pad_id: int = 11,
    eos_id: int = 10,
):
    """
    Run the model under multiple augmented views and vote on the output.

    Views
    ─────
    use_geometric × n_color_perms. With the paper config
    (use_geometric=True, n_color_perms=1) we get 8 views (D4 only).

    Voting
    ──────
    * "confidence" : each view contributes `max(conf, 0.01)` × one_hot(pred).
      This is the setting that produced 12.25% on ARC-AGI-1 eval.
    * "majority"   : each view contributes one vote.

    Returns
    ───────
        (final_pred [B, H, W], mean_conf, n_views, agreement)
    """
    model.eval()
    device = qx.device
    B_orig, H_pad, W_pad = qx.shape
    V = int(getattr(model, "vocab_size", 12))

    present_colors = _get_present_colors(sx, sy, qx, eos_id, pad_id)

    geo_transforms = (
        _d4_content_transforms() if use_geometric
        else _d4_content_transforms()[:1])
    color_perms = _make_task_color_perms(
        n_color_perms, present_colors, device)

    all_preds = []
    all_confs = []

    for geo_name, geo_fwd, geo_inv in geo_transforms:
        for color_fwd, color_inv in color_perms:
            sx_aug, sy_aug = _transform_batch_supports(
                sx, sy, geo_fwd, color_fwd, eos_id, pad_id)
            qx_aug = _transform_batch_query(
                qx, geo_fwd, color_fwd, eos_id, pad_id)

            sm_aug = s_mask.clone() if s_mask is not None else None

            pred_aug, conf, _ = model.predict(
                sx_aug, sy_aug, qx_aug,
                s_mask=sm_aug,
                task_id=task_id,
                T_max=T_max,
                confidence_threshold=confidence_threshold,
            )

            pred_restored = _inverse_transform_pred(
                pred_aug, geo_inv, color_inv, eos_id, pad_id)

            all_preds.append(pred_restored)
            all_confs.append(float(conf))

    N_aug = len(all_preds)
    preds_stack = torch.stack(all_preds, dim=0)  # [N_aug, B, H, W]
    vote_map = torch.zeros(B_orig, H_pad, W_pad, V, device=device)

    if voting == "confidence":
        for i in range(N_aug):
            w = max(all_confs[i], 0.01)
            one_hot = F.one_hot(
                preds_stack[i].long().clamp(0, V - 1), V).float()
            vote_map += w * one_hot
    else:  # majority
        for i in range(N_aug):
            one_hot = F.one_hot(
                preds_stack[i].long().clamp(0, V - 1), V).float()
            vote_map += one_hot

    final_pred = vote_map.argmax(-1)
    agreement = (preds_stack == final_pred.unsqueeze(0)).float().mean().item()
    mean_conf = sum(all_confs) / max(len(all_confs), 1)

    return final_pred, mean_conf, N_aug, agreement


__all__ = [
    "tta_predict",
    "_extract_content",
    "_rebuild_padded",
    "_d4_content_transforms",
    "_get_present_colors",
    "_make_task_color_perms",
]