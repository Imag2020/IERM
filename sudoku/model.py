"""
sudoku/model.py
───────────────
IERM specialized for Sudoku — SELF-CONTAINED.

This module contains EVERYTHING needed to load and run the Sudoku
checkpoints without any dependency on sibling packages (`arc/`, etc.).
Primitives (RMSNorm, CrossAttention, PsiMemProg, PhiTRM) are defined
locally so that changes to the ARC model will never break Sudoku
reproducibility.

Public API
──────────
    build_sudoku_model(cfg=CFG_SUDOKU) -> IERM_TRM_Sudoku
    load_sudoku_checkpoint(model, path, device, strict=True) -> dict
    IERMSudoku(embedder, ...)      # nn.Module
    SudokuEmbedder                      # structural 3×3 embeddings
    TokenMixerMLP                       # replaces attention on y-pathway
    CFG_SUDOKU                          # paper config (d=192, d_psi=120)

Reproduction
────────────
    >>> from sudoku.model import build_sudoku_model, load_sudoku_checkpoint
    >>> model = build_sudoku_model().to(device)
    >>> ckpt = load_sudoku_checkpoint(
    ...     model, "model_checkpoints/sudoku_mlp_62pct.pt",
    ...     device=device, strict=True)
    >>> pred, conf, T = model.predict(sx, sy, qx, use_tta=True, n_aug_passes=8)

Architecture choices (Sudoku-specific)
──────────────────────────────────────
  * phi.y_sa is a TokenMixerMLP instead of self-attention
    (L=81 is small enough that a direct cell-to-cell MLP is more
    effective than learned attention patterns).
  * Copy-rewrite output head: the model predicts (i) a per-cell mask
    for "rewrite vs copy" and (ii) a value distribution; the final
    prediction mixes the copy and the rewrite.
  * SudokuEmbedder adds block/band/stack embeddings that encode the
    3×3 structure directly into the input representation.
  * Digit-permutation TTA: at inference, we can average logits over
    N passes, each with a random permutation of digits 1..9 (with
    correct de-permutation of the output logits).

Sizes from the shipped checkpoint
─────────────────────────────────
  d_model=192, d_psi=120, n_heads=6, vocab=10,
  z_tokens=8, n_mem=64, n_prog=32, psi_refine_layers=2, T_psi=2.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════
# 1. Helpers
# ══════════════════════════════════════════════════════════════

def _force_keep_at_least_one(keep: torch.Tensor) -> torch.Tensor:
    """
    Ensure that at least one token is unmasked per batch item.

    When every key is masked, SDPA returns NaNs. We guarantee at least
    one live key by force-keeping the first position of any all-masked row.
    """
    all_masked = ~keep.any(dim=-1)
    if all_masked.any():
        keep = keep.clone()
        keep[all_masked, 0] = True
    return keep


def _sanitize(t: torch.Tensor,
              lo: float = -50.0, hi: float = 50.0) -> torch.Tensor:
    """Clamp and remove NaN/Inf, for use between recurrent steps."""
    return torch.nan_to_num(t, nan=0.0, posinf=hi, neginf=lo).clamp(lo, hi)


# ══════════════════════════════════════════════════════════════
# 2. Normalizations and FFN
# ══════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    """Root-mean-square layer normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.to(dtype=x.dtype)
        if hasattr(F, "rms_norm"):
            return F.rms_norm(x, self.weight.shape, weight=w, eps=self.eps)
        inv = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * inv).to(x.dtype) * w


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network: SiLU(gate) ⊙ up -> down."""

    def __init__(self, d_model: int, expand: float = 2.0, dropout: float = 0.1):
        super().__init__()
        hidden = ((int(d_model * expand) + 7) // 8) * 8
        self.w_gate = nn.Linear(d_model, hidden, bias=False)
        self.w_up = nn.Linear(d_model, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


# ══════════════════════════════════════════════════════════════
# 3. ARCEmbedder — token + position + type + pair + task embeddings
# ══════════════════════════════════════════════════════════════

class ARCEmbedder(nn.Module):
    """
    Grid embedder shared with the ARC pipeline.

    Combines five additive embeddings:
      - token (per-cell value)
      - row / column (2D positional)
      - type (0=support input, 1=support output, 2=query, 3=reserved)
      - pair (which support pair within a batch of S)
      - task (optional, per-task learnable embedding)

    A subclass (SudokuEmbedder) adds 3×3 block/band/stack embeddings.
    """

    def __init__(self, *, vocab_size=12, d=128, Hmax=30, Wmax=30,
                 eos_id=10, pad_id=11, n_types=4, Smax=8,
                 n_tasks=2048, use_task_id=True, dropout=0.1):
        super().__init__()
        self.d = int(d)
        self.vocab_size = int(vocab_size)
        self.Hmax = int(Hmax)
        self.Wmax = int(Wmax)
        self.pad_id = int(pad_id)
        self.eos_id = int(eos_id)

        self.emb_tok = nn.Embedding(vocab_size, d)
        self.emb_row = nn.Embedding(Hmax, d)
        self.emb_col = nn.Embedding(Wmax, d)
        self.emb_type = nn.Embedding(n_types, d)
        self.emb_pair = nn.Embedding(Smax, d)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(float(dropout))

        for emb in (self.emb_tok, self.emb_row, self.emb_col,
                    self.emb_type, self.emb_pair):
            nn.init.normal_(emb.weight, std=0.02)

        self.use_task_id = bool(use_task_id)
        self.n_tasks = int(n_tasks)
        if self.use_task_id:
            self.emb_task = nn.Embedding(self.n_tasks, d)
            nn.init.normal_(self.emb_task.weight, std=0.02)

    def forward(self, x, type_id, pair_id=None, task_id=None,
                return_mask=False):
        if x.dtype != torch.long:
            x = x.long()
        B, H, W = x.shape
        L = H * W
        device = x.device
        x_flat = x.reshape(B, L)

        e = self.emb_tok(x_flat.clamp(0, self.vocab_size - 1))
        rows = torch.arange(H, device=device)
        cols = torch.arange(W, device=device)
        pos = (self.emb_row(rows)[:, None, :]
               + self.emb_col(cols)[None, :, :]).reshape(1, L, self.d)
        e = e + pos

        type_id_t = self._to_long_vector(type_id, B, device).clamp(
            0, self.emb_type.num_embeddings - 1)
        e = e + self.emb_type(type_id_t)[:, None, :]

        if pair_id is not None:
            pair_id_t = self._to_long_vector(pair_id, B, device).clamp(
                0, self.emb_pair.num_embeddings - 1)
            e = e + self.emb_pair(pair_id_t)[:, None, :]

        if self.use_task_id and task_id is not None:
            tid = self._to_long_vector(task_id, B, device).clamp(
                0, self.n_tasks - 1)
            e = e + self.emb_task(tid)[:, None, :]

        e = self.drop(self.norm(e))

        mask_keep = (x_flat != self.pad_id)
        all_pad = (~mask_keep).all(dim=1)
        if all_pad.any():
            mask_keep = mask_keep.clone()
            mask_keep[all_pad, 0] = True

        if return_mask:
            return e, mask_keep
        return e

    @staticmethod
    def _to_long_vector(val, B, device):
        """Coerce `val` (int | scalar tensor | 1-D tensor) to a [B] LongTensor."""
        if isinstance(val, int):
            return torch.full((B,), val, device=device, dtype=torch.long)
        t = val.to(device=device, dtype=torch.long)
        if t.numel() == 1:
            return t.view(1).expand(B)
        t_flat = t.reshape(-1)
        if t_flat.shape[0] < B:
            pad = t_flat[-1:].expand(B - t_flat.shape[0])
            t_flat = torch.cat([t_flat, pad], dim=0)
        return t_flat[:B]


# ══════════════════════════════════════════════════════════════
# 4. 2D Rotary Position Embedding
# ══════════════════════════════════════════════════════════════

class RoPE2D(nn.Module):
    """
    2D Rotary Position Embedding for grids.

    Splits the head dimension in two halves, applying 1D RoPE to one
    half using row indices and to the other half using column indices.
    """

    def __init__(self, d_head: int, max_h: int = 30, max_w: int = 30,
                 theta: float = 10000.0):
        super().__init__()
        assert d_head % 4 == 0, "d_head must be divisible by 4 for 2D RoPE"
        self.d_head = d_head
        self.d_quat = d_head // 4
        inv_freq = 1.0 / (theta ** (
            torch.arange(0, self.d_quat, dtype=torch.float) / self.d_quat))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_h, max_w)

    def _build_cache(self, H: int, W: int):
        device = self.inv_freq.device
        rows = torch.arange(H, dtype=torch.float, device=device)
        cols = torch.arange(W, dtype=torch.float, device=device)
        fr = torch.outer(rows, self.inv_freq)[:, None, :].expand(H, W, self.d_quat)
        fc = torch.outer(cols, self.inv_freq)[None, :, :].expand(H, W, self.d_quat)
        emb_r = fr.reshape(H * W, self.d_quat).repeat(1, 2)
        emb_c = fc.reshape(H * W, self.d_quat).repeat(1, 2)
        emb = torch.cat([emb_r, emb_c], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        self._cached_hw = (H, W)

    def _rotate_half(self, x):
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    def forward(self, q, k, H: int, W: int):
        ch, cw = getattr(self, "_cached_hw", (0, 0))
        if ch < H or cw < W:
            self._build_cache(max(H, ch), max(W, cw))
        _, Wc = self._cached_hw
        idx = (torch.arange(H, device=q.device)[:, None] * Wc
               + torch.arange(W, device=q.device)[None, :]).reshape(-1).long()
        cos = self.cos_cached[idx].to(q.dtype)[None, None]
        sin = self.sin_cached[idx].to(q.dtype)[None, None]
        return (q * cos + self._rotate_half(q) * sin,
                k * cos + self._rotate_half(k) * sin)


# ══════════════════════════════════════════════════════════════
# 5. Attention modules
# ══════════════════════════════════════════════════════════════

class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with optional QK-norm and 2D-RoPE."""

    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1, *,
                 use_rope_2d: bool = False, max_h: int = 30, max_w: int = 30,
                 use_qk_norm: bool = False):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.dropout = float(dropout)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.use_rope_2d = bool(use_rope_2d)
        if self.use_rope_2d:
            self.rope = RoPE2D(self.d_head, max_h=max_h, max_w=max_w)
        self.use_qk_norm = bool(use_qk_norm)
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.d_head)
            self.k_norm = RMSNorm(self.d_head)

    def forward(self, x, *, key_padding_keep=None, grid_hw=None):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_head, self.d_head)
        q, k, v = qkv.unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if self.use_qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        if self.use_rope_2d and grid_hw is not None:
            q, k = self.rope(q, k, int(grid_hw[0]), int(grid_hw[1]))
        attn_mask = None
        if key_padding_keep is not None:
            keep = _force_keep_at_least_one(key_padding_keep)
            attn_mask = torch.zeros(B, 1, 1, N, device=x.device,
                                     dtype=torch.float32)
            attn_mask.masked_fill_(~keep[:, None, None, :], float("-inf"))
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0)
        return self.out(y.transpose(1, 2).reshape(B, N, D))


class CrossAttention(nn.Module):
    """Multi-head cross-attention with optional QK-norm."""

    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1, *,
                 use_qk_norm: bool = False):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.dropout = float(dropout)
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        self.use_qk_norm = bool(use_qk_norm)
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.d_head)
            self.k_norm = RMSNorm(self.d_head)

    def forward(self, q, kv, *, kv_keep=None):
        B, M, D = q.shape
        N = kv.shape[1]
        Q = self.wq(q).reshape(B, M, self.n_head, self.d_head).transpose(1, 2)
        K = self.wk(kv).reshape(B, N, self.n_head, self.d_head).transpose(1, 2)
        V = self.wv(kv).reshape(B, N, self.n_head, self.d_head).transpose(1, 2)
        if self.use_qk_norm:
            Q, K = self.q_norm(Q), self.k_norm(K)
        attn_mask = None
        if kv_keep is not None:
            keep = _force_keep_at_least_one(kv_keep)
            attn_mask = torch.zeros(B, 1, 1, N, device=q.device,
                                     dtype=torch.float32)
            attn_mask.masked_fill_(~keep[:, None, None, :], float("-inf"))
        y = F.scaled_dot_product_attention(
            Q, K, V, attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0)
        return self.wo(y.transpose(1, 2).reshape(B, M, D))


# ══════════════════════════════════════════════════════════════
# 6. TRMSelfBlock
# ══════════════════════════════════════════════════════════════

class TRMSelfBlock(nn.Module):
    """Pre-norm self-attention + FFN block used in Ψ's refinement stage."""

    def __init__(self, d: int, n_heads: int = 4, dropout: float = 0.1,
                 use_rope_2d: bool = False, use_qk_norm: bool = False):
        super().__init__()
        self.pre_sa = RMSNorm(d)
        self.sa = MultiHeadSelfAttention(d, n_heads, dropout,
                                          use_rope_2d=use_rope_2d,
                                          use_qk_norm=use_qk_norm)
        self.pre_ff = RMSNorm(d)
        self.ff = SwiGLUFFN(d, expand=2.0, dropout=dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, *, key_padding_keep=None, grid_hw=None):
        x = x + self.drop(self.sa(self.pre_sa(x),
                                   key_padding_keep=key_padding_keep,
                                   grid_hw=grid_hw))
        x = x + self.drop(self.ff(self.pre_ff(x)))
        return x


# ══════════════════════════════════════════════════════════════
# 7. Ψ — Interactive Memory-Program encoder
# ══════════════════════════════════════════════════════════════

class PsiMemProg(nn.Module):
    """
    Interactive Memory-Program encoder.

    Encodes (sx_i, sy_i)_{i=0..S-1} into an abstract program ctx via
    T_psi iterations of co-refinement between a Memory M and a Program P:

        supp → M_0
        for t in T_psi:
            M ← M + xa(M, supp)          # M re-reads the supports
            M ← M + gate * xa(M, P)      # P feedback guides M's focus
            P ← P + xa(P, M)             # P refines from M
            P ← P + ff(P)
        ctx = Refine(P)

    The p2m_gate (init sigmoid(-1) ≈ 0.27) prevents the P→M feedback
    from destabilizing training early on.
    """

    def __init__(self, embedder, D_phi: int, D_psi: int, n_heads: int = 4,
                 n_mem: int = 64, n_prog: int = 16, dropout: float = 0.1,
                 pad_id: int = 11, eos_id: int = 10, refine_layers: int = 1,
                 T_psi: int = 2):
        super().__init__()
        self.emb = embedder
        self.D_phi = int(D_phi)
        self.D_psi = int(D_psi)
        self.pad_id = int(pad_id)
        self.eos_id = int(eos_id)
        self.n_mem = int(n_mem)
        self.n_prog = int(n_prog)
        self.T_psi = int(T_psi)

        self.proj = (nn.Linear(D_phi, D_psi, bias=False)
                     if D_phi != D_psi else nn.Identity())
        if D_phi != D_psi:
            nn.init.normal_(self.proj.weight, std=0.02)

        self.mem_tokens = nn.Parameter(torch.randn(n_mem, D_psi) * 0.02)
        self.prog_tokens = nn.Parameter(torch.randn(n_prog, D_psi) * 0.02)

        # Stage 1: memory initialization (supp → M_0)
        self.mem_init_pre_kv = RMSNorm(D_psi)
        self.mem_init_pre_q = RMSNorm(D_psi)
        self.mem_init_xa = CrossAttention(D_psi, n_heads, dropout,
                                           use_qk_norm=True)
        self.mem_init_pre_ff = RMSNorm(D_psi)
        self.mem_init_ff = SwiGLUFFN(D_psi, dropout=dropout)
        self.mem_init_drop = nn.Dropout(dropout)

        # Interactive loop: M re-reads supp
        self.loop_m_supp_pre_kv = RMSNorm(D_psi)
        self.loop_m_supp_pre_q = RMSNorm(D_psi)
        self.loop_m_supp_xa = CrossAttention(D_psi, n_heads, dropout,
                                              use_qk_norm=True)

        # Interactive loop: M reads P (feedback)
        self.loop_m_prog_pre_kv = RMSNorm(D_psi)
        self.loop_m_prog_pre_q = RMSNorm(D_psi)
        self.loop_m_prog_xa = CrossAttention(D_psi, n_heads, dropout,
                                              use_qk_norm=True)
        self.loop_m_ff_pre = RMSNorm(D_psi)
        self.loop_m_ff = SwiGLUFFN(D_psi, dropout=dropout)
        self.loop_m_drop = nn.Dropout(dropout)

        # Interactive loop: P reads M
        self.loop_p_mem_pre_kv = RMSNorm(D_psi)
        self.loop_p_mem_pre_q = RMSNorm(D_psi)
        self.loop_p_mem_xa = CrossAttention(D_psi, n_heads, dropout,
                                             use_qk_norm=True)
        self.loop_p_ff_pre = RMSNorm(D_psi)
        self.loop_p_ff = SwiGLUFFN(D_psi, expand=2.0, dropout=dropout)
        self.loop_p_drop = nn.Dropout(dropout)

        # Gate for P→M feedback.
        self.p2m_gate = nn.Parameter(torch.full((D_psi,), -1.0))

        # Final stage: Refine(P).
        self.refine = nn.ModuleList([
            TRMSelfBlock(D_psi, n_heads=n_heads, dropout=dropout,
                          use_qk_norm=True)
            for _ in range(refine_layers)
        ])

    def forward(self, sx, sy, s_mask=None, task_id=None):
        B, S, H, W = sx.shape
        device = sx.device
        L = H * W

        # Embed supports.
        sx_flat = sx.reshape(B * S, H, W)
        sy_flat = sy.reshape(B * S, H, W)
        pair_ids = torch.arange(S, device=device).unsqueeze(0).expand(
            B, S).reshape(B * S)

        t = None
        if task_id is not None:
            t = self.emb._to_long_vector(task_id, B, device)
            t = t.view(B, 1).expand(B, S).reshape(B * S)

        ex, mx = self.emb(sx_flat, type_id=0, pair_id=pair_ids,
                           task_id=t, return_mask=True)
        ey, my = self.emb(sy_flat, type_id=1, pair_id=pair_ids,
                           task_id=t, return_mask=True)

        ex = self.proj(ex)
        ey = self.proj(ey)

        supp = torch.cat([ex, ey], dim=1).reshape(B, S * 2 * L, self.D_psi)
        supp_keep = torch.cat([mx, my], dim=1).reshape(B, S * 2 * L)

        if s_mask is not None:
            sm = s_mask.bool()
            sm_exp = sm[:, :, None].expand(B, S, 2 * L).reshape(B, S * 2 * L)
            supp_keep = supp_keep & sm_exp

        supp_keep = _force_keep_at_least_one(supp_keep)

        # Memory init.
        M = self.mem_tokens.unsqueeze(0).expand(B, -1, -1).clone()
        M = M + self.mem_init_drop(
            self.mem_init_xa(self.mem_init_pre_q(M),
                              self.mem_init_pre_kv(supp),
                              kv_keep=supp_keep))
        M = M + self.mem_init_drop(self.mem_init_ff(self.mem_init_pre_ff(M)))

        # Program init.
        P = self.prog_tokens.unsqueeze(0).expand(B, -1, -1).clone()

        # Interactive loop.
        p2m_gate = torch.sigmoid(self.p2m_gate).view(1, 1, -1)

        for _ in range(self.T_psi):
            # M re-reads supp.
            M = M + self.loop_m_drop(
                self.loop_m_supp_xa(
                    self.loop_m_supp_pre_q(M),
                    self.loop_m_supp_pre_kv(supp),
                    kv_keep=supp_keep))

            # P → M feedback.
            M = M + p2m_gate * self.loop_m_drop(
                self.loop_m_prog_xa(
                    self.loop_m_prog_pre_q(M),
                    self.loop_m_prog_pre_kv(P)))
            M = M + self.loop_m_drop(self.loop_m_ff(self.loop_m_ff_pre(M)))

            # P refines from M.
            P = P + self.loop_p_drop(
                self.loop_p_mem_xa(
                    self.loop_p_mem_pre_q(P),
                    self.loop_p_mem_pre_kv(M)))
            P = P + self.loop_p_drop(self.loop_p_ff(self.loop_p_ff_pre(P)))

        # Refine P.
        p_keep = torch.ones(B, self.n_prog, device=device, dtype=torch.bool)
        for blk in self.refine:
            P = blk(P, key_padding_keep=p_keep)

        P = _sanitize(P)

        return {
            "ctx": P,
            "ctx_keep": p_keep,
            "M": M,
            "supp": supp,
            "supp_keep": supp_keep,
        }


# ══════════════════════════════════════════════════════════════
# 8. Φ — PhiTRM with copy-rewrite head
# ══════════════════════════════════════════════════════════════

class PhiTRM(nn.Module):
    """
    Latent-recursion decoder with a copy-rewrite output head.

    Forward pass
    ────────────
      Given the query embedding x, an evolving working state y, and
      the abstract program ctx from Ψ, repeatedly update (y, z) via
      inner latent recursion, then decode y to output logits.

    Output head (copy-rewrite)
    ──────────────────────────
      For each cell i, predicts:
        m_i  — probability that the cell is "rewritten" (vs copied)
        v_i  — distribution over vocab (the rewrite value)
      Final mix:
        p_i  = (1 - m_i) * one_hot(x_i) + m_i * softmax(v_i)

      For Sudoku, "given" cells (x_i != 0) should be copied (m_i → 0),
      and "empty" cells (x_i == 0) should be rewritten (m_i → 1) with
      the correct digit.
    """

    def __init__(self, *, d_model: int, vocab_size: int, n_heads: int,
                 n_inner: int, T_max: int, d_psi: Optional[int] = None,
                 dropout: float = 0.1, use_rope_2d: bool = True,
                 use_qk_norm: bool = True, pad_id: int = 11,
                 eos_id: int = 10, z_tokens: int = 8):
        super().__init__()
        self.D = int(d_model)
        self.vocab_size = int(vocab_size)
        self.pad_id = int(pad_id)
        self.eos_id = int(eos_id)
        self.T_max = int(T_max)
        self.z_tokens = int(z_tokens)
        self.n_inner = int(n_inner)

        D_psi = int(d_psi) if d_psi else self.D
        if D_psi != self.D:
            self.ctx_proj = nn.Linear(D_psi, self.D, bias=False)
            nn.init.normal_(self.ctx_proj.weight, std=0.02)
        else:
            self.ctx_proj = nn.Identity()
        self.ctx_norm = RMSNorm(self.D)

        self.z_init = nn.Parameter(torch.randn(z_tokens, self.D) * 0.02)
        self.drop = nn.Dropout(float(dropout))

        # z pathway.
        self.z_pre_sa = RMSNorm(self.D)
        self.z_sa = MultiHeadSelfAttention(
            self.D, n_heads, dropout, use_qk_norm=use_qk_norm)

        self.z_pre_y = RMSNorm(self.D)
        self.z_xa_y = CrossAttention(self.D, n_heads, dropout,
                                      use_qk_norm=use_qk_norm)

        self.z_pre_x = RMSNorm(self.D)
        self.z_xa_x = CrossAttention(self.D, n_heads, dropout,
                                      use_qk_norm=use_qk_norm)

        self.z_pre_ctx = RMSNorm(self.D)
        self.z_xa_ctx = CrossAttention(self.D, n_heads, dropout,
                                        use_qk_norm=use_qk_norm)

        self.z_pre_ff = RMSNorm(self.D)
        self.z_ff = SwiGLUFFN(self.D, expand=2.0, dropout=dropout)

        # z recalibration via ctx (late correction pathway).
        self.z_recalib_pre_q = RMSNorm(self.D)
        self.z_recalib_pre_kv = RMSNorm(self.D)
        self.z_recalib_xa = CrossAttention(self.D, n_heads, dropout,
                                            use_qk_norm=use_qk_norm)
        self.z_recalib_pre_ff = RMSNorm(self.D)
        self.z_recalib_ff = SwiGLUFFN(self.D, expand=2.0, dropout=dropout)
        self.z_recalib_drop = nn.Dropout(float(dropout))
        self.z_recalib_gate = nn.Parameter(torch.full((self.D,), -1.0))

        # y pathway. (y_sa is replaced by TokenMixerMLP in IERM_TRM_Sudoku.)
        self.y_pre_sa = RMSNorm(self.D)
        self.y_sa = MultiHeadSelfAttention(
            self.D, n_heads, dropout,
            use_rope_2d=use_rope_2d, use_qk_norm=use_qk_norm)

        self.y_pre_z = RMSNorm(self.D)
        self.y_xa_z = CrossAttention(self.D, n_heads, dropout,
                                      use_qk_norm=use_qk_norm)

        self.y_pre_ctx = RMSNorm(self.D)
        self.y_xa_ctx = CrossAttention(self.D, n_heads, dropout,
                                        use_qk_norm=use_qk_norm)

        self.y_pre_ff = RMSNorm(self.D)
        self.y_ff = SwiGLUFFN(self.D, expand=2.0, dropout=dropout)

        # Copy-rewrite output heads.
        head_in = 4 * self.D
        self.norm_out = RMSNorm(self.D)
        self.mask_head = nn.Sequential(
            RMSNorm(head_in),
            nn.Linear(head_in, self.D),
            nn.GELU(),
            nn.Linear(self.D, 1))
        nn.init.constant_(self.mask_head[-1].bias, 0.0)
        self.val_head = nn.Sequential(
            RMSNorm(head_in),
            nn.Linear(head_in, self.D),
            nn.GELU(),
            nn.Linear(self.D, vocab_size))
        self.q_head = nn.Sequential(
            RMSNorm(self.D),
            nn.Linear(self.D, 1))

    def init_z(self, B: int, device, dtype):
        z = self.z_init.unsqueeze(0).expand(B, -1, -1).clone().to(
            device=device, dtype=dtype)
        z_keep = torch.ones(B, self.z_tokens, device=device, dtype=torch.bool)
        return z, z_keep

    def set_n_inner(self, n: int):
        self.n_inner = int(n)

    # ────────────────────────────────────────────────────────
    # latent_recursion: one block of n_inner iterations
    # ────────────────────────────────────────────────────────
    def latent_recursion(self, x, y, z, ctx, *,
                          x_keep=None, y_keep=None, z_keep=None,
                          ctx_keep=None, grid_hw=None):
        for _ in range(self.n_inner):
            # z update.
            z = z + self.drop(self.z_sa(self.z_pre_sa(z),
                                         key_padding_keep=z_keep))
            if y is not None:
                z = z + self.drop(self.z_xa_y(self.z_pre_y(z), y,
                                                kv_keep=y_keep))
            z = z + self.drop(self.z_xa_x(self.z_pre_x(z), x,
                                            kv_keep=x_keep))
            if ctx is not None:
                z = z + self.drop(self.z_xa_ctx(self.z_pre_ctx(z), ctx,
                                                  kv_keep=ctx_keep))
            z = z + self.drop(self.z_ff(self.z_pre_ff(z)))
            z = _sanitize(z)

            # z recalibration.
            if ctx is not None:
                gate = torch.sigmoid(self.z_recalib_gate).view(1, 1, -1)
                z_upd = self.z_recalib_xa(
                    self.z_recalib_pre_q(z),
                    self.z_recalib_pre_kv(ctx),
                    kv_keep=ctx_keep)
                z = z + gate * self.z_recalib_drop(z_upd)
                z = z + gate * self.z_recalib_drop(
                    self.z_recalib_ff(self.z_recalib_pre_ff(z)))

            # y update.
            y = y + self.drop(self.y_sa(self.y_pre_sa(y),
                                         key_padding_keep=y_keep,
                                         grid_hw=grid_hw))
            y = y + self.drop(self.y_xa_z(self.y_pre_z(y), z,
                                            kv_keep=z_keep))
            if ctx is not None:
                y = y + self.drop(self.y_xa_ctx(self.y_pre_ctx(y), ctx,
                                                  kv_keep=ctx_keep))
            y = y + self.drop(self.y_ff(self.y_pre_ff(y)))
            y = _sanitize(y)

        return y, z

    # ────────────────────────────────────────────────────────
    # _output_heads: copy-rewrite decoder
    # ────────────────────────────────────────────────────────
    def _output_heads(self, y, y_keep=None, x_embed=None, x_ids=None):
        B, L, D = y.shape
        V = self.vocab_size

        yn = self.norm_out(y)
        if x_embed is not None:
            delta = yn - x_embed
            feat = torch.cat([yn, x_embed, delta, delta.abs()], dim=-1)
        else:
            z4 = torch.zeros_like(yn)
            feat = torch.cat([yn, z4, z4, z4], dim=-1)

        m_logits = self.mask_head(feat).squeeze(-1).clamp(-20., 20.)
        v_logits = self.val_head(feat).clamp(-20., 20.)

        m_prob = torch.sigmoid(m_logits * 3.0).unsqueeze(-1)
        v_prob = F.softmax(v_logits, dim=-1)

        if x_ids is not None:
            copy_prob = F.one_hot(
                x_ids.clamp(0, V - 1).long(), num_classes=V).float()
            pad_pos = (x_ids == self.pad_id)
            if pad_pos.any():
                copy_prob = copy_prob.clone()
                copy_prob[pad_pos] = 0.0
                m_prob = m_prob.clone()
                m_prob[pad_pos] = 1.0
        else:
            copy_prob = torch.zeros(B, L, V, device=y.device)

        p_mix = (1.0 - m_prob) * copy_prob + m_prob * v_prob
        logits = torch.log(p_mix + 1e-8)

        if y_keep is not None:
            w = y_keep.float().unsqueeze(-1)
            ymean = (y.float() * w).sum(1) / w.sum(1).clamp_min(1.0)
        else:
            ymean = y.float().mean(1)

        q_logit = self.q_head(ymean).squeeze(-1).clamp(-20., 20.)
        q_hat = torch.sigmoid(q_logit).to(y.dtype)

        return logits, q_hat, q_logit, m_logits, v_logits

    # ────────────────────────────────────────────────────────
    # deep_recursion: T steps with residual-α and partial grad
    # ────────────────────────────────────────────────────────
    def deep_recursion(self, x_embed, y, ctx, *,
                        ctx_keep=None, x_keep=None, y_keep=None,
                        grid_hw=None, T=None,
                        z_prev=None, z_keep_prev=None,
                        x_ids=None,
                        detach_state: bool = True,
                        n_grad_steps: int = 2,
                        residual_alpha: float = 0.4):
        """
        T steps of latent_recursion with explicit residual:
            y_{t+1} = (1 - α) * y_t + α * latent_recursion(y_t)
        Only the last n_grad_steps carry gradients.
        """
        T = int(T or self.T_max)
        device = y.device
        B = y.shape[0]

        if ctx_keep is None and ctx is not None:
            ctx_keep = torch.ones(ctx.shape[:2], device=device,
                                   dtype=torch.bool)
        if x_keep is None:
            x_keep = torch.ones(x_embed.shape[:2], device=device,
                                 dtype=torch.bool)
        if y_keep is None:
            y_keep = torch.ones(y.shape[:2], device=device, dtype=torch.bool)

        ctx_phi = self.ctx_norm(self.ctx_proj(ctx)) if ctx is not None else None

        if z_prev is None:
            z, z_keep = self.init_z(B, device, y.dtype)
        else:
            z = z_prev
            z_keep = (z_keep_prev if z_keep_prev is not None
                      else torch.ones(z_prev.shape[:2], device=device,
                                      dtype=torch.bool))

        alpha = float(residual_alpha)

        def _step(y_in, z_in):
            y_new, z_new = self.latent_recursion(
                x_embed, y_in, z_in, ctx_phi,
                x_keep=x_keep, y_keep=y_keep, z_keep=z_keep,
                ctx_keep=ctx_keep, grid_hw=grid_hw)
            y_out = (1.0 - alpha) * y_in + alpha * y_new
            return y_out, z_new

        n_nograd = max(0, T - n_grad_steps)
        if n_nograd > 0:
            with torch.no_grad():
                for _ in range(n_nograd):
                    y, z = _step(y, z)
            y = y.detach().requires_grad_(True)
            z = z.detach().requires_grad_(True)

        for _ in range(min(n_grad_steps, T)):
            y, z = _step(y, z)

        logits, q_hat, q_logit, m_logits, v_logits = self._output_heads(
            y, y_keep, x_embed, x_ids)

        state = (y.detach(), z.detach()) if detach_state else (y, z)
        return state, logits, q_hat, q_logit, m_logits, v_logits


# ══════════════════════════════════════════════════════════════
# 9. IERM_TRM — base (shared with ARC)
# ══════════════════════════════════════════════════════════════

class IERM(nn.Module):
    """
    Interactive Reasoning Model.

    Composition of an interactive encoder (Ψ = PsiMemProg) and a
    latent-recursion decoder (Φ = PhiTRM) TRM style.

    The Sudoku subclass (below) replaces phi.y_sa with a TokenMixerMLP
    and adds TTA-aware predict().
    """

    def __init__(self, embedder, *, vocab_size=12, d_model=128, d_psi=64,
                 n_heads=4, n_inner=4, T_max=3, use_rope_2d=True,
                 use_qk_norm=True, pad_id=11, eos_id=10, dropout=0.1,
                 loso_lambda=0.1, loso_max_holds=4,
                 support_verify_lambda=0.3,
                 n_mem=64, n_prog=16, psi_refine_layers=1, z_tokens=8,
                 T_psi=2,
                 aug_supports_prob=0.5,
                 aug_n_views=2):
        super().__init__()
        self.embedder = embedder
        self.vocab_size = int(vocab_size)
        self.D = int(d_model)
        self.d_psi = int(d_psi)
        self.pad_id = int(pad_id)
        self.eos_id = int(eos_id)
        self.T_max = int(T_max)
        self.loso_lambda = float(loso_lambda)
        self.loso_max_holds = int(loso_max_holds)
        self.support_verify_lambda = float(support_verify_lambda)
        self.aug_supports_prob = float(aug_supports_prob)
        self.aug_n_views = int(aug_n_views)

        self.psi = PsiMemProg(
            embedder, D_phi=d_model, D_psi=d_psi,
            n_heads=n_heads, n_mem=n_mem, n_prog=n_prog,
            dropout=dropout, pad_id=pad_id, eos_id=eos_id,
            refine_layers=psi_refine_layers,
            T_psi=T_psi)

        self.phi = PhiTRM(
            d_model=d_model, vocab_size=vocab_size,
            n_heads=n_heads, n_inner=n_inner, T_max=T_max,
            d_psi=d_psi, dropout=dropout,
            use_rope_2d=use_rope_2d, use_qk_norm=use_qk_norm,
            pad_id=pad_id, eos_id=eos_id, z_tokens=z_tokens)

    def _embed_query(self, qx, task_id=None):
        return self.embedder(qx, type_id=2, task_id=task_id, return_mask=True)

    def _run_decoder(self, qx, ctx, ctx_keep, *, T=None,
                      y_prev=None, z_prev=None, z_keep=None, task_id=None):
        B, H, W = qx.shape
        x_embed, x_keep = self._embed_query(qx, task_id=task_id)
        x_ids = qx.reshape(B, H * W)
        y = y_prev if y_prev is not None else x_embed

        (y_det, z_det), logits, q_hat, q_logit, m_logits, v_logits = \
            self.phi.deep_recursion(
                x_embed, y, ctx,
                ctx_keep=ctx_keep, x_keep=x_keep,
                y_keep=x_keep, grid_hw=(H, W), T=T,
                z_prev=z_prev, z_keep_prev=z_keep,
                x_ids=x_ids,
                residual_alpha=getattr(self, "_residual_alpha", 0.4))

        return (y_det, z_det, logits, q_hat, q_logit, m_logits, v_logits,
                x_embed, x_keep)

    def encode_supports(self, sx, sy, s_mask=None, task_id=None):
        """Run Ψ on the supports. Returns (psi_out, loso_loss)."""
        B, S, H, W = sx.shape
        device = sx.device

        if s_mask is None:
            content = (((sx != self.pad_id) & (sx != self.eos_id))
                       | ((sy != self.pad_id) & (sy != self.eos_id)))
            s_mask = content.any(dim=(-2, -1))
        s_mask = s_mask.bool()

        psi_out = self.psi(sx, sy, s_mask=s_mask, task_id=task_id)
        loso_loss = torch.tensor(0.0, device=device)
        psi_out["loso_loss"] = loso_loss
        psi_out["s_mask"] = s_mask
        return psi_out, loso_loss


# ══════════════════════════════════════════════════════════════
# 10. SudokuEmbedder — adds 3×3 block/band/stack embeddings
# ══════════════════════════════════════════════════════════════

class SudokuEmbedder(ARCEmbedder):
    """
    ARCEmbedder + structural Sudoku embeddings.
      - emb_block : which 3×3 block (0..8)
      - emb_band  : horizontal band (row // 3 ∈ {0,1,2})
      - emb_stack : vertical stack (col // 3 ∈ {0,1,2})

    Applied only on 9×9 grids. For other sizes, falls back to base behavior.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        d = self.d

        self.emb_block = nn.Embedding(9, d)
        self.emb_band = nn.Embedding(3, d)
        self.emb_stack = nn.Embedding(3, d)
        for emb in (self.emb_block, self.emb_band, self.emb_stack):
            nn.init.normal_(emb.weight, std=0.02)

        rows = torch.arange(9)
        cols = torch.arange(9)
        block_ids = ((rows // 3)[:, None] * 3
                     + (cols // 3)[None, :]).reshape(81).long()
        band_ids = (rows // 3)[:, None].expand(9, 9).reshape(81).long()
        stack_ids = (cols // 3)[None, :].expand(9, 9).reshape(81).long()
        self.register_buffer("_block_ids", block_ids, persistent=False)
        self.register_buffer("_band_ids", band_ids, persistent=False)
        self.register_buffer("_stack_ids", stack_ids, persistent=False)

    def forward(self, x, type_id, pair_id=None, task_id=None,
                return_mask=False):
        if x.dtype != torch.long:
            x = x.long()
        B, H, W = x.shape
        L = H * W
        device = x.device
        x_flat = x.reshape(B, L)

        e = self.emb_tok(x_flat.clamp(0, self.vocab_size - 1))
        rows = torch.arange(H, device=device)
        cols = torch.arange(W, device=device)
        pos = (self.emb_row(rows)[:, None, :]
               + self.emb_col(cols)[None, :, :]).reshape(1, L, self.d)
        e = e + pos

        if H == 9 and W == 9:
            e = e + self.emb_block(self._block_ids.to(device)).unsqueeze(0)
            e = e + self.emb_band(self._band_ids.to(device)).unsqueeze(0)
            e = e + self.emb_stack(self._stack_ids.to(device)).unsqueeze(0)

        type_id_t = self._to_long_vector(type_id, B, device).clamp(
            0, self.emb_type.num_embeddings - 1)
        e = e + self.emb_type(type_id_t)[:, None, :]

        if pair_id is not None:
            pair_id_t = self._to_long_vector(pair_id, B, device).clamp(
                0, self.emb_pair.num_embeddings - 1)
            e = e + self.emb_pair(pair_id_t)[:, None, :]

        if self.use_task_id and task_id is not None:
            tid = self._to_long_vector(task_id, B, device).clamp(
                0, self.n_tasks - 1)
            e = e + self.emb_task(tid)[:, None, :]

        e = self.drop(self.norm(e))

        mask_keep = (x_flat != self.pad_id)
        all_pad = (~mask_keep).all(dim=1)
        if all_pad.any():
            mask_keep = mask_keep.clone()
            mask_keep[all_pad, 0] = True

        if return_mask:
            return e, mask_keep
        return e


# ══════════════════════════════════════════════════════════════
# 11. TokenMixerMLP — replaces phi.y_sa for Sudoku
# ══════════════════════════════════════════════════════════════

class TokenMixerMLP(nn.Module):
    """
    MLP-Mixer-style token mixing on the sequence dimension.

    For Sudoku (L=81), every cell has a fixed learnable weight to every
    other cell, encoding row/column/block constraints directly without
    having to learn attention patterns.

    Forward: [B, L, D] → transpose → fc1(L → h) → GELU → fc2(h → L) → transpose.

    Accepts and ignores key_padding_keep / grid_hw so that it is a
    drop-in replacement for MultiHeadSelfAttention.
    """

    def __init__(self, seq_len: int, d_model: int, expand: float = 2.0,
                 dropout: float = 0.1):
        super().__init__()
        hidden = int(seq_len * expand)
        self.fc1 = nn.Linear(seq_len, hidden)
        self.fc2 = nn.Linear(hidden, seq_len)
        self.drop = nn.Dropout(dropout)

        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x, *, key_padding_keep=None, grid_hw=None):
        xt = x.transpose(1, 2)          # [B, D, L]
        xt = self.drop(F.gelu(self.fc1(xt)))
        xt = self.fc2(xt)
        return xt.transpose(1, 2)        # [B, L, D]


# ══════════════════════════════════════════════════════════════
# 12. IERM_TRM_Sudoku — the Sudoku-specialized model
# ══════════════════════════════════════════════════════════════

class IERMSudoku(IERM):
    """
    Sudoku specialization.

    - phi.y_sa is replaced by a TokenMixerMLP (enabled by default).
    - predict() supports TTA via digit permutations (correctly de-permuted).
    """

    def __init__(self, embedder, *, seq_len: int = 81,
                 mixer_expand: float = 2.0,
                 use_token_mixer: bool = True,
                 **kwargs):
        super().__init__(embedder, **kwargs)
        self.use_token_mixer = bool(use_token_mixer)

        if self.use_token_mixer:
            D = self.phi.D
            dropout_p = float(self.phi.drop.p)
            # Replace phi.y_sa BEFORE load_state_dict is called, so that
            # checkpoint keys (phi.y_sa.fc1.*, phi.y_sa.fc2.*) match.
            self.phi.y_sa = TokenMixerMLP(
                seq_len=seq_len,
                d_model=D,
                expand=mixer_expand,
                dropout=dropout_p)
        # else: keep the inherited MultiHeadSelfAttention on phi.y_sa.

    # ────────────────────────────────────────────────────────
    # Inference
    # ────────────────────────────────────────────────────────

    def _predict_logits_internal(self, sx, sy, qx, *, s_mask=None,
                                  task_id=None, T_max=None,
                                  residual_alpha: float = 0.4):
        """Return raw logits [B, L, V] for one forward pass (no sampling)."""
        device = qx.device
        B, H, W = qx.shape
        L = H * W

        if s_mask is None:
            s_mask = torch.ones(B, sx.shape[1], device=device,
                                 dtype=torch.bool)

        psi_out, _ = self.encode_supports(
            sx, sy, s_mask=s_mask, task_id=task_id)
        ctx = psi_out["ctx"]
        ctx_keep = psi_out.get("ctx_keep")

        T = int(T_max or self.T_max)
        alpha = float(residual_alpha)

        x_embed, x_keep = self._embed_query(qx, task_id=task_id)
        x_ids = qx.reshape(B, L)
        y = x_embed
        z, z_keep = self.phi.init_z(B, device, x_embed.dtype)
        ctx_phi = self.phi.ctx_norm(self.phi.ctx_proj(ctx))

        for _ in range(T):
            y_new, z = self.phi.latent_recursion(
                x_embed, y, z, ctx_phi,
                x_keep=x_keep, y_keep=x_keep,
                z_keep=z_keep, ctx_keep=ctx_keep,
                grid_hw=(H, W))
            y = (1.0 - alpha) * y + alpha * y_new

        logits, q_hat, _, _, _ = self.phi._output_heads(
            y, x_keep, x_embed, x_ids)
        return logits, float(q_hat.mean().item())

    @torch.no_grad()
    def predict(self, sx, sy, qx, *,
                n_aug_passes: int = 8,
                s_mask=None,
                T_max: Optional[int] = None,
                confidence_threshold: float = 0.95,
                task_id=None,
                residual_alpha: float = 0.4,
                use_tta: bool = False,
                iterative_rounds: int = 1):
        """
        Predict the completed Sudoku grid.

        TTA de-permutation note
        ───────────────────────
        If `remap[v]` maps the original digit v to the permuted digit
        remap[v], and the network runs on permuted inputs, then the
        probability of the original digit v in the original space is
        the probability of digit remap[v] in the permuted output space.
        Indexing the permuted logits along the vocab dim with `remap`
        realizes this de-permutation exactly.
        """
        self.eval()
        aug_backup = self.aug_supports_prob
        self.aug_supports_prob = 0.0

        try:
            device = qx.device
            B, H, W = qx.shape
            V = self.vocab_size

            if s_mask is None:
                s_mask = torch.ones(B, sx.shape[1], device=device,
                                     dtype=torch.bool)
            s_mask = s_mask.to(device).bool()

            T = int(T_max or self.T_max)

            # Fast path: no TTA.
            if not use_tta:
                logits, conf = self._predict_logits_internal(
                    sx, sy, qx, s_mask=s_mask, task_id=task_id,
                    T_max=T, residual_alpha=residual_alpha)

                # Optional iterative decoding.
                for _ in range(iterative_rounds - 1):
                    pred_cur = logits.argmax(-1).reshape(B, H, W)
                    qx_next = qx.clone()
                    mask_empty = (qx == 0)
                    qx_next[mask_empty] = pred_cur[mask_empty]
                    logits, conf = self._predict_logits_internal(
                        sx, sy, qx_next, s_mask=s_mask, task_id=task_id,
                        T_max=T, residual_alpha=residual_alpha)

                return logits.argmax(-1).reshape(B, H, W), conf, T

            # TTA: average logits over digit permutations.
            logits_sum = torch.zeros(B, H * W, V, device=device)
            logits_orig, conf = self._predict_logits_internal(
                sx, sy, qx, s_mask=s_mask, task_id=task_id,
                T_max=T, residual_alpha=residual_alpha)
            logits_sum = logits_sum + logits_orig

            for _ in range(n_aug_passes):
                remap = torch.arange(V, device=device, dtype=torch.long)
                # Digits 1..9 are permuted; digit 0 (empty) stays fixed.
                remap[1:10] = torch.randperm(9, device=device) + 1

                sx_a = remap[sx.clamp(0, V - 1).long()]
                sy_a = remap[sy.clamp(0, V - 1).long()]
                qx_a = remap[qx.clamp(0, V - 1).long()]

                logits_a, _ = self._predict_logits_internal(
                    sx_a, sy_a, qx_a, s_mask=s_mask, task_id=task_id,
                    T_max=T, residual_alpha=residual_alpha)

                # De-permute: logits_orig[v] = logits_a[remap[v]].
                logits_sum = logits_sum + logits_a[:, :, remap]

            avg_logits = logits_sum / float(1 + n_aug_passes)
            pred = avg_logits.argmax(-1).reshape(B, H, W)
            return pred, conf, T
        finally:
            self.aug_supports_prob = aug_backup


# ══════════════════════════════════════════════════════════════
# 13. Default config + builder + loader
# ══════════════════════════════════════════════════════════════

# Config matching the shipped checkpoint (Sudoku Extreme > 63% with TTA).
# Verified via direct state_dict shape inspection (see scripts/dump_keys.py).
CFG_SUDOKU: dict = dict(
    vocab_size=10,
    pad_id=99,          # sentinel — never appears in Sudoku data
    eos_id=99,          # sentinel
    d_model=192,
    d_psi=120,
    n_heads=6,
    dropout=0.1,
    T_max=4,
    n_inner=6,
    n_mem=64,
    n_prog=32,
    psi_refine_layers=2,
    z_tokens=8,
    Hmax=9, Wmax=9,
    use_rope_2d=True,
    use_qk_norm=True,
    use_task_id=False,
    n_tasks=1,
    T_psi=2,
)


def build_sudoku_model(cfg: Optional[dict] = None,
                       *,
                       use_token_mixer: bool = True) -> IERM_TRM_Sudoku:
    """
    Build a fresh IERM_TRM_Sudoku according to `cfg`.

    Args:
        cfg : architecture config. Defaults to CFG_SUDOKU (paper config).
        use_token_mixer : if True (default), `phi.y_sa` is TokenMixerMLP.
            Set False to use self-attention; useful for training a
            baseline from scratch, but no such checkpoint is shipped.

    Returns:
        Model with the correct architecture in place BEFORE checkpoint
        load, so `load_state_dict(sd, strict=True)` works as expected.
    """
    if cfg is None:
        cfg = CFG_SUDOKU

    embedder = SudokuEmbedder(
        vocab_size=int(cfg["vocab_size"]),
        d=int(cfg["d_model"]),
        Hmax=int(cfg["Hmax"]),
        Wmax=int(cfg["Wmax"]),
        eos_id=int(cfg["eos_id"]),
        pad_id=int(cfg["pad_id"]),
        n_types=4,
        Smax=8,
        n_tasks=int(cfg.get("n_tasks", 1)),
        use_task_id=bool(cfg.get("use_task_id", False)),
        dropout=float(cfg["dropout"]),
    )

    seq_len = int(cfg["Hmax"]) * int(cfg["Wmax"])

    model = IERMSudoku(
        embedder,
        seq_len=seq_len,
        mixer_expand=2.0,
        use_token_mixer=bool(use_token_mixer),
        vocab_size=int(cfg["vocab_size"]),
        d_model=int(cfg["d_model"]),
        d_psi=int(cfg["d_psi"]),
        n_heads=int(cfg["n_heads"]),
        n_inner=int(cfg["n_inner"]),
        T_max=int(cfg["T_max"]),
        use_rope_2d=bool(cfg["use_rope_2d"]),
        use_qk_norm=bool(cfg["use_qk_norm"]),
        pad_id=int(cfg["pad_id"]),
        eos_id=int(cfg["eos_id"]),
        dropout=float(cfg["dropout"]),
        n_mem=int(cfg["n_mem"]),
        n_prog=int(cfg["n_prog"]),
        psi_refine_layers=int(cfg["psi_refine_layers"]),
        z_tokens=int(cfg["z_tokens"]),
        T_psi=int(cfg["T_psi"]),
        loso_lambda=0.10,
        loso_max_holds=1,
        support_verify_lambda=0.15,
        aug_supports_prob=0.0,
        aug_n_views=1,
    )
    return model


def load_sudoku_checkpoint(model: IERM_TRM_Sudoku, ckpt_path: str,
                           device: str = "cpu",
                           strict: bool = True,
                           strip_legacy_keys: bool = True) -> dict:
    """
    Load a checkpoint into `model`.

    Supports multiple top-level wrappers ('model_state_dict', 'model',
    'state_dict', 'ema'). Strips legacy `phi._original*` keys left over
    from an old monkey-patch (safe: never used in the forward pass).

    Returns the raw checkpoint dict (for step, metrics, etc.).
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = (ckpt.get("model_state_dict")
          or ckpt.get("model")
          or ckpt.get("ema")
          or ckpt.get("state_dict")
          or ckpt)

    if strip_legacy_keys:
        sd = {k: v for k, v in sd.items()
              if not k.startswith("phi._original")}

    missing, unexpected = model.load_state_dict(sd, strict=strict)
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"Checkpoint mismatch:\n"
            f"  missing keys  : {list(missing)[:10]}"
            f"{' ...' if len(missing) > 10 else ''}\n"
            f"  unexpected    : {list(unexpected)[:10]}"
            f"{' ...' if len(unexpected) > 10 else ''}"
        )
    return ckpt


__all__ = [
    # Primitives
    "RMSNorm", "SwiGLUFFN", "RoPE2D",
    "MultiHeadSelfAttention", "CrossAttention", "TRMSelfBlock",
    # Model components
    "ARCEmbedder", "PsiMemProg", "PhiTRM", "IERM",
    # Sudoku
    "SudokuEmbedder", "TokenMixerMLP", "IERMSudoku",
    # API
    "CFG_SUDOKU",
    "build_sudoku_model",
    "load_sudoku_checkpoint",
]