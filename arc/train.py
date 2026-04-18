from dataset import *
from model import *
from viz import *

import os, math, time
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F

SAVE_DIR = Path("checkpoints_v4_4")
SAVE_DIR.mkdir(exist_ok=True)


# ════════════════════════════════════════════════════════════════════
#  BUILD MODEL
# ════════════════════════════════════════════════════════════════════

def build_model():
    embedder = ARCEmbedder(
        vocab_size=CFG["vocab_size"],
        d=CFG["d_model"],
        Hmax=CFG["Hmax"],
        Wmax=CFG["Wmax"],
        eos_id=CFG["eos_id"],
        pad_id=CFG["pad_id"],
        n_types=4,
        Smax=8,
        n_tasks=CFG.get("n_tasks", 2048),
        use_task_id=CFG.get("use_task_id", True),
        dropout=CFG["dropout"],
    )
    model = IERM_TRM(
        embedder,
        vocab_size=CFG["vocab_size"],
        d_model=CFG["d_model"],
        d_psi=CFG.get("d_psi", CFG["d_model"] // 2),
        n_heads=CFG["n_heads"],
        n_inner=CFG.get("n_inner", 2),
        T_max=CFG["T_max"],
        use_rope_2d=CFG.get("use_rope_2d", True),
        use_qk_norm=CFG.get("use_qk_norm", True),
        pad_id=CFG["pad_id"],
        eos_id=CFG["eos_id"],
        dropout=CFG["dropout"],
        loso_lambda=TRAIN_CFG.get("loso_lambda", 0.15),
        support_verify_lambda=TRAIN_CFG.get("support_verify_lambda", 0.30),
        loso_max_holds=4,
        n_mem=CFG.get("n_mem", 64),
        n_prog=CFG.get("n_prog", 16),
        psi_refine_layers=CFG.get("psi_refine_layers", 1),
        z_tokens=CFG.get("z_tokens", 8),
        T_psi=CFG.get("T_psi", 2),
        aug_supports_prob=TRAIN_CFG.get("aug_supports_prob", 0.5),
        aug_n_views=TRAIN_CFG.get("aug_n_views", 2),
    )
    return model


# ════════════════════════════════════════════════════════════════════
#  EMA / SCHEDULER / CHECKPOINT
# ════════════════════════════════════════════════════════════════════

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = float(decay)
        self.shadow = {n: p.clone().detach() for n, p in model.named_parameters()}

    def update(self, model):
        for n, p in model.named_parameters():
            self.shadow[n].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def apply(self, model):
        self.backup = {n: p.data.clone() for n, p in model.named_parameters()}
        for n, p in model.named_parameters():
            p.data.copy_(self.shadow[n])

    def restore(self, model):
        for n, p in model.named_parameters():
            p.data.copy_(self.backup[n])


def build_warmup_cosine_restart_scheduler(optimizer, *, warmup_updates,
                                          cycle_updates, min_lr_ratio=0.05):
    warmup_updates = int(max(0, warmup_updates))
    cycle_updates = int(max(1, cycle_updates))

    def lr_lambda(step):
        if warmup_updates > 0 and step < warmup_updates:
            return (step + 1) / float(warmup_updates)
        t = step - warmup_updates
        phase = (t % cycle_updates) / float(cycle_updates)
        cos = 0.5 * (1.0 + math.cos(math.pi * phase))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cos

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _save_training_checkpoint(model, optimizer, scheduler, epoch, metrics, path):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "metrics": metrics,
        "config": CFG,
    }, path)


# ════════════════════════════════════════════════════════════════════
#  HELPERS LOSS
# ════════════════════════════════════════════════════════════════════

def _smoothed_nll_from_logprobs(log_probs, target, vocab_size, smoothing):
    """
    log_probs: [B, L, V] déjà en log_softmax
    target:    [B, L]
    retourne:  [B, L]
    """
    smooth = float(smoothing)
    onehot = F.one_hot(target.clamp(0, vocab_size - 1), vocab_size).float()
    onehot = onehot * (1.0 - smooth) + smooth / vocab_size
    return -(onehot * log_probs.float()).sum(-1)


def _plain_nll_from_logprobs(log_probs, target, vocab_size):
    B, L, V = log_probs.shape
    return F.nll_loss(
        log_probs.float().reshape(-1, V),
        target.clamp(0, vocab_size - 1).reshape(-1),
        reduction="none"
    ).reshape(B, L)


def _weighted_token_loss_from_logprobs(log_probs, target, token_weight,
                                       vocab_size, label_smoothing=0.0):
    if float(label_smoothing) > 0:
        ce_per_tok = _smoothed_nll_from_logprobs(
            log_probs, target, vocab_size, label_smoothing
        )
    else:
        ce_per_tok = _plain_nll_from_logprobs(log_probs, target, vocab_size)

    w_sum = token_weight.sum().clamp_min(1.0)
    return (ce_per_tok * token_weight).sum() / w_sum, ce_per_tok


# ════════════════════════════════════════════════════════════════════
#  TRAIN STEP — lm_head + deep supervision
# ════════════════════════════════════════════════════════════════════

def train_step_clean(
    batch, model, optimizer, *,
    scheduler=None,
    T_per_step=1,
    change_weight=4.0,
    unchanged_weight=1.0,
    eos_weight=1.5,
    grad_clip=0.5,
    augment=True,
    aug_prob=0.5,
    use_amp=True,
    scaler=None,
    amp_dtype="bf16",
    loso_lambda=0.15,
    support_verify_lambda=0.30,
    label_smoothing=0.03,
    mask_loss_weight=0.20,
    mask_temp=3.0,
    global_step=0,
    **_extra,
):
    model.train()
    device = next(model.parameters()).device
    pad_id = int(model.pad_id)
    eos_id = int(model.eos_id)
    V = int(model.vocab_size)

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

    B, H, W = qx.shape
    L = H * W
    T = int(T_per_step) if T_per_step > 0 else int(model.T_max)

    if augment and aug_prob > 0:
        sx, sy, qx, qy = augment_arc_colors(
            sx, sy, qx, qy, n_colors=10, aug_prob=float(aug_prob)
        )

    gt_flat = qy.reshape(B, L).long()
    qx_flat = qx.reshape(B, L).long()

    mask_valid = (gt_flat != pad_id)
    mask_changed = mask_valid & (gt_flat != qx_flat)
    mask_eos = mask_valid & (gt_flat == eos_id)
    mask_unch = mask_valid & ~mask_changed & ~mask_eos

    token_weight = torch.zeros(B, L, device=device)
    token_weight[mask_changed] = float(change_weight)
    token_weight[mask_unch] = float(unchanged_weight)
    token_weight[mask_eos] = float(eos_weight)

    amp_dt = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    optimizer.zero_grad(set_to_none=True)

    model._residual_alpha = float(_extra.get("residual_alpha", 0.4))
    n_grad_steps = int(_extra.get("n_grad_steps", 2))
    use_deep_supervision = bool(_extra.get("use_deep_supervision", False))
    ds_weight = float(_extra.get("ds_weight", 0.0))
    ds_decay = float(_extra.get("ds_decay", 0.7))

    with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dt):

        psi_out, loso_loss = model.encode_supports(
            sx, sy, s_mask=s_mask, task_id=task_id
        )

        sv_loss = torch.tensor(0.0, device=device)
        if float(support_verify_lambda) > 0:
            sv_loss = model._support_verify_loss(
                sx, sy, s_mask, psi_out,
                task_id=task_id, T=1
            )

        logits, q_hat, y_det, z_det, aux = model(
            qx, ctx=psi_out["ctx"], ctx_keep=psi_out["ctx_keep"],
            T=T, task_id=task_id,
            n_grad_steps=n_grad_steps,
            return_all_steps=use_deep_supervision,
        )

        # ── Main CE pondérée ─────────────────────────────────────────────
        loss_ce, ce_per_tok = _weighted_token_loss_from_logprobs(
            logits, gt_flat, token_weight, V,
            label_smoothing=float(label_smoothing)
        )

        # ── Hard-negative mining ─────────────────────────────────────────
        hard_mining_factor = float(_extra.get("hard_mining_factor", 1.0))
        if hard_mining_factor > 1.0:
            with torch.no_grad():
                pred_ids = logits.argmax(-1)
                conf_max = logits.float().exp().max(-1).values
                is_conf_wrong = (pred_ids != gt_flat) & mask_changed & (conf_max > 0.5)

            token_weight_hm = token_weight.clone()
            token_weight_hm[is_conf_wrong] *= hard_mining_factor
            loss_ce, ce_per_tok = _weighted_token_loss_from_logprobs(
                logits, gt_flat, token_weight_hm, V,
                label_smoothing=float(label_smoothing)
            )
            token_weight_used = token_weight_hm
        else:
            token_weight_used = token_weight

        loss = (
            loss_ce
            + float(loso_lambda) * loso_loss
            + float(support_verify_lambda) * sv_loss
        )

        # ── q_head loss ──────────────────────────────────────────────────
        pred_train = logits.argmax(-1)
        solved_target = ((pred_train == gt_flat) | ~mask_valid).all(1).float()

        q_logit = aux.get("q_logit")
        loss_q = torch.tensor(0.0, device=device)
        if q_logit is not None:
            loss_q = F.binary_cross_entropy_with_logits(
                q_logit.float(), solved_target
            )
            loss = loss + 0.05 * loss_q

        # ── Deep supervision réelle ──────────────────────────────────────
        loss_ds = torch.tensor(0.0, device=device)
        logits_per_step = aux.get("logits_per_step")
        q_logit_per_step = aux.get("q_logit_per_step")

        if use_deep_supervision and ds_weight > 0.0 and logits_per_step is not None and len(logits_per_step) > 0:
            K = len(logits_per_step)
            weights = [ds_decay ** (K - 1 - i) for i in range(K)]
            wnorm = sum(weights)

            ds_accum = torch.tensor(0.0, device=device)
            for w_i, step_logits in zip(weights, logits_per_step):
                step_loss, _ = _weighted_token_loss_from_logprobs(
                    step_logits, gt_flat, token_weight_used, V,
                    label_smoothing=float(label_smoothing)
                )
                ds_accum = ds_accum + float(w_i) * step_loss

            loss_ds = ds_accum / max(wnorm, 1e-8)
            loss = loss + ds_weight * loss_ds

            q_ds_weight = float(_extra.get("q_ds_weight", 0.0))
            if q_ds_weight > 0.0 and q_logit_per_step is not None and len(q_logit_per_step) == len(logits_per_step):
                qds_accum = torch.tensor(0.0, device=device)
                for w_i, ql in zip(weights, q_logit_per_step):
                    qds_accum = qds_accum + float(w_i) * F.binary_cross_entropy_with_logits(
                        ql.float(), solved_target
                    )
                loss_q_ds = qds_accum / max(wnorm, 1e-8)
                loss = loss + q_ds_weight * loss_q_ds

        # ── Code mort copy/rewrite : gardé neutre pour compat ───────────
        MASK_TEMP = float(mask_temp)
        m_logits_raw = aux.get("m_logits")
        loss_mask = torch.tensor(0.0, device=device)
        if m_logits_raw is not None:
            mask_target = torch.zeros_like(m_logits_raw.float())
            mask_target[mask_changed] = 1.0
            mask_for_loss = mask_valid
            if mask_for_loss.any():
                loss_mask = F.binary_cross_entropy_with_logits(
                    (m_logits_raw.float() * MASK_TEMP)[mask_for_loss],
                    mask_target[mask_for_loss],
                    reduction="mean"
                )
            loss = loss + float(mask_loss_weight) * loss_mask

        val_loss_weight = float(_extra.get("val_loss_weight", 0.0))
        loss_val = torch.tensor(0.0, device=device)
        if val_loss_weight > 0.0:
            v_logits = aux.get("v_logits")
            if v_logits is not None:
                val_ce = F.nll_loss(
                    v_logits.float().reshape(-1, V),
                    gt_flat.clamp(0, V - 1).reshape(-1),
                    reduction="none"
                ).reshape(B, L)
                w_sum = token_weight_used.sum().clamp_min(1.0)
                loss_val = (val_ce * token_weight_used).sum() / w_sum
                loss = loss + val_loss_weight * loss_val

    # ── Backward ────────────────────────────────────────────────────────
    if use_amp and amp_dtype == "fp16" and scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        optimizer.step()

    if scheduler is not None:
        scheduler.step()

    # ── Métriques ───────────────────────────────────────────────────────
    with torch.no_grad():
        pred_mix = logits.argmax(-1)
        v_logits = aux.get("v_logits")
        m_logits = aux.get("m_logits")

        dv = mask_valid.float().sum().clamp_min(1)
        px = float(((pred_mix == gt_flat) & mask_valid).float().sum() / dv)

        mask_content = mask_valid & (gt_flat != eos_id)
        dc = mask_content.float().sum().clamp_min(1)
        px_content = float(((pred_mix == gt_flat) & mask_content).float().sum() / dc)

        dch = mask_changed.float().sum().clamp_min(1)
        ch = float(((pred_mix == gt_flat) & mask_changed).float().sum() / dch)

        mask_unch_met = mask_valid & ~mask_changed & (gt_flat != eos_id)
        d_unch = mask_unch_met.float().sum().clamp_min(1)
        unch_acc = float(((pred_mix == gt_flat) & mask_unch_met).float().sum() / d_unch)

        ch_v = float("nan")
        if v_logits is not None:
            pred_v = v_logits.argmax(-1)
            ch_v = float(((pred_v == gt_flat) & mask_changed).float().sum() / dch)

        solved = float(((pred_mix == gt_flat) | ~mask_valid).all(1).float().mean())
        err_per_grid = ((pred_mix != gt_flat) & mask_valid).sum(1)
        solved_at_2 = float((err_per_grid <= 2).float().mean())
        solved_at_5 = float((err_per_grid <= 5).float().mean())

        m_unch = float("nan")
        m_ch = float("nan")
        if m_logits is not None:
            m = torch.sigmoid(m_logits.float() * MASK_TEMP)
            mask_unch_for_m = mask_valid & ~mask_changed
            if mask_unch_for_m.any():
                m_unch = float(m[mask_unch_for_m].mean())
            if mask_changed.any():
                m_ch = float(m[mask_changed].mean())

    return {
        "loss": float(loss.item()),
        "ce_loss": float(loss_ce.item()),
        "ds_loss": float(loss_ds.item()),
        "sv_loss": float(sv_loss.item()),
        "loso_loss": float(loso_loss.item()),
        "val_loss": float(loss_val.item()),
        "px_acc": px,
        "px_content": px_content,
        "change_acc": ch,
        "unchanged_acc": unch_acc,
        "change_acc_v": ch_v,
        "solved_strict": solved,
        "solved_at_2": solved_at_2,
        "solved_at_5": solved_at_5,
        "m_prob_unch": m_unch,
        "m_prob_ch": m_ch,
        "confidence": float(q_hat.mean().item()),
        "q_loss": float(loss_q.item()),
        "mask_loss": float(loss_mask.item()),
    }


# ════════════════════════════════════════════════════════════════════
#  EVALUATE_SEQ
# ════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_seq(
    model, loader, device, *, cfg=None, max_batches=None,
    pred_T=None, confidence_threshold=0.95,
    print_diag=True, run_recursion_diag=True,
    diag_max_batches=8, near_solved_ks=(1, 2, 3, 5, 10),
):
    model.eval()

    _aug_prob_backup = getattr(model, "aug_supports_prob", 0.0)
    model.aug_supports_prob = 0.0

    pad_id = int(cfg["pad_id"]) if cfg else int(getattr(model, "pad_id", 11))
    eos_id = int(cfg["eos_id"]) if cfg else int(getattr(model, "eos_id", 10))
    T_max = int(cfg.get("T_max", getattr(model, "T_max", 3))) if cfg else 3
    n_inner = int(getattr(model.phi, "n_inner", 2))
    T_psi = int(getattr(model.psi, "T_psi", 1))
    T_eval = int(pred_T) if pred_T is not None else T_max
    V = int(cfg["vocab_size"]) if cfg else 12

    totals = defaultdict(float)
    n_grids = 0
    err_hist = defaultdict(int)
    mprob_unch_sum = 0.0
    mprob_ch_sum = 0.0
    mprob_cnt = 0

    diag_px = defaultdict(float)
    diag_ch = defaultdict(float)
    diag_cnt = 0
    diag_ctx = defaultdict(float)
    diag_ctx_cnt = 0

    MASK_TEMP = float(cfg.get("mask_temp", 3.0)) if cfg else 3.0

    def _px_ch(pred_flat, gt_flat, qx_flat, mask_valid):
        changed = mask_valid & (gt_flat != qx_flat)
        denom = int(mask_valid.sum().item())
        px = float(((pred_flat == gt_flat) & mask_valid).sum().item() / max(denom, 1))
        denom_ch = int(changed.sum().item())
        ch = float(((pred_flat == gt_flat) & changed).sum().item() / max(denom_ch, 1)) if denom_ch > 0 else float("nan")
        return px, ch

    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break

        sx = batch["sx"].to(device)
        sy = batch["sy"].to(device)
        qx = batch["qx"].to(device)
        qy = batch["qy"].to(device)

        B, H, W = qy.shape
        L = H * W
        n_grids += B

        s_mask = batch.get("s_mask")
        if s_mask is not None:
            s_mask = s_mask.to(device).bool()

        task_id = batch.get("task_id")
        if isinstance(task_id, torch.Tensor):
            task_id = task_id.to(device)

        gt_flat = qy.reshape(B, L).long()
        qx_flat = qx.reshape(B, L).long()
        mask_valid = (gt_flat != pad_id)
        mask_content = mask_valid & (gt_flat != eos_id)
        mask_changed = mask_valid & (gt_flat != qx_flat)

        psi_out, _ = model.encode_supports(sx, sy, s_mask=s_mask, task_id=task_id)
        ctx = psi_out["ctx"]
        ctx_keep = psi_out.get("ctx_keep")
        supp = psi_out.get("supp")
        supp_keep = psi_out.get("supp_keep")

        logits, q_hat, y_det, z_det, aux = model(
            qx, ctx=ctx, ctx_keep=ctx_keep,
            T=T_eval, task_id=task_id,
            n_grad_steps=min(T_eval, int(getattr(model, "T_max", T_eval))),
            return_all_steps=False,
        )

        pr_flat = logits.argmax(-1)

        denom = int(mask_valid.sum().item())
        px = float(((pr_flat == gt_flat) & mask_valid).sum().item() / max(denom, 1))

        denom_c = int(mask_content.sum().item())
        px_c = float(((pr_flat == gt_flat) & mask_content).sum().item() / max(denom_c, 1)) if denom_c else px

        denom_ch = int(mask_changed.sum().item())
        ch_acc = float(((pr_flat == gt_flat) & mask_changed).sum().item() / max(denom_ch, 1)) if denom_ch > 0 else float("nan")

        mask_unch = mask_valid & ~mask_changed & (gt_flat != eos_id)
        denom_unch = int(mask_unch.sum().item())
        unch_acc = float(((pr_flat == gt_flat) & mask_unch).sum().item() / max(denom_unch, 1)) if denom_unch > 0 else float("nan")

        solved = float(((pr_flat == gt_flat) | ~mask_valid).all(1).float().mean().item())

        err_per_grid = ((pr_flat != gt_flat) & mask_valid).sum(1).long()
        for k in near_solved_ks:
            totals[f"near_solved_{k}_sum"] += float((err_per_grid <= k).float().sum().item())
        for ev in err_per_grid.tolist():
            bucket = int(ev) if int(ev) <= 10 else 11
            err_hist[bucket] += 1

        totals["px_sum"] += px * B
        totals["px_content_sum"] += px_c * B
        totals["ch_sum"] += (ch_acc if not math.isnan(ch_acc) else 0.0) * B
        totals["ch_valid"] += B if not math.isnan(ch_acc) else 0
        totals["unch_sum"] += (unch_acc if not math.isnan(unch_acc) else 0.0) * B
        totals["unch_valid"] += B if not math.isnan(unch_acc) else 0
        totals["solved_sum"] += solved * B

        idx = mask_valid.reshape(-1)
        if idx.any():
            ce = F.nll_loss(
                logits.reshape(-1, V)[idx],
                gt_flat.reshape(-1)[idx].clamp(0, V - 1)
            )
        else:
            ce = logits.sum() * 0.0
        totals["loss_sum"] += float(ce.item()) * B

        m_logits_eval = aux.get("m_logits")
        if m_logits_eval is not None:
            m = torch.sigmoid(m_logits_eval.float() * MASK_TEMP)
            mask_unch_m = mask_valid & ~mask_changed
            mprob_unch_sum += float(m[mask_unch_m].mean().item()) if mask_unch_m.any() else 0.0
            mprob_ch_sum += float(m[mask_changed].mean().item()) if mask_changed.any() else 0.0
            mprob_cnt += 1

        v_logits_eval = aux.get("v_logits")
        if v_logits_eval is not None:
            v_pred = v_logits_eval.argmax(-1)
            v_ch = float(((v_pred == gt_flat) & mask_changed).sum().item() / max(denom_ch, 1))
            totals["v_ch_sum"] += v_ch * B
            totals["v_ch_valid"] += B

        # ── Recursion diagnostic ────────────────────────────────────────
        if run_recursion_diag and diag_cnt < diag_max_batches:
            x_embed, x_keep = model._embed_query(qx, task_id=task_id)
            x_ids = qx.reshape(B, L)

            logits_t0, _, _, _, _ = model.phi._output_heads(
                x_embed, x_keep, x_embed, x_ids
            )
            p0 = logits_t0.argmax(-1)
            px_t0, ch_t0 = _px_ch(p0, gt_flat, qx_flat, mask_valid)
            diag_px[0] += px_t0
            diag_ch[0] += ch_t0 if not math.isnan(ch_t0) else 0.0

            for t in range(1, T_eval + 1):
                logits_t, *_ = model(
                    qx, ctx=ctx, ctx_keep=ctx_keep,
                    T=t, task_id=task_id,
                    n_grad_steps=min(t, int(getattr(model, "T_max", t))),
                    return_all_steps=False,
                )
                pt = logits_t.argmax(-1)
                px_t, ch_t = _px_ch(pt, gt_flat, qx_flat, mask_valid)
                diag_px[t] += px_t
                diag_ch[t] += ch_t if not math.isnan(ch_t) else 0.0
            diag_cnt += 1

        # ── Ctx diagnostic ──────────────────────────────────────────────
        if run_recursion_diag and diag_ctx_cnt < diag_max_batches:
            ctx_noise = torch.randn_like(ctx)

            logits_real, *_ = model(
                qx, ctx=ctx, ctx_keep=ctx_keep,
                T=T_eval, task_id=task_id,
                n_grad_steps=min(T_eval, int(getattr(model, "T_max", T_eval))),
                return_all_steps=False,
            )
            logits_noise, *_ = model(
                qx, ctx=ctx_noise, ctx_keep=ctx_keep,
                T=T_eval, task_id=task_id,
                n_grad_steps=min(T_eval, int(getattr(model, "T_max", T_eval))),
                return_all_steps=False,
            )

            pr_real = logits_real.argmax(-1)
            pr_noise = logits_noise.argmax(-1)
            px_r, ch_r = _px_ch(pr_real, gt_flat, qx_flat, mask_valid)
            px_n, ch_n = _px_ch(pr_noise, gt_flat, qx_flat, mask_valid)

            diag_ctx["px_real"] += px_r
            diag_ctx["ch_real"] += ch_r if not math.isnan(ch_r) else 0.0
            diag_ctx["px_noise"] += px_n
            diag_ctx["ch_noise"] += ch_n if not math.isnan(ch_n) else 0.0
            diag_ctx_cnt += 1

    model.aug_supports_prob = _aug_prob_backup

    ng = max(n_grids, 1)
    ch_valid = max(int(totals["ch_valid"]), 1)
    unch_valid = max(int(totals.get("unch_valid", 1)), 1)

    results = {
        "tasks": int(n_grids),
        "loss": float(totals["loss_sum"] / ng),
        "px": float(totals["px_sum"] / ng),
        "px_content": float(totals["px_content_sum"] / ng),
        "change_acc": float(totals["ch_sum"] / ch_valid),
        "unchanged_acc": float(totals.get("unch_sum", 0) / unch_valid),
        "grid_solved": float(totals["solved_sum"] / ng),
        "T_used": T_eval,
        **{f"near_solved_{k}": float(totals[f"near_solved_{k}_sum"] / ng) for k in near_solved_ks},
        "err_hist": {
            (str(k) if k <= 10 else ">10"): err_hist[k] / ng
            for k in sorted(err_hist)
        },
        "mean_m_prob_unchanged": float(mprob_unch_sum / max(mprob_cnt, 1)),
        "mean_m_prob_changed": float(mprob_ch_sum / max(mprob_cnt, 1)),
        "v_change_acc": float(totals.get("v_ch_sum", 0) / max(int(totals.get("v_ch_valid", 1)), 1)),
        "recursion_gain": 0.0,
        "psi_ctx_lift": 0.0,
        "psi_change_acc": float("nan"),
    }

    if print_diag and n_grids:
        print("\n  ┌─ EVAL ─────────────────────────────────────────────────────")
        print(f"  │ T_eval={T_eval}  n_inner={n_inner}  T_psi={T_psi}")
        print(f"  │ loss          = {results['loss']:.4f}")
        print(f"  │ px            = {results['px']:.4f}    (gt≠PAD)")
        print(f"  │ px_content    = {results['px_content']:.4f}  (gt∉{{PAD,EOS}})")

        ch = results["change_acc"]
        ch_flag = "  ← CRITIQUE" if ch < 0.30 else ("  ✓" if ch > 0.60 else "")
        print(f"  │ change_acc    = {ch:.4f}{ch_flag}  (pred_mix)")

        uc = results["unchanged_acc"]
        uc_flag = "  ← DÉGRADÉ" if uc < 0.90 else ("  ✓" if uc > 0.95 else "")
        print(f"  │ unchanged_acc = {uc:.4f}{uc_flag}  (copie)")

        v_ch = results.get("v_change_acc", 0)
        gap = v_ch - ch
        print(f"  │ v_change_acc  = {v_ch:.4f}  (val_head, gap={gap:+.4f})")
        print(f"  │ grid_solved   = {results['grid_solved']:.4f}")

        ns_parts = "  ".join(f"≤{k}:{results[f'near_solved_{k}']:.3f}" for k in near_solved_ks)
        print(f"  │ near_solved   : {ns_parts}")

        hist = results["err_hist"]
        hist_str = "  ".join(
            f"e={k}:{v:.3f}"
            for k, v in sorted(hist.items(), key=lambda x: (x[0] == ">10", str(x[0]).zfill(3)))
        )
        print(f"  │ err_hist      : {hist_str}")
        print(f"  │ mask_prob     : unchanged={results['mean_m_prob_unchanged']:.3f}  "
              f"changed={results['mean_m_prob_changed']:.3f}")
        print(f"  └────────────────────────────────────────────────────────────\n")

    if run_recursion_diag and diag_cnt > 0 and print_diag:
        dc = max(diag_cnt, 1)
        px_t0 = diag_px[0] / dc
        ch_t0 = diag_ch[0] / dc
        px_end = diag_px[T_eval] / dc
        total_gain = px_end - px_t0
        results["recursion_gain"] = float(total_gain)

        print(f"  ┌── Récursion Φ : T=0 → T={T_eval} ─────────────────────────")
        print(f"  │ T=0 (no Φ) : px={px_t0:.4f}  ch={ch_t0:.4f}")
        prev_px = px_t0
        for t in range(1, T_eval + 1):
            px_t = diag_px[t] / dc
            ch_t = diag_ch[t] / dc
            delta = px_t - prev_px
            arrow = "↑" if delta > 0.001 else ("↓" if delta < -0.001 else "→")
            print(f"  │ T={t}        : px={px_t:.4f}  ch={ch_t:.4f}  {arrow}{delta:+.4f}")
            prev_px = px_t
        label = ("✓ Φ améliore" if total_gain > 0.02
                 else ("⚠ gain ≈ 0" if total_gain > -0.01 else "✗ Φ dégrade"))
        print(f"  │ {label} px de {total_gain:+.4f}")
        print(f"  └────────────────────────────────────────────────────────────\n")

    if run_recursion_diag and diag_ctx_cnt > 0 and print_diag:
        dc = max(diag_ctx_cnt, 1)
        px_r = diag_ctx["px_real"] / dc
        ch_r = diag_ctx["ch_real"] / dc
        px_n = diag_ctx["px_noise"] / dc
        ch_n = diag_ctx["ch_noise"] / dc
        lift_ctx = ch_r - ch_n

        results["psi_ctx_lift"] = float(lift_ctx)
        results["psi_change_acc"] = float(ch_r)

        print(f"  ┌── Ψ interactive : ctx_real vs ctx_noise ───────────────────")
        print(f"  │ T_psi={T_psi}")
        print(f"  │ ctx_real  : px={px_r:.4f}  ch={ch_r:.4f}")
        print(f"  │ ctx_noise : px={px_n:.4f}  ch={ch_n:.4f}")
        lift_flag = "✓" if lift_ctx > 0.10 else ("⚠" if lift_ctx > 0.03 else "✗")
        print(f"  │ Ψ_lift = {lift_ctx:+.4f}  {lift_flag}")
        print(f"  └────────────────────────────────────────────────────────────\n")

    model.train()
    return results


# ════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ════════════════════════════════════════════════════════════════════

def run_training_seq(**overrides):
    cfg = dict(TRAIN_CFG)
    cfg.update(overrides)

    global model
    model = build_model().to(DEVICE)

    n_total = sum(p.numel() for p in model.parameters())
    n_psi = sum(p.numel() for n, p in model.named_parameters() if "psi" in n)
    n_phi = sum(p.numel() for n, p in model.named_parameters() if "phi" in n)
    n_emb = sum(p.numel() for n, p in model.named_parameters() if "embedder" in n)

    print(f"\n  [IERM-TRM v4.3] params: total={n_total:,}  emb={n_emb:,}  "
          f"psi={n_psi:,}  phi={n_phi:,}")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    use_amp = bool(cfg.get("use_amp", True))
    amp_dtype = cfg.get("amp_dtype", "bf16")
    scaler = torch.amp.GradScaler(enabled=(use_amp and amp_dtype == "fp16"))

    # ── Resume ────────────────────────────────────────────────────────
    resume_path = cfg.get("resume_from")
    if resume_path and os.path.exists(str(resume_path)):
        ckpt = torch.load(resume_path, map_location=DEVICE, weights_only=False)
        old_sd = ckpt.get("model_state_dict", ckpt)
        new_sd = model.state_dict()

        loaded = skipped = new_params = 0
        for k in new_sd:
            if k in old_sd and old_sd[k].shape == new_sd[k].shape:
                new_sd[k] = old_sd[k]
                loaded += 1
            elif k in old_sd:
                skipped += 1
                print(f"    ⚠ shape mismatch: {k}  old={old_sd[k].shape} new={new_sd[k].shape}")
            else:
                new_params += 1
                print(f"    ★ new param: {k}")

        model.load_state_dict(new_sd)
        print(f"  Resumed ({loaded}/{len(new_sd)} params, "
              f"{skipped} shape-mismatch, {new_params} new [lm_head+DS])")

        old_gate = model.phi.z_recalib_gate.data[0].item()
        if old_gate < -1.5:
            with torch.no_grad():
                model.phi.z_recalib_gate.fill_(-1.0)
            print(f"  ⚡ z_recalib_gate: {old_gate:.2f} → -1.00")
    else:
        print(f"  ★ Training FROM SCRATCH")

    with torch.no_grad():
        gate_sig = torch.sigmoid(model.phi.z_recalib_gate).mean().item()
        p2m_sig = torch.sigmoid(model.psi.p2m_gate).mean().item()

    print(f"  z_recalib_gate sigmoid={gate_sig:.3f}")
    print(f"  p2m_gate sigmoid={p2m_sig:.3f}  (feedback P→M)")
    print(f"  T_psi={model.psi.T_psi}  aug_n_views={model.aug_n_views}")
    print(f"  support_verify_lambda={cfg.get('support_verify_lambda', 0.3)}")
    print(f"  aug_supports_prob={model.aug_supports_prob}  aug_n_views={model.aug_n_views}")
    print(f"  use_deep_supervision={cfg.get('use_deep_supervision', False)}  "
          f"ds_weight={cfg.get('ds_weight', 0.0)}  ds_decay={cfg.get('ds_decay', 0.7)}")
    print(f"  n_grad_steps={cfg.get('n_grad_steps', 2)}")

    steps_per_epoch = len(trainext_loader)
    train_eval_max = int(cfg.get("train_eval_max_batches", 30))

    # ── Optimizer ─────────────────────────────────────────────────────
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name for nd in ["bias", "scale", "norm", "emb_"]):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": float(cfg["weight_decay"])},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=float(cfg["lr"]), betas=(0.9, 0.98))

    warmup_updates = int(cfg.get("warmup_steps", 0))
    cycle_updates = int(cfg.get("cycle_epochs", 1)) * max(steps_per_epoch, 1)
    scheduler = build_warmup_cosine_restart_scheduler(
        optimizer, warmup_updates=warmup_updates,
        cycle_updates=cycle_updates, min_lr_ratio=0.05
    )

    ema = EMA(model, decay=float(cfg.get("ema_decay", 0.999)))

    tracker = defaultdict(list)
    best_eval_solved = -1.0
    best_eval_px = -1.0
    patience_counter = 0
    global_update = 0
    start_time = time.time()

    print(f"{'=' * 70}")
    print(
        f"  IERM-TRM v4.3 — T_psi={CFG.get('T_psi', 2)}"
        f"  aug_views={cfg.get('aug_n_views', 2)}"
        f"  sv={cfg.get('support_verify_lambda')}"
        f"  loso={cfg.get('loso_lambda')}"
        f"  chg_w={cfg['change_weight']}"
        f"  DS={cfg.get('use_deep_supervision', False)}"
        f"  ds_w={cfg.get('ds_weight', 0.0)}"
    )
    print(f"{'=' * 70}\n")

    for epoch in range(int(cfg["epochs"])):
        apply_curriculum(epoch, cfg, model=model)
        model.loso_lambda = float(cfg.get("loso_lambda", 0.15))
        model.support_verify_lambda = float(cfg.get("support_verify_lambda", 0.30))

        model.train()
        epoch_metrics = defaultdict(float)
        n_batches = 0

        for batch in trainext_loader:
            batch = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}

            metrics = train_step_clean(
                batch, model, optimizer,
                scheduler=scheduler,
                T_per_step=int(cfg.get("T_per_step", 1)),
                change_weight=float(cfg.get("change_weight", 4.0)),
                unchanged_weight=float(cfg.get("unchanged_weight", 1.0)),
                eos_weight=float(cfg.get("eos_weight", 1.5)),
                grad_clip=float(cfg.get("grad_clip", 0.5)),
                augment=bool(cfg.get("augment", True)),
                aug_prob=float(cfg.get("aug_prob", 0.70)),
                use_amp=use_amp,
                scaler=scaler,
                amp_dtype=amp_dtype,
                loso_lambda=float(cfg.get("loso_lambda", 0.15)),
                support_verify_lambda=float(cfg.get("support_verify_lambda", 0.30)),
                label_smoothing=float(cfg.get("label_smoothing", 0.03)),
                mask_loss_weight=float(cfg.get("mask_loss_weight", 0.20)),
                val_loss_weight=float(cfg.get("val_loss_weight", 0.0)),
                mask_temp=float(cfg.get("mask_temp", 3.0)),
                global_step=global_update,
                residual_alpha=float(cfg.get("residual_alpha", 0.4)),
                hard_mining_factor=float(cfg.get("hard_mining_factor", 1.0)),
                n_grad_steps=int(cfg.get("n_grad_steps", 2)),
                use_deep_supervision=bool(cfg.get("use_deep_supervision", False)),
                ds_weight=float(cfg.get("ds_weight", 0.0)),
                ds_decay=float(cfg.get("ds_decay", 0.7)),
                q_ds_weight=float(cfg.get("q_ds_weight", 0.0)),
            )

            ema.update(model)
            global_update += 1

            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    epoch_metrics[k] += float(v)
            n_batches += 1

            if global_update % int(cfg["log_every"]) == 0:
                lr_now = float(optimizer.param_groups[0]["lr"])
                print(
                    f"  upd {global_update:6d} | "
                    f"loss={metrics.get('loss', 0):.4f} "
                    f"ce={metrics.get('ce_loss', 0):.4f} "
                    f"ds={metrics.get('ds_loss', 0):.4f} "
                    f"sv={metrics.get('sv_loss', 0):.4f} "
                    f"px={metrics.get('px_acc', 0):.3f} "
                    f"ch={metrics.get('change_acc', 0):.3f} "
                    f"uc={metrics.get('unchanged_acc', 0):.3f} "
                    f"chV={metrics.get('change_acc_v', 0):.3f} "
                    f"sol={metrics.get('solved_strict', 0):.4f} "
                    f"m_u={metrics.get('m_prob_unch', 0):.3f} "
                    f"m_c={metrics.get('m_prob_ch', 0):.3f} "
                    f"| lr={lr_now:.2e}"
                )

        nb = max(n_batches, 1)
        train_avg = {k: v / nb for k, v in epoch_metrics.items()}

        for key in [
            "loss", "ce_loss", "ds_loss", "sv_loss", "loso_loss",
            "px_acc", "px_content", "change_acc",
            "solved_strict", "solved_at_2", "solved_at_5"
        ]:
            tracker[f"train_{key}"].append(float(train_avg.get(key, 0.0)))
        tracker["train_m_prob_unch"].append(float(train_avg.get("m_prob_unch", 0.0)))
        tracker["train_m_prob_ch"].append(float(train_avg.get("m_prob_ch", 0.0)))

        # ── Eval ────────────────────────────────────────────────────────
        if epoch % int(cfg["eval_every"]) == 0:
            T_eval_now = int(cfg.get("T_per_step", 1))

            ema.apply(model)

            train_eval_avg = evaluate_seq(
                model, train_loader, DEVICE, cfg=CFG,
                max_batches=train_eval_max,
                pred_T=T_eval_now,
                print_diag=False, run_recursion_diag=False
            )

            eval_avg = evaluate_seq(
                model, eval_loader, DEVICE, cfg=CFG,
                max_batches=None,
                pred_T=T_eval_now,
                print_diag=True, run_recursion_diag=True,
                diag_max_batches=8
            )

            ema.restore(model)

            tracker["eval_epoch"].append(int(epoch))
            for key in ["loss", "px", "px_content", "change_acc", "grid_solved"]:
                tracker[f"train_eval_{key}"].append(float(train_eval_avg.get(key, 0.0)))
                tracker[f"eval_{key}"].append(float(eval_avg.get(key, 0.0)))

            tracker["eval_psi_ctx_lift"].append(float(eval_avg.get("psi_ctx_lift", 0.0)))
            tracker["eval_recursion_gain"].append(float(eval_avg.get("recursion_gain", 0.0)))
            tracker["eval_m_prob_changed"].append(float(eval_avg.get("mean_m_prob_changed", 0.0)))
            tracker["eval_m_prob_unchanged"].append(float(eval_avg.get("mean_m_prob_unchanged", 0.0)))

            print(f"  ┌─ TRAIN-EVAL ep {epoch} ──────────────────────────────────")
            print(f"  │ loss={train_eval_avg['loss']:.4f}  "
                  f"px={train_eval_avg['px']:.4f}  "
                  f"pxC={train_eval_avg['px_content']:.4f}  "
                  f"solved={train_eval_avg['grid_solved']:.4f}")
            print(f"  │ T_psi={model.psi.T_psi}  aug_prob={model.aug_supports_prob:.2f}"
                  f"  aug_views={model.aug_n_views}")
            print(f"  └──────────────────────────────────────────────────────\n")

            eval_solved = eval_avg["grid_solved"]
            eval_px_now = eval_avg["px"]
            improved = (
                eval_solved > best_eval_solved + 1e-9
            ) or (
                abs(eval_solved - best_eval_solved) < 1e-9
                and eval_px_now > best_eval_px + 1e-9
            )

            if improved:
                best_eval_solved = float(eval_solved)
                best_eval_px = float(eval_px_now)
                patience_counter = 0

                ema.apply(model)
                _save_training_checkpoint(
                    model, optimizer, scheduler, epoch,
                    {"eval": eval_avg, "train_eval": train_eval_avg},
                    os.path.join(str(SAVE_DIR), "best_ema.pt")
                )
                ema.restore(model)

                print(f"  ★ New best: solved={best_eval_solved:.4f}  "
                      f"px={best_eval_px:.4f}")
            else:
                patience_counter += 1

            try:
                plot_training_curves(tracker, epoch)
            except Exception:
                pass

        if epoch % int(cfg.get("viz_every", 999999)) == 0:
            ema.apply(model)
            try:
                viz_batch_raw = next(iter(eval_loader))
                model.eval()
                with torch.no_grad():
                    viz_pred, viz_conf, viz_steps = model.predict(
                        viz_batch_raw["sx"].to(DEVICE),
                        viz_batch_raw["sy"].to(DEVICE),
                        viz_batch_raw["qx"].to(DEVICE),
                        s_mask=(viz_batch_raw["s_mask"].to(DEVICE).bool()
                                if "s_mask" in viz_batch_raw else None),
                        task_id=(viz_batch_raw["task_id"].to(DEVICE)
                                 if "task_id" in viz_batch_raw else None)
                    )
                viz_pred_cpu = viz_pred.detach().cpu()
                viz_batch = {
                    k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v
                    for k, v in viz_batch_raw.items()
                }
                n_viz = min(int(cfg.get("n_viz", 4)), viz_batch["sx"].shape[0])
                for viz_idx in range(n_viz):
                    visualize_predictions(
                        viz_batch, viz_pred_cpu,
                        idx=viz_idx, conf=float(viz_conf), steps=int(viz_steps),
                        eos_id=int(CFG["eos_id"]), pad_id=int(CFG["pad_id"]),
                        title_prefix=f"EVAL ep{epoch} #{viz_idx}"
                    )
            except Exception:
                import traceback
                traceback.print_exc()
            ema.restore(model)

        elapsed = time.time() - start_time
        print(
            f"  ep {epoch:3d} | "
            f"ce={train_avg.get('ce_loss', 0):.4f}  "
            f"ds={train_avg.get('ds_loss', 0):.4f}  "
            f"sv={train_avg.get('sv_loss', 0):.4f}  "
            f"px={train_avg.get('px_acc', 0):.3f}  "
            f"ch={train_avg.get('change_acc', 0):.3f}  "
            f"uc={train_avg.get('unchanged_acc', 0):.3f}  "
            f"chV={train_avg.get('change_acc_v', 0):.3f}  "
            f"sol={train_avg.get('solved_strict', 0):.4f}  "
            f"m_u={train_avg.get('m_prob_unch', 0):.3f}  "
            f"m_c={train_avg.get('m_prob_ch', 0):.3f} | "
            f"pat={patience_counter}/{cfg['patience']} | "
            f"{elapsed / 60:.1f}min"
        )

        if patience_counter >= int(cfg["patience"]):
            print(f"\n  ★ Early stopping ep {epoch}")
            break

    total_time = time.time() - start_time
    print(f"\nDone in {total_time / 60:.1f} min")
    print(f"Best solved={best_eval_solved:.4f}  px={best_eval_px:.4f}")
    return tracker
