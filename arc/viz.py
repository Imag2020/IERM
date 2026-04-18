import torch
import matplotlib
matplotlib.use("Agg")

# ════════════════════════════════════════════════════════════════════
#  PLOT — inchangé vs v4.1 (p2m_gate ajouté)
# ════════════════════════════════════════════════════════════════════

def plot_training_curves(tracker, epoch):
    import matplotlib.pyplot as plt
    import numpy as np

    def get(k): return tracker.get(k, []) or []

    epochs_train = np.arange(len(get("train_ce_loss")))
    ep_eval      = (np.array(get("eval_epoch"), dtype=int)
                    if len(get("eval_epoch")) else np.arange(len(get("eval_loss"))))

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    ax = axes[0, 0]
    if len(get("train_ce_loss")):   ax.plot(epochs_train, get("train_ce_loss"), label="train_ce", alpha=0.7)
    if len(get("eval_loss")):       ax.plot(ep_eval, get("eval_loss"), label="eval_ce(EMA)")
    if len(get("train_sv_loss")):   ax.plot(epochs_train, get("train_sv_loss"), label="sv_loss", alpha=0.5, ls="--")
    if len(get("train_loso_loss")): ax.plot(epochs_train, get("train_loso_loss"), label="loso_loss", alpha=0.5, ls=":")
    ax.set_title("Losses"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    if len(get("train_px_acc")): ax.plot(epochs_train, get("train_px_acc"), label="train_px", alpha=0.7)
    if len(get("eval_px")):      ax.plot(ep_eval, get("eval_px"), label="eval_px(EMA)")
    ax.set_title("Token accuracy"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    if len(get("eval_change_acc")):  ax.plot(ep_eval, get("eval_change_acc"), "r--", label="eval_ch")
    if len(get("train_change_acc")): ax.plot(epochs_train, get("train_change_acc"), "r-", alpha=0.4, label="train_ch")
    ax.set_title("Change accuracy (pred_mix)"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if len(get("eval_grid_solved")):       ax.plot(ep_eval, get("eval_grid_solved"), label="eval_solved(EMA)")
    if len(get("train_eval_grid_solved")): ax.plot(ep_eval, get("train_eval_grid_solved"), label="train_eval_solved")
    ax.set_title("Grid solved"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[2, 0]
    if len(get("train_m_prob_unch")):     ax.plot(epochs_train, get("train_m_prob_unch"), label="m_unch(train)", color="teal", alpha=0.7)
    if len(get("train_m_prob_ch")):       ax.plot(epochs_train, get("train_m_prob_ch"), label="m_ch(train)", color="orange", alpha=0.7)
    if len(get("eval_m_prob_changed")):   ax.plot(ep_eval, get("eval_m_prob_changed"), "s-", color="purple", label="m_ch(eval)")
    if len(get("eval_m_prob_unchanged")): ax.plot(ep_eval, get("eval_m_prob_unchanged"), "^-", color="green", label="m_unch(eval)")
    ax.axhline(0.05, ls="--", color="green",  alpha=0.4, label="target unch≈0.05")
    ax.axhline(0.95, ls="--", color="purple", alpha=0.4, label="target ch≈0.95")
    ax.set_title("Mask probabilities"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    if len(get("eval_psi_ctx_lift")):   ax.plot(ep_eval, get("eval_psi_ctx_lift"), "s-", color="green", label="Ψ_lift")
    if len(get("eval_recursion_gain")): ax.plot(ep_eval, get("eval_recursion_gain"), "^-", color="purple", label="Φ_gain")
    ax.axhline(0.10, color="green", ls="--", alpha=0.4)
    ax.axhline(0.0,  color="red",   ls="--", alpha=0.3)
    ax.set_title("Ψ_lift (interactive P↔M) + Φ_gain (recursion)"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.suptitle(f"IERM-TRM v4.2 — epoch {epoch}", fontsize=14)
    plt.tight_layout()
    #plt.show()
    fig.savefig("training_curves.jpg", dpi=150, bbox_inches="tight")
    plt.close(fig)

# ════════════════════════════════════════════════════════════════════
#  VISUALIZE (inchangé)
# ════════════════════════════════════════════════════════════════════

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def infer_hw_from_first_eos(g: torch.Tensor, *, eos_id: int = 10) -> tuple:
    g      = g.long()
    col0   = g[:, 0]
    eos_rows = (col0 == eos_id).nonzero(as_tuple=True)[0]
    h      = int(eos_rows[0]) if len(eos_rows) > 0 else 30
    if h == 0:
        return 0, 0
    row0   = g[0, :]
    eos_cols = (row0 == eos_id).nonzero(as_tuple=True)[0]
    w      = int(eos_cols[0]) if len(eos_cols) > 0 else 30
    return h, w


def visualize_predictions(
    batch: dict, pred: torch.Tensor, *,
    idx: int = 0, conf: float = 0.0, steps: int = 0,
    eos_id: int = 10, pad_id: int = 11,
    title_prefix: str = "", figsize_per_cell: float = 2.5,
):
    ARC_COLORS_12 = [
        "#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00",
        "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25",
        "#FF00FF", "#DDDDDD",
    ]
    ARC_CMAP = mpl.colors.ListedColormap(ARC_COLORS_12, name="arc12")
    ARC_NORM = mpl.colors.BoundaryNorm(boundaries=[-0.5 + i for i in range(13)], ncolors=12)
    IMSHOW_KW = dict(cmap=ARC_CMAP, norm=ARC_NORM, interpolation="nearest")

    sx     = batch["sx"][idx].cpu().clone()
    sy     = batch["sy"][idx].cpu().clone()
    qx     = batch["qx"][idx].cpu().clone()
    qy     = batch["qy"][idx].cpu().clone()
    pred_b = pred[idx].cpu().clone()

    S       = sx.shape[0]
    H, W    = qx.shape
    n_cols  = max(S, 5)
    fig, axes = plt.subplots(3, n_cols,
                             figsize=(figsize_per_cell * n_cols, figsize_per_cell * 3))

    def _show(ax, g, label):
        ax.imshow(g.numpy(), **IMSHOW_KW); ax.set_title(label, fontsize=8); ax.axis("off")
    def _blank(ax): ax.axis("off")

    for s in range(S):
        _show(axes[0, s], sx[s], f"sx[{s}]")
    for c in range(S, n_cols): _blank(axes[0, c])
    for s in range(S):
        _show(axes[1, s], sy[s], f"sy[{s}]")
    for c in range(S, n_cols): _blank(axes[1, c])

    start = max(0, (n_cols - 5) // 2)
    for c in range(n_cols): _blank(axes[2, c])
    _show(axes[2, start],     qx,     "qx (input)")
    _show(axes[2, start + 1], qy,     "qy (GT)")
    _show(axes[2, start + 2], pred_b, "PRED")

    h_gt, w_gt = infer_hw_from_first_eos(qy, eos_id=eos_id)
    diff       = np.ones((H, W, 3), dtype=np.float32) * 0.85
    pr_np      = pred_b.numpy()
    gt_np      = qy.numpy()
    qx_np      = qx.numpy()
    h_end      = min(h_gt + 1, H)
    w_end      = min(w_gt + 1, W)
    for r in range(h_end):
        for c in range(w_end):
            p, g, x = int(pr_np[r, c]), int(gt_np[r, c]), int(qx_np[r, c])
            if g == pad_id:
                diff[r, c] = [0.85, 0.85, 0.85]
            elif g == eos_id:
                diff[r, c] = [0.6, 1.0, 0.6] if p == eos_id else [1.0, 0.3, 0.3]
            elif p == g and g != x:
                diff[r, c] = [0.0, 1.0, 0.7]
            elif p == g:
                diff[r, c] = [0.7, 1.0, 0.7]
            else:
                diff[r, c] = [1.0, 0.0, 0.0]

    axes[2, start + 3].imshow(diff)
    axes[2, start + 3].set_title("Diff", fontsize=8)
    axes[2, start + 3].axis("off")

    gt_crop  = qy[:h_gt, :w_gt]
    pr_crop  = pred_b[:h_gt, :w_gt]
    content  = (gt_crop != pad_id) & (gt_crop != eos_id)
    px_acc   = float((pr_crop[content] == gt_crop[content]).float().mean()) if content.any() else 0.0
    task_id  = int(batch["task_id"][idx]) if "task_id" in batch else -1
    axes[2, start + 4].text(
        0.5, 0.5,
        f"task={task_id}\ncrop={h_gt}×{w_gt}\npx_acc={px_acc:.3f}",
        ha="center", va="center", fontsize=9, family="monospace",
        transform=axes[2, start + 4].transAxes)
    axes[2, start + 4].set_title("Stats", fontsize=8)
    axes[2, start + 4].axis("off")

    fig.suptitle(f"{title_prefix}  conf={conf:.3f}  T={steps}", fontsize=10)
    plt.tight_layout()
    #plt.show()
    fig.savefig("last_prediction.jpg", dpi=150, bbox_inches="tight")
    plt.close(fig)
