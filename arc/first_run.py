print("importing..")
from pathlib import Path
import json
import torch
from torch.utils.data import DataLoader

print("import dataset...")
from dataset import *
print("dataset OK")

print("import model...")
from model import *
print("model OK")

print("import viz...")
from viz import *
print("viz OK")

print("import train...")
import train as train_mod
print("train OK")


# ============================================================
# MODEL CFG
# ============================================================

CFG = dict(
    vocab_size=12,
    pad_id=11,
    eos_id=10,

    d_model=256,
    d_psi=96,
    n_heads=8,
    dropout=0.04,

    T_max=2,
    n_inner=2,          # <-- plus stable que 4 from scratch

    n_mem=96,
    n_prog=32,
    psi_refine_layers=2,
    z_tokens=12,

    Hmax=30,
    Wmax=30,

    use_rope_2d=True,
    use_qk_norm=True,
    use_task_id=True,
    use_lm_head=True,
    n_tasks=2048,
    T_psi=2,
)


# ============================================================
# TRAIN CFG
# ============================================================

TRAIN_CFG = dict(
    epochs=600,
    lr=2e-4,
    weight_decay=0.05,
    warmup_steps=800,
    cycle_epochs=1,
    grad_clip=0.5,

    T_per_step=1,
    n_grad_steps=2,
    residual_alpha=0.4,

    # plus neutre / stable pour repartir proprement
    change_weight=4.0,
    unchanged_weight=1.0,
    eos_weight=1.2,

    loso_lambda=0.20,
    loso_max_holds=3,
    support_verify_lambda=0.05,

    # inutiles en lm_head pur, laissés à 0 pour éviter toute ambiguïté
    mask_loss_weight=0.0,
    val_loss_weight=0.0,
    mask_temp=3.0,

    label_smoothing=0.03,

    augment=True,
    aug_prob=0.70,
    use_task_color_perm=True,

    aug_supports_prob=0.50,
    aug_n_views=2,

    color_dropout_prob=0.0,
    equivariance_lambda=0.0,
    hard_mining_factor=1.0,

    ema_decay=0.999,
    eval_every=20,              # moins d’overhead
    viz_every=200,              # quasi désactivé
    n_viz=4,
    log_every=20,
    train_eval_max_batches=5,   # moins cher
    patience=120,
    use_amp=True,
    amp_dtype="bf16",

    # on coupe la DS pour l’instant
    use_deep_supervision=False,
    ds_weight=0.0,
    ds_decay=0.70,
    q_ds_weight=0.0,

    resume_from=None,
)


def apply_curriculum_v4(epoch, cfg, model=None):
    """
    Curriculum prudent pour stabiliser d'abord:
      - copie de base
      - Φ non destructif
      - Ψ utile

    Puis seulement ensuite:
      - plus d'augmentation
      - T=2
      - hard mining léger
    """

    # ── Phase 1 : stabilisation minimale ─────────────────────
    if epoch < 40:
        cfg["T_per_step"] = 1
        cfg["aug_prob"] = 0.60
        cfg["color_dropout_prob"] = 0.0
        cfg["equivariance_lambda"] = 0.0
        cfg["hard_mining_factor"] = 1.0
        cfg["residual_alpha"] = 0.35
        cfg["support_verify_lambda"] = 0.03
        cfg["loso_lambda"] = 0.15
        if model is not None:
            model.psi.T_psi = 1

    # ── Phase 2 : on remet un peu plus de contrainte ─────────
    elif epoch < 120:
        cfg["T_per_step"] = 1
        cfg["aug_prob"] = 0.70
        cfg["color_dropout_prob"] = 0.0
        cfg["equivariance_lambda"] = 0.0
        cfg["hard_mining_factor"] = 1.0
        cfg["residual_alpha"] = 0.40
        cfg["support_verify_lambda"] = 0.05
        cfg["loso_lambda"] = 0.20
        if model is not None:
            model.psi.T_psi = 2

    # ── Phase 3 : augmentation un peu plus forte ─────────────
    elif epoch < 220:
        cfg["T_per_step"] = 1
        cfg["aug_prob"] = 0.85
        cfg["color_dropout_prob"] = 0.05
        cfg["equivariance_lambda"] = 0.0
        cfg["hard_mining_factor"] = 1.1
        cfg["residual_alpha"] = 0.40
        cfg["support_verify_lambda"] = 0.05
        cfg["loso_lambda"] = 0.20
        if model is not None:
            model.psi.T_psi = 2

    # ── Phase 4 : on introduit T=2 seulement après stabilité ─
    elif epoch < 360:
        cfg["T_per_step"] = 2
        cfg["aug_prob"] = 0.90
        cfg["color_dropout_prob"] = 0.05
        cfg["equivariance_lambda"] = 0.0
        cfg["hard_mining_factor"] = 1.2
        cfg["residual_alpha"] = 0.40
        cfg["support_verify_lambda"] = 0.05
        cfg["loso_lambda"] = 0.20
        if model is not None:
            model.psi.T_psi = 2

    # ── Phase 5 : régime principal ───────────────────────────
    else:
        cfg["T_per_step"] = 2
        cfg["aug_prob"] = 1.0
        cfg["color_dropout_prob"] = 0.10
        cfg["equivariance_lambda"] = 0.0
        cfg["hard_mining_factor"] = 1.3
        cfg["residual_alpha"] = 0.40
        cfg["support_verify_lambda"] = 0.05
        cfg["loso_lambda"] = 0.20
        if model is not None:
            model.psi.T_psi = 2

    return cfg


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
        loso_max_holds=TRAIN_CFG.get("loso_max_holds", 4),
        n_mem=CFG.get("n_mem", 64),
        n_prog=CFG.get("n_prog", 16),
        psi_refine_layers=CFG.get("psi_refine_layers", 1),
        z_tokens=CFG.get("z_tokens", 8),
        T_psi=CFG.get("T_psi", 2),
        aug_supports_prob=TRAIN_CFG.get("aug_supports_prob", 0.5),
        aug_n_views=TRAIN_CFG.get("aug_n_views", 2),
    )
    return model


def main():
    print("running ..", flush=True)

    output_dir = Path("~/work/ierm/runs/arc_lmhead_stable_v1").expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("datasets..", flush=True)
    ds_train, ds_trainext, ds_eval, n_tasks, _, _, _ = build_all(
        batch_size=8,
        p_task_aug_train=0.5,
    )

    CFG["n_tasks"] = int(n_tasks)

    collate_train = lambda b: collate_arc_fixedS(b, p_task_aug=0.5)
    collate_eval  = lambda b: collate_arc_fixedS(b, p_task_aug=0.0)

    print("loaders ..", flush=True)

    num_workers = 6
    persistent = num_workers > 0

    train_loader = DataLoader(
        ds_train,
        batch_size=8,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_train,
        pin_memory=True,
        persistent_workers=persistent,
        prefetch_factor=4,
    )

    trainext_loader = DataLoader(
        ds_trainext,
        batch_size=8,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_train,
        pin_memory=True,
        persistent_workers=persistent,
        prefetch_factor=4,
    )

    eval_loader = DataLoader(
        ds_eval,
        batch_size=8,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_eval,
        pin_memory=True,
        persistent_workers=persistent,
        prefetch_factor=2,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device, flush=True)

    # Injection des globals attendus par train.py
    train_mod.CFG = CFG
    train_mod.TRAIN_CFG = TRAIN_CFG
    train_mod.DEVICE = device
    train_mod.train_loader = train_loader
    train_mod.eval_loader = eval_loader
    train_mod.trainext_loader = trainext_loader
    train_mod.build_model = build_model
    train_mod.OUTPUT_DIR = str(output_dir)
    train_mod.apply_curriculum = apply_curriculum_v4

    with open(output_dir / "model_cfg.json", "w") as f:
        json.dump(CFG, f, indent=2)

    with open(output_dir / "train_cfg.json", "w") as f:
        json.dump(TRAIN_CFG, f, indent=2)

    print(f"[RUN] device={device}", flush=True)
    print(f"[RUN] output_dir={output_dir}", flush=True)
    print(f"[RUN] n_tasks={n_tasks}", flush=True)
    print(f"[RUN] train={len(ds_train)} eval={len(ds_eval)} trainext={len(ds_trainext)}", flush=True)
    print(
        f"[RUN] batch_size=8 num_workers={num_workers} "
        f"use_deep_supervision={TRAIN_CFG['use_deep_supervision']}",
        flush=True
    )
    print(
        f"[RUN] n_inner={CFG['n_inner']} T_max={CFG['T_max']} "
        f"change_weight={TRAIN_CFG['change_weight']} "
        f"unchanged_weight={TRAIN_CFG['unchanged_weight']} "
        f"support_verify_lambda={TRAIN_CFG['support_verify_lambda']}",
        flush=True
    )

    tracker = train_mod.run_training_seq()

    print("[RUN] finished", flush=True)
    return tracker


if __name__ == "__main__":
    main()
