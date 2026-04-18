"""
dump_keys.py
Robust checkpoint inspection. Handles several common saving conventions:
  - plain state_dict (no wrapping)
  - wrapped with "model_state_dict" (default in our save_checkpoint)
  - wrapped with "model" / "state_dict" / "ema"
  - nested wrappers (e.g. {"ema": {"model_state_dict": ...}})
  - EMA wrapper objects that need .state_dict() or .shadow
"""

import sys
from pathlib import Path

import torch


# ════════════════════════════════════════════════════════════════
# CONFIG — adapt paths to your repo
# ════════════════════════════════════════════════════════════════

CKPTS = [
    ("MLP (63%)",        "model_checkpoints/sudoku_mlp_62pct.pt"),
    ("ATTENTION (58%)",  "model_checkpoints/best_55.pt"),
]


# ════════════════════════════════════════════════════════════════
# Helpers to find the state_dict in whatever structure we get
# ════════════════════════════════════════════════════════════════

def looks_like_state_dict(d) -> bool:
    """A state_dict maps str -> Tensor. Check the first few items."""
    if not isinstance(d, dict) or len(d) == 0:
        return False
    for i, (k, v) in enumerate(d.items()):
        if i >= 5:
            break
        if not isinstance(k, str):
            return False
        if not isinstance(v, torch.Tensor):
            return False
    return True


def find_state_dict(obj, path=""):
    """
    Recursively find a state_dict inside a checkpoint.
    Returns (state_dict, path_taken) or (None, "").
    """
    # Direct hit.
    if looks_like_state_dict(obj):
        return obj, path or "<root>"

    # Common wrappers to try, in order of likelihood.
    if isinstance(obj, dict):
        for key in ("model_state_dict", "model", "state_dict", "ema",
                    "net", "module", "params"):
            if key in obj:
                sub = obj[key]
                found, sub_path = find_state_dict(
                    sub, f"{path}.{key}" if path else key)
                if found is not None:
                    return found, sub_path

    # EMA-like objects: try common attribute names.
    for attr in ("state_dict", "shadow_params", "shadow"):
        if hasattr(obj, attr):
            got = getattr(obj, attr)
            # If it's a method, call it.
            if callable(got):
                try:
                    got = got()
                except Exception:
                    continue
            found, sub_path = find_state_dict(
                got, f"{path}.{attr}" if path else attr)
            if found is not None:
                return found, sub_path

    return None, ""


def show_top_level(ckpt, max_items=20):
    """Describe the top level of the checkpoint for diagnosis."""
    if isinstance(ckpt, dict):
        print(f"  Top-level type: dict with {len(ckpt)} keys")
        for i, (k, v) in enumerate(ckpt.items()):
            if i >= max_items:
                print(f"  ... ({len(ckpt) - max_items} more keys)")
                break
            vtype = type(v).__name__
            extra = ""
            if isinstance(v, dict):
                extra = f" ({len(v)} sub-keys)"
            elif isinstance(v, torch.Tensor):
                extra = f" shape={tuple(v.shape)}"
            elif hasattr(v, "__len__"):
                try:
                    extra = f" len={len(v)}"
                except Exception:
                    pass
            print(f"    {k!r:30s}  {vtype}{extra}")
    else:
        print(f"  Top-level type: {type(ckpt).__name__} (not a dict!)")


# ════════════════════════════════════════════════════════════════
# Main analysis
# ════════════════════════════════════════════════════════════════

def analyze(label: str, path: str):
    print("\n" + "=" * 72)
    print(f"  {label}   ({path})")
    print("=" * 72)

    if not Path(path).exists():
        print("  ✗ FILE NOT FOUND")
        return None

    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"  ✗ torch.load FAILED: {type(e).__name__}: {e}")
        return None

    print(f"  File size : {Path(path).stat().st_size / 1e6:.1f} MB")
    show_top_level(ckpt)

    sd, sd_path = find_state_dict(ckpt)
    if sd is None:
        print("\n  ✗ Could not find a state_dict in this checkpoint.")
        print("    → Paste the full top-level listing above for diagnosis.")
        return None

    print(f"\n  state_dict found at : {sd_path}")
    print(f"  state_dict size     : {len(sd)} parameters")

    # Extract metadata from the top level.
    if isinstance(ckpt, dict):
        for meta_key in ("step", "epoch", "global_step", "metrics",
                          "best_metric", "loss"):
            if meta_key in ckpt:
                val = ckpt[meta_key]
                s = repr(val)
                if len(s) > 200:
                    s = s[:200] + "..."
                print(f"  meta [{meta_key}]    : {s}")

    # Dump phi keys.
    phi_keys = [k for k in sd.keys()
                if k.startswith("phi.") and "_original" not in k]
    legacy = [k for k in sd.keys() if "_original" in k]

    print(f"\n  PHI KEYS ({len(phi_keys)}):")
    for k in sorted(phi_keys):
        print(f"    {k:58s} {tuple(sd[k].shape)}")

    if legacy:
        print(f"\n  LEGACY KEYS ({len(legacy)}, will be stripped on load):")
        for k in sorted(legacy):
            print(f"    {k:58s} {tuple(sd[k].shape)}")

    # Architectural flags.
    has_lm_proj     = any("lm_proj"     in k for k in sd)
    has_lm_norm     = any("lm_norm"     in k for k in sd)
    has_mask_head   = any("mask_head"   in k for k in sd)
    has_val_head    = any("val_head"    in k for k in sd)
    has_q_head      = any("q_head"      in k for k in sd)
    has_norm_out    = any("norm_out"    in k for k in sd)
    y_sa_is_mlp     = any("y_sa.fc1"    in k for k in sd)
    y_sa_is_attn    = any("y_sa.qkv"    in k for k in sd)
    has_block_emb   = any("emb_block"   in k for k in sd)
    has_band_emb    = any("emb_band"    in k for k in sd)
    has_stack_emb   = any("emb_stack"   in k for k in sd)

    print(f"\n  ARCHITECTURE SUMMARY:")
    if y_sa_is_mlp:
        print(f"    phi.y_sa        : TokenMixerMLP  (fc1 / fc2 keys present)")
    elif y_sa_is_attn:
        print(f"    phi.y_sa        : MultiHeadSelfAttention  (qkv / out keys)")
    else:
        print(f"    phi.y_sa        : ??? (neither fc1 nor qkv found)")

    head = []
    if has_lm_proj:   head.append("lm_proj")
    if has_lm_norm:   head.append("lm_norm")
    if has_mask_head: head.append("mask_head")
    if has_val_head:  head.append("val_head")
    if has_norm_out:  head.append("norm_out")
    print(f"    output head     : {', '.join(head) if head else '??'}")
    print(f"    has q_head      : {has_q_head}")
    print(f"    sudoku emb      : block={has_block_emb}  "
          f"band={has_band_emb}  stack={has_stack_emb}")

    return sd


if __name__ == "__main__":
    sds = {}
    for label, path in CKPTS:
        sds[label] = analyze(label, path)

    # Cross-compare if both loaded.
    if all(v is not None for v in sds.values()) and len(sds) == 2:
        labels = list(sds.keys())
        a_label, b_label = labels
        a, b = sds[a_label], sds[b_label]
        only_a = set(a.keys()) - set(b.keys())
        only_b = set(b.keys()) - set(a.keys())
        common = set(a.keys()) & set(b.keys())

        print("\n" + "=" * 72)
        print(f"  DIFF  {a_label}  vs  {b_label}")
        print("=" * 72)
        print(f"  Common keys            : {len(common)}")
        print(f"  Keys only in {a_label:15s}: {len(only_a)}")
        for k in sorted(only_a)[:30]:
            print(f"    + {k}")
        if len(only_a) > 30:
            print(f"    ... ({len(only_a) - 30} more)")
        print(f"  Keys only in {b_label:15s}: {len(only_b)}")
        for k in sorted(only_b)[:30]:
            print(f"    + {k}")
        if len(only_b) > 30:
            print(f"    ... ({len(only_b) - 30} more)")