"""
GAP-1+2 (RESEARCH_GAP.md): Transfer / Sharing analysis.

    GAP-1: "no unified quantum framework addresses discriminative +
            generative multimodal tasks end-to-end"
            → Joint training curve: shared PQC learns signal cho CẢ 2 tasks.

    GAP-2: "the SAME shared quantum fusion layer be used successfully
            across tasks from very different families?"
            - "thaméliage" transfer: ΔMSA = joint − separate
            - "bảng tham số shared vs separate": quantum param count
            - "phân tích gradient conflict giữa 2 tasks trên shared PQC
              weights": cos(g_msa, g_cap)

Cách dùng:
    python experiments/run_transfer_analysis.py \
        --config experiments/configs/qfl_tensor.yaml \
        --epochs 5 --device auto
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from src.training.train import (
    Config,
    load_config,
    set_seed,
    get_device,
    build_model,
    build_dataloaders,
    train_sentiment_only,
    train_captioning_only,
    train_multitask,
    evaluate_msa,
    evaluate_caption_bleu,
    save_checkpoint,
    save_results,
)
from src.evaluation.metrics import count_parameters
from src.training.gradient_conflict import (
    shared_param_names,
    gradient_cosine_similarity,
    analyze_conflict_over_batches,
)
from src.quantum.noise_wrapper import ResidualQuantumMitigation

RESULTS_DIR = Path(__file__).parent / "results"


def _deep_copy_cfg(base_cfg):
    raw = {
        "model": vars(base_cfg.model),
        "training": vars(base_cfg.training),
        "data": vars(base_cfg.data),
    }
    for extra in ["noise", "checkpoint_dir", "log_dir", "model_name"]:
        if hasattr(base_cfg, extra):
            val = getattr(base_cfg, extra)
            raw[extra] = vars(val) if hasattr(val, "__dict__") else val
    return Config(raw)


# ------------------------------------------------------------------
# A. Joint vs Separate training
# ------------------------------------------------------------------

def train_joint_vs_separate(base_cfg, device):
    """ΔMSA, ΔBLEU: joint multitask vs separate."""
    print("\n=== [A] Joint training ===")
    cfg = _deep_copy_cfg(base_cfg)
    cfg.checkpoint_dir = "checkpoints/_transfer"
    set_seed(42)
    joint = train_multitask(cfg, device)
    joint_test_acc = joint["test"].get("accuracy", 0)

    # Evaluate captioning BLEU on joint model (reload best checkpoint)
    print("  → Rebuilding joint model for BLEU eval ...")
    j_model = build_model(cfg, task="sentiment", model_name="qmmf")
    j_ckpt = Path(cfg.checkpoint_dir) / "best_multitask_latest.pt"
    if j_ckpt.exists():
        ckpt = torch.load(j_ckpt, map_location=device, weights_only=True)
        j_model.load_state_dict(ckpt["model_state_dict"])
    j_model.to(device)
    cap_loaders = build_dataloaders(cfg, "captioning")
    joint_bleu = evaluate_caption_bleu(j_model, cap_loaders["test"], device)["bleu_4"]
    del j_model

    print("\n=== [A] Separate MSA ===")
    cfg_s1 = _deep_copy_cfg(base_cfg)
    cfg_s1.checkpoint_dir = "checkpoints/_transfer_sep_senti"
    set_seed(42)
    sep_senti = train_sentiment_only(cfg_s1, device)
    sep_senti_acc = sep_senti["test"].get("accuracy", 0)

    print("\n=== [A] Separate Caption ===")
    cfg_s2 = _deep_copy_cfg(base_cfg)
    cfg_s2.checkpoint_dir = "checkpoints/_transfer_sep_cap"
    set_seed(42)
    sep_cap = train_captioning_only(cfg_s2, device)
    sep_cap_bleu = sep_cap["test"].get("bleu_4", 0)

    return {
        "joint_test_msa": joint_test_acc,
        "separate_test_msa": sep_senti_acc,
        "delta_msa": joint_test_acc - sep_senti_acc,
        "joint_test_bleu4": joint_bleu,
        "separate_test_bleu4": sep_cap_bleu,
        "delta_bleu4": joint_bleu - sep_cap_bleu,
    }


# ------------------------------------------------------------------
# B. Parameter accounting: shared vs separate quantum spaces
# ------------------------------------------------------------------

def compare_shared_vs_separate(base_cfg, device="cpu"):
    """Shared 1× PQC fusion params vs 2× separate PQC fusion params."""
    cfg = _deep_copy_cfg(base_cfg)
    model = build_model(cfg, task="sentiment", model_name="qmmf")

    shared_total = count_parameters(model)
    shared_fusion = sum(p.numel() for n, p in model.named_parameters()
                        if "q_fusion" in n)
    shared_proj = sum(p.numel() for n, p in model.named_parameters()
                      if "q_proj" in n or "caption_adapter" in n)

    separate_fusion_total = shared_fusion * 2
    separate_proj_total = shared_proj * 2
    separate_total_quantum = separate_fusion_total + separate_proj_total

    return {
        "shared": {
            "total": shared_total,
            "fusion_params": shared_fusion,
            "proj_params": shared_proj,
        },
        "separate_approx": {
            "fusion_params": separate_fusion_total,
            "proj_params": separate_proj_total,
            "quantum_params": separate_total_quantum,
        },
        "savings": shared_fusion,
        "savings_pct": shared_fusion / separate_fusion_total if separate_fusion_total else 0,
    }


# ------------------------------------------------------------------
# C. Gradient conflict on shared PQC params
# ------------------------------------------------------------------

def gradient_conflict_analysis(base_cfg, device, n_batches=20):
    """
    Phân tích cos(g_msa, g_cap) trên shared PQC params.
    Trả về mean_cos, min_cos, frac_conflict_batches.
    """
    cfg = _deep_copy_cfg(base_cfg)
    model = build_model(cfg, task="sentiment", model_name="qmmf").to(device)
    loaders = build_dataloaders(cfg, "multitask")

    criterion = nn.CrossEntropyLoss()

    def _make_losses_fn(batch_idx):
        def fn():
            # Pull one batch from each loader (reuse same batch)
            for i, b in enumerate(loaders["train"]):
                if i > batch_idx:
                    break
                msa_batch = {k: v.to(device) for k, v in b.items()}
                break
            for i, b in enumerate(loaders["cap_train"]):
                if i > batch_idx:
                    break
                cap_batch = {k: v.to(device) for k, v in b.items()}
                break
            msa_logits = model(task="msa", input_ids=msa_batch["input_ids"],
                               attention_mask=msa_batch["attention_mask"],
                               images=msa_batch["image"])
            l_msa = criterion(msa_logits, msa_batch["label"])
            cap_logits = model(task="caption", images=cap_batch["image"],
                               decoder_input_ids=cap_batch["decoder_input_ids"])
            l_cap = criterion(cap_logits.reshape(-1, cap_logits.size(-1)),
                              cap_batch["labels"].reshape(-1))
            return l_msa, l_cap
        return fn

    from src.training.gradient_conflict import cosine_similarity as cs_fn

    cos_vals = []
    for b_idx in range(min(n_batches, len(loaders["train"]) - 1)):
        msa_loss, cap_loss = _make_losses_fn(b_idx)()
        model.zero_grad(set_to_none=True)
        msa_loss.backward(retain_graph=True)
        g_msa_raw = {n: p.grad.clone() for n, p in model.named_parameters()
                     if "q_fusion" in n and p.grad is not None}
        model.zero_grad(set_to_none=True)
        cap_loss.backward()
        g_cap_raw = {n: p.grad.clone() for n, p in model.named_parameters()
                     if "q_fusion" in n and p.grad is not None}
        model.zero_grad(set_to_none=True)

        # flatten matching keys
        common = [n for n in g_msa_raw if n in g_cap_raw]
        if not common:
            continue
        v1 = torch.cat([g_msa_raw[n].reshape(-1) for n in common])
        v2 = torch.cat([g_cap_raw[n].reshape(-1) for n in common])
        cos_vals.append(float(cs_fn(v1, v2)))

    if not cos_vals:
        return {"error": "no valid gradient pairs computed"}

    import numpy as np
    a = np.array(cos_vals)
    return {
        "mean_cos": float(a.mean()),
        "min_cos": float(a.min()),
        "max_cos": float(a.max()),
        "frac_conflict": float((a < 0).mean()),
        "per_batch_cos": cos_vals,
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GAP-1+2 Transfer Analysis")
    parser.add_argument("--config", default="experiments/configs/qfl_tensor.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip", nargs="+", default=[],
                        choices=["joint", "params", "gradient"])
    parser.add_argument("--output", default=str(RESULTS_DIR / "transfer_analysis.json"))
    args = parser.parse_args()

    device = get_device(args.device)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    base_cfg = load_config(args.config)
    if args.epochs:
        base_cfg.training.epochs = args.epochs

    results = {}
    if "joint" not in args.skip:
        results["joint_vs_separate"] = train_joint_vs_separate(base_cfg, device)
    if "params" not in args.skip:
        results["param_accounting"] = compare_shared_vs_separate(base_cfg, device)
    if "gradient" not in args.skip:
        results["gradient_conflict"] = gradient_conflict_analysis(base_cfg, device)

    save_results(results, args.output)
    print("\nTransfer analysis complete.")
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
