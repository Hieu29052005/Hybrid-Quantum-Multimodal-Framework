"""
GAP-5 + GAP-6 (RESEARCH_GAP.md): Entanglement / interpretability diagnostics
+ ansatz profiling (E11).

    GAP-5: "entanglement or circuit metrics ... as interpretable cross-modal
            diagnostics"
            - Von Neumann entropy of reduced density matrix ρ_text, ρ_image
            - Mutual information I(text : image) across training & noise levels
            - Correlate MI with prediction correctness
            - MI collapse under noise

    GAP-6 / §2.7: "the expressibility and entangling capability of the
            employed parameterized quantum circuit (ansatz)"
            - Expressibility KL divergence vs Haar (Sim et al. 2019)
            - Meyer–Wallach global entanglement

Cách dùng:
    python experiments/run_entanglement_diagnostics.py \
        --config experiments/configs/qfl_tensor.yaml --epochs 5 --device cpu
"""

import argparse
import json
import math
import sys
import time
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn

from src.training.train import (
    Config,
    load_config,
    set_seed,
    get_device,
    build_model,
    build_dataloaders,
    train_epoch_sentiment,
    collect_msa_predictions,
    save_results,
    _move_to_device,
)
from src.training.optimizer_utils import setup_optimizer, clip_gradients
from src.training.loss import MultiTaskLoss
from src.evaluation.metrics import sentiment_metrics
from src.quantum.entanglement import CrossModalEntanglementAnalyzer
from src.quantum.ansatz_metrics import profile_ansatz

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "paper" / "figures"


def _deep_copy_cfg(base_cfg):
    raw = {
        "model": vars(base_cfg.model),
        "training": vars(base_cfg.training),
        "data": vars(base_cfg.data),
    }
    for extra in ["noise", "checkpoint_dir", "log_dir"]:
        if hasattr(base_cfg, extra):
            val = getattr(base_cfg, extra)
            raw[extra] = vars(val) if hasattr(val, "__dict__") else val
    return Config(raw)


def _get_fusion_weights(model):
    """Extract PQC fusion weights tensor from model."""
    for name, p in model.named_parameters():
        if "q_fusion.weights" in name:
            return p.detach().cpu()
    return None


# ------------------------------------------------------------------
# A. Ansatz profile (GAP-6)
# ------------------------------------------------------------------

def run_ansatz_profile(base_cfg):
    """Expressibility + entangling capability across depths & qubit counts."""
    mcfg = getattr(base_cfg, "model", base_cfg)
    ansatz_name = getattr(mcfg, "fusion_type", "tensor")
    print(f"\n--- Ansatz profile: {ansatz_name} ---")
    profile = profile_ansatz(
        ansatz_names=[ansatz_name],
        qubits_list=[4, 6, 8, 10],
        depths=[1, 2, 3, 5],
        n_samples=150,
        seed=42,
    )
    for row in profile:
        kl = row.get("expressibility_kl")
        mw = row.get("entangling_capability")
        print(f"  {row['ansatz']:12s} | q={row['n_qubits']} | L={row['depth']} | "
              f"KL={kl if kl is not None else 'N/A':>8} | "
              f"MW={mw if mw is not None else 'N/A':.3f}" if mw is not None else "")
    return profile


# ------------------------------------------------------------------
# B. MI across training epochs
# ------------------------------------------------------------------

def mi_across_training(base_cfg, device, max_batches_analyzed=8):
    """
    Train a short run; after each epoch, compute MI(text:image) via
    CrossModalEntanglementAnalyzer on val batches → track MI vs epoch.
    """
    print("\n--- MI across training epochs ---")
    cfg = _deep_copy_cfg(base_cfg)
    model = build_model(cfg, task="sentiment", model_name="qmmf").to(device)
    loaders = build_dataloaders(cfg, "sentiment")

    criterion = MultiTaskLoss(lambda_sentiment=1.0, lambda_caption=0.0, lambda_reg=0.0)
    n_steps = max(cfg.training.epochs * len(loaders["train"]), 1)
    optimizer, scheduler = setup_optimizer(
        model, lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        warmup_steps=cfg.training.warmup_steps,
        total_steps=n_steps,
    )

    mcfg = getattr(cfg, "model", cfg)
    n_qubits = getattr(mcfg, "n_qubits", 8)
    half = n_qubits // 2
    n_layers = getattr(mcfg, "n_q_layers", 3)
    epoch_mi = []

    for epoch in range(cfg.training.epochs):
        train_loss = train_epoch_sentiment(
            model, loaders["train"], criterion, optimizer, scheduler,
            device, cfg.training.max_grad_norm,
        )

        # Evaluate accuracy
        preds, labels = collect_msa_predictions(model, loaders["val"], device)
        val_acc = sentiment_metrics(preds, labels)["accuracy"]

        # Compute MI on val batch subset
        w = _get_fusion_weights(model)
        if w is None:
            print(f"  Epoch {epoch+1}: cannot extract q_fusion.weights; skipping MI")
            continue
        analyzer = CrossModalEntanglementAnalyzer(n_qubits=n_qubits,
                                                  n_layers=n_layers,
                                                  weights=w)
        mi_vals = []
        with torch.no_grad():
            for b_idx, batch in enumerate(loaders["val"]):
                if b_idx >= max_batches_analyzed:
                    break
                batch = _move_to_device(batch, device)
                t_emb = model.text_encoder(batch["input_ids"],
                                           batch["attention_mask"]).cpu().numpy()
                i_emb = model.image_encoder(batch["image"]).cpu().numpy()
                stats = analyzer.analyze_batch(
                    t_emb[:, :half], i_emb[:, :half])
                mi_vals.append(stats["MI_text_image_mean"])
        avg_mi = float(np.mean(mi_vals)) if mi_vals else None
        epoch_mi.append({
            "epoch": epoch + 1, "val_acc": val_acc,
            "train_loss": train_loss,
            "MI_mean": avg_mi,
        })
        print(f"  Epoch {epoch+1}: loss={train_loss:.4f} acc={val_acc:.4f} "
              f"MI={avg_mi}")

    return epoch_mi


# ------------------------------------------------------------------
# C. MI vs prediction correctness
# ------------------------------------------------------------------

def mi_vs_correctness(base_cfg, device, max_samples=64):
    """MI on test set; group by correct/incorrect predictions."""
    print("\n--- MI vs prediction correctness ---")
    cfg = _deep_copy_cfg(base_cfg)
    model = build_model(cfg, task="sentiment", model_name="qmmf").to(device)
    loaders = build_dataloaders(cfg, "sentiment")

    mcfg = getattr(cfg, "model", cfg)
    n_qubits = getattr(mcfg, "n_qubits", 8)
    half = n_qubits // 2
    n_layers = getattr(mcfg, "n_q_layers", 3)

    w = _get_fusion_weights(model)
    if w is None:
        return {"error": "no q_fusion.weights found"}

    analyzer = CrossModalEntanglementAnalyzer(n_qubits=n_qubits,
                                              n_layers=n_layers, weights=w)

    all_t, all_i, all_correct = [], [], []
    model.eval()
    with torch.no_grad():
        for b_idx, batch in enumerate(loaders["test"]):
            if len(all_t) >= max_samples:
                break
            batch = _move_to_device(batch, device)
            logits = model(task="msa", input_ids=batch["input_ids"],
                           attention_mask=batch["attention_mask"],
                           images=batch["image"])
            correct = (logits.argmax(-1) == batch["label"]).cpu().numpy().tolist()
            t_emb = model.text_encoder(batch["input_ids"],
                                       batch["attention_mask"]).cpu().numpy()
            i_emb = model.image_encoder(batch["image"]).cpu().numpy()
            all_t.append(t_emb[:, :half])
            all_i.append(i_emb[:, :half])
            all_correct.extend(correct)

    if not all_t:
        return {"error": "no samples"}
    all_t = np.concatenate(all_t)[:max_samples]
    all_i = np.concatenate(all_i)[:max_samples]
    all_correct = np.array(all_correct[:max_samples])

    return analyzer.mi_vs_correctness(all_t, all_i, all_correct)


# ------------------------------------------------------------------
# D. MI under noise (GAP-5 collapse)
# ------------------------------------------------------------------

def mi_under_noise(base_cfg, noise_levels=(0.0, 0.005, 0.01, 0.02)):
    """MI collapse curve on a single sample (small n_qubits for speed)."""
    print("\n--- MI under noise ---")
    mcfg = getattr(base_cfg, "model", base_cfg)
    n_qubits = getattr(mcfg, "n_qubits", 8)
    half = n_qubits // 2
    n_layers = getattr(mcfg, "n_q_layers", 3)
    w = torch.randn(n_layers, n_qubits * 2) * 0.01

    text_sample = np.random.uniform(-math.pi / 2, math.pi / 2, half)
    image_sample = np.random.uniform(-math.pi / 2, math.pi / 2, half)

    analyzer = CrossModalEntanglementAnalyzer(n_qubits=n_qubits,
                                              n_layers=n_layers, weights=w)
    results = analyzer.mi_under_noise(text_sample, image_sample, noise_levels)
    for p, s_full, mi in results:
        print(f"  p={p:.3f}  MI={mi:.4f}  S_full={s_full:.4f}")
    return [{"p": float(p), "S_full": float(s), "MI": float(mi)}
            for p, s, mi in results]


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GAP-5+6 Entanglement & Ansatz Diagnostics")
    parser.add_argument("--config", default="experiments/configs/qfl_tensor.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=64)
    parser.add_argument("--skip", nargs="+", default=[],
                        choices=["ansatz", "mi_epoch", "mi_correctness",
                                 "mi_noise"])
    parser.add_argument("--output", default=str(RESULTS_DIR / "entanglement_diagnostics.json"))
    args = parser.parse_args()

    device = get_device(args.device)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    base_cfg = load_config(args.config)
    base_cfg.training.epochs = args.epochs

    results = {}
    if "ansatz" not in args.skip:
        results["ansatz_profile"] = run_ansatz_profile(base_cfg)
    if "mi_epoch" not in args.skip:
        results["mi_across_epochs"] = mi_across_training(base_cfg, device)
    if "mi_correctness" not in args.skip:
        results["mi_vs_correctness"] = mi_vs_correctness(
            base_cfg, device, max_samples=args.max_samples)
    if "mi_noise" not in args.skip:
        results["mi_under_noise"] = mi_under_noise(base_cfg)

    save_results(results, args.output)
    print("\nEntanglement diagnostics complete.")
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
