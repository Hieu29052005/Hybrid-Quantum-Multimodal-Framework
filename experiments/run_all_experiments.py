"""
Run all experiments for Q-MMF paper (E1-E11).
Generates all tables and figures.

Experiment matrix (per plan §6.4 + RESEARCH_GAP.md):
    E1 : Classical baselines (3 methods × 2 tasks)
    E2 : QFL-Tensor (8 qubits)
    E3 : QFL-Attention (8 qubits)
    E4 : QFL-Interference (8 qubits)
    E5 : Multi-task (shared Q layer) vs separate
    E6 : Ablation — #qubits {4,6,8,10,12}
    E7 : Ablation — PQC depth {1,2,3,5}
    E8 : Ablation — encoding {Angle, Amplitude, IQP}
    E9 : Noise simulation {0, 0.005, 0.01, 0.02}
    E10: Parameter efficiency + ansatz profile (GAP-6)
    E11: Entanglement diagnostics — MI curves, MI-vs-correctness,
         MI-collapse-under-noise (GAP-5)

Usage:
    python experiments/run_all_experiments.py                 # everything
    python experiments/run_all_experiments.py --skip E6 E7   # skip slow ablations
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.training.train import (
    load_config,
    set_seed,
    get_device,
    build_model,
    train_sentiment_only,
    train_multitask,
    save_results,
)
from src.evaluation.metrics import count_parameters

CONFIGS_DIR = Path(__file__).parent / "configs"
RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "paper" / "figures"


# ============================================================
# Experiment runners
# ============================================================

def run_classical_baselines(cfg, device):
    """E1: Train all classical baselines for both tasks."""
    results = {}
    for model_name in ["early_fusion", "late_fusion", "cross_attention"]:
        print(f"\n--- E1 baseline: {model_name} ---")
        set_seed(42)
        r = train_sentiment_only(cfg, device)
        results[model_name] = {
            "accuracy": r["test"].get("accuracy"),
            "f1_macro": r["test"].get("f1_macro"),
            "params": count_parameters(build_model(cfg, "sentiment", model_name)),
        }
    return results


def run_qfl_variant(cfg, fusion_type, device):
    """E2-E4: Train one QFL variant."""
    cfg.model.fusion_type = fusion_type
    set_seed(42)
    model = build_model(cfg, task="sentiment", model_name="qmmf")
    params = count_parameters(model)
    del model

    r = train_sentiment_only(cfg, device)
    return {
        "accuracy": r["test"].get("accuracy"),
        "f1_macro": r["test"].get("f1_macro"),
        "history": r["history"],
        "params": params,
    }


def run_multi_task_experiment(cfg, device):
    """E5: Joint multi-task training vs separate training."""
    print("\n--- E5: multi-task joint training ---")
    set_seed(42)
    joint = train_multitask(cfg, device)

    print("\n--- E5: separate training reference ---")
    set_seed(42)
    separate = train_sentiment_only(cfg, device)

    return {
        "joint_val_acc": joint.get("best_val_acc"),
        "joint_test_acc": joint["test"].get("accuracy"),
        "separate_test_acc": separate["test"].get("accuracy"),
        "joint_history": joint["history"],
    }


def run_qubit_scaling_experiment(cfg, device):
    from experiments.run_ablation import run_e6_qubit_scaling
    return run_e6_qubit_scaling(cfg, device)


def run_depth_scaling_experiment(cfg, device):
    from experiments.run_ablation import run_e7_depth_scaling
    return run_e7_depth_scaling(cfg, device)


def run_encoding_experiment(cfg, device):
    from experiments.run_ablation import run_e8_encoding
    return run_e8_encoding(cfg, device)


def run_noise_experiment(cfg, device):
    from experiments.run_ablation import run_e9_noise
    return run_e9_noise(cfg, device)


def run_parameter_counting(cfg, device="cpu"):
    from experiments.run_ablation import run_e10_parameter_counting
    return run_e10_parameter_counting(cfg)


def run_entanglement_diagnostics(cfg, device, max_samples=64):
    """E11 (GAP-5): MI across epochs + MI vs correctness + MI under noise."""
    from src.quantum.entanglement import CrossModalEntanglementAnalyzer
    from src.quantum.ansatz_metrics import profile_ansatz
    from src.training.train import (
        build_model, build_dataloaders, train_epoch_sentiment,
        collect_msa_predictions, _move_to_device,
    )
    from src.training.loss import MultiTaskLoss
    from src.training.optimizer_utils import setup_optimizer, clip_gradients
    from src.evaluation.metrics import sentiment_metrics
    import numpy as np

    mcfg = getattr(cfg, "model", cfg)
    n_qubits = getattr(mcfg, "n_qubits", 8)
    half = n_qubits // 2
    n_layers = getattr(mcfg, "n_q_layers", 3)

    results = {}

    # --- Part A: MI across training epochs ---
    print("  E11-A: MI across training epochs")
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
    epoch_mi = []
    for epoch in range(min(cfg.training.epochs, 10)):
        train_loss = train_epoch_sentiment(
            model, loaders["train"], criterion, optimizer, scheduler,
            device, cfg.training.max_grad_norm)
        preds, labels = collect_msa_predictions(model, loaders["val"], device)
        val_acc = sentiment_metrics(preds, labels)["accuracy"]
        w = next((p.detach().cpu() for n, p in model.named_parameters()
                   if "q_fusion.weights" in n), None)
        mi_mean = None
        if w is not None:
            analyzer = CrossModalEntanglementAnalyzer(n_qubits, n_layers, w)
            mi_vals = []
            for b_idx, batch in enumerate(loaders["val"]):
                if b_idx >= 4:
                    break
                batch = _move_to_device(batch, device)
                with torch.no_grad():
                    t = model.text_encoder(batch["input_ids"],
                                           batch["attention_mask"]).cpu().numpy()
                    im = model.image_encoder(batch["image"]).cpu().numpy()
                mi_vals.append(analyzer.analyze_batch(
                    t[:, :half], im[:, :half])["MI_text_image_mean"])
            mi_mean = float(np.mean(mi_vals)) if mi_vals else None
        epoch_mi.append({"epoch": epoch+1, "val_acc": val_acc,
                         "MI_mean": mi_mean})
        print(f"    Epoch {epoch+1}: acc={val_acc:.4f} MI={mi_mean}")
    results["mi_across_epochs"] = epoch_mi

    # --- Part B: MI under noise on single sample ---
    print("  E11-B: MI under noise (collapse)")
    text_sample = np.random.uniform(-np.pi/2, np.pi/2, half)
    image_sample = np.random.uniform(-np.pi/2, np.pi/2, half)
    w_rand = torch.randn(n_layers, n_qubits * 2) * 0.01
    analyzer = CrossModalEntanglementAnalyzer(n_qubits, n_layers, w_rand)
    mi_noise = analyzer.mi_under_noise(text_sample, image_sample,
                                       [0.0, 0.005, 0.01, 0.02])
    results["mi_under_noise"] = [{"p": float(p), "S": float(s), "MI": float(mi)}
                                  for p, s, mi in mi_noise]
    for p, s, mi in mi_noise:
        print(f"    p={p:.3f} MI={mi:.4f}")

    return results


# ============================================================
# Figures
# ============================================================

def generate_all_figures(results):
    """Generate all paper figures from experiment results."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from src.evaluation.visualize import (
            plot_quantum_vs_classical_accuracy,
            plot_qubit_scaling,
            plot_noise_robustness,
            plot_parameter_efficiency,
        )
    except ImportError:
        print("matplotlib/seaborn not available; skipping figures")
        return

    # Quantum vs Classical comparison
    sentiment = {}
    captioning = {}
    for key in ["E1_baselines"]:
        if key in results:
            for model_name, r in results[key].items():
                label = f"classical_{model_name}"
                sentiment[label] = {"accuracy": r.get("accuracy") or 0}
                captioning[label] = {"bleu_4": r.get("bleu_4") or 0}
    for key in ["E_qfl_tensor", "E_qfl_attention", "E_qfl_interference"]:
        if key in results and isinstance(results[key], dict):
            label = f"quantum_{key.replace('E_qfl_', '')}"
            acc = results[key].get("accuracy")
            if acc is not None:
                sentiment[label] = {"accuracy": acc}
                captioning.setdefault(label, {"bleu_4": 0})
    if sentiment or captioning:
        plot_quantum_vs_classical_accuracy(
            {"sentiment": sentiment, "captioning": captioning},
            save_path=str(FIGURES_DIR / "quantum_vs_classical.png"),
        )

    # Qubit scaling
    if "E6_qubit_scaling" in results:
        data = {}
        raw = results["E6_qubit_scaling"]
        # Convert int-keyed dict into per-variant structure expected by plot
        variant_data = {int(k): v for k, v in raw.items() if v.get("accuracy") is not None}
        if variant_data:
            data["qfl_tensor"] = variant_data
            plot_qubit_scaling(data, save_path=str(FIGURES_DIR / "qubit_scaling.png"))

    # Noise robustness
    if "E9_noise" in results:
        noise_raw = results["E9_noise"]
        clean = {float(k): v for k, v in noise_raw.items()
                 if isinstance(v, dict) and v.get("accuracy") is not None}
        if clean:
            plot_noise_robustness(
                {"qfl_tensor": clean},
                save_path=str(FIGURES_DIR / "noise_robustness.png"),
            )

    # Parameter efficiency
    if "E10_params" in results:
        param_counts = {k: v for k, v in results["E10_params"].items()
                        if isinstance(v, dict) and "quantum" in v}
        if param_counts:
            plot_parameter_efficiency(
                param_counts, save_path=str(FIGURES_DIR / "param_efficiency.png"))

    print(f"Figures saved to: {FIGURES_DIR}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Q-MMF Full Experiment Suite (E1-E10)")
    parser.add_argument("--config", default=str(CONFIGS_DIR / "qfl_tensor.yaml"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epochs (use small value for smoke tests)")
    parser.add_argument("--only", nargs="+", default=None,
                        choices=[f"E{i}" for i in range(1, 12)],
                        help="Run only these experiments")
    parser.add_argument("--skip", nargs="+", default=[],
                        choices=[f"E{i}" for i in range(1, 12)],
                        help="Skip these experiments (default skips none)")
    parser.add_argument("--no_figures", action="store_true")
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Device: {device}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    base_cfg = load_config(args.config)
    if args.epochs:
        base_cfg.training.epochs = args.epochs

    wanted = args.only if args.only else [f"E{i}" for i in range(1, 12)]
    wanted = [e for e in wanted if e not in args.skip]

    results = {}

    def want(eid):
        return eid in wanted

    # ---- Sentiment experiments ----
    if want("E1"):
        print("\n" + "#" * 60 + "\n# E1: Classical baselines\n" + "#" * 60)
        results["E1_baselines"] = run_classical_baselines(base_cfg, device)

    for eid, fusion in [("E2", "tensor"), ("E3", "attention"), ("E4", "interference")]:
        if want(eid):
            print(f"\n{'#' * 60}\n# {eid}: QFL-{fusion}\n{'#' * 60}")
            cfg = load_config(args.config)
            if args.epochs:
                cfg.training.epochs = args.epochs
            set_seed(args.seed)
            results[f"E_qfl_{fusion}"] = run_qfl_variant(cfg, fusion, device)

    if want("E5"):
        print("\n" + "#" * 60 + "\n# E5: Multi-task\n" + "#" * 60)
        cfg = load_config(args.config)
        if args.epochs:
            cfg.training.epochs = args.epochs
        set_seed(args.seed)
        results["E5_multi_task"] = run_multi_task_experiment(cfg, device)

    # ---- Ablations (delegate to run_ablation) ----
    ablation_map = [
        ("E6", "run_qubit_scaling_experiment", "E6_qubit_scaling"),
        ("E7", "run_depth_scaling_experiment", "E7_depth_scaling"),
        ("E8", "run_encoding_experiment", "E8_encoding"),
        ("E9", "run_noise_experiment", "E9_noise"),
        ("E10", "run_parameter_counting", "E10_params"),
    ]
    for eid, fn_name, result_key in ablation_map:
        if want(eid):
            print(f"\n{'#' * 60}\n# {eid}\n{'#' * 60}")
            cfg = load_config(args.config)
            if args.epochs and eid != "E10":
                cfg.training.epochs = args.epochs
            set_seed(args.seed)
            fn = globals()[fn_name]
            results[result_key] = fn(cfg, device) if eid != "E10" else fn(cfg)

    # ---- E11: Entanglement diagnostics (GAP-5) ----
    if want("E11"):
        print(f"\n{'#' * 60}\n# E11: Entanglement diagnostics (GAP-5)\n{'#' * 60}")
        cfg = load_config(args.config)
        if args.epochs:
            cfg.training.epochs = min(args.epochs, 10)
        set_seed(args.seed)
        results["E11_entanglement"] = run_entanglement_diagnostics(cfg, device)

    # ---- Save & visualize ----
    out_path = RESULTS_DIR / "results.json"
    save_results(results, str(out_path))

    if not args.no_figures:
        generate_all_figures(results)

    print("\nAll requested experiments complete!")
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
