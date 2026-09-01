"""
Ablation studies for Q-MMF (E6-E10).

    E6  : Qubit scaling      {4, 6, 8, 10, 12}
    E7  : PQC depth scaling  {1, 2, 3, 5}
    E8  : Encoding comparison {angle, amplitude, iqp}
    E9  : NISQ noise robustness {0.0, 0.005, 0.01, 0.02}
    E10 : Parameter efficiency (quantum vs classical param counting)

Usage:
    python experiments/run_ablation.py --experiment E6 --config experiments/configs/qfl_tensor.yaml
    python experiments/run_ablation.py --experiment all
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.training.train import (
    Config,
    load_config,
    set_seed,
    get_device,
    train_sentiment_only,
    save_results,
)
from src.evaluation.metrics import count_parameters


# ============================================================
# E6: Qubit scaling ablation
# ============================================================

def run_e6_qubit_scaling(cfg, device):
    """Accuracy vs number of qubits {4, 6, 8, 10, 12}."""
    results = {}
    for n_qubits in [4, 6, 8, 10, 12]:
        print(f"\n--- E6: n_qubits={n_qubits} ---")
        run_cfg = load_config_from(cfg)
        run_cfg.model.n_qubits = n_qubits
        set_seed(42)

        r = train_sentiment_only(run_cfg, device)
        results[n_qubits] = {
            "accuracy": r["test"].get("accuracy", 0),
            "f1_macro": r["test"].get("f1_macro", 0),
        }
        print(f"E6 | qubits={n_qubits} | acc={results[n_qubits]['accuracy']:.4f}")
    return results


# ============================================================
# E7: Circuit depth ablation
# ============================================================

def run_e7_depth_scaling(cfg, device):
    """Accuracy vs PQC depth {1, 2, 3, 5}."""
    results = {}
    for depth in [1, 2, 3, 5]:
        print(f"\n--- E7: n_layers={depth} ---")
        run_cfg = load_config_from(cfg)
        run_cfg.model.n_q_layers = depth
        set_seed(42)

        r = train_sentiment_only(run_cfg, device)
        results[depth] = {
            "accuracy": r["test"].get("accuracy", 0),
            "f1_macro": r["test"].get("f1_macro", 0),
        }
        print(f"E7 | depth={depth} | acc={results[depth]['accuracy']:.4f}")
    return results


# ============================================================
# E8: Encoding comparison
# ============================================================

def run_e8_encoding(cfg, device):
    """
    Compare encoding strategies.
    Note: requires QuantumFusionTensor variant that supports `encoding`
    parameter; angle encoding is the default used by other variants.
    """
    from src.quantum.encoding import get_encoding, estimate_encoding_cost

    results = {}
    for enc_name in ["angle", "amplitude", "iqp"]:
        print(f"\n--- E8: encoding={enc_name} ---")

        # Static cost analysis (always available without training)
        costs = estimate_encoding_cost(enc_name, n_features=4)
        entry = {"gate_cost": costs}

        # Training-based accuracy (only if a variant supports it)
        try:
            run_cfg = load_config_from(cfg)
            run_cfg.model.encoding = enc_name
            set_seed(42)
            r = train_sentiment_only(run_cfg, device)
            entry["accuracy"] = r["test"].get("accuracy", 0)
            entry["f1_macro"] = r["test"].get("f1_macro", 0)
        except Exception as e:
            print(f"  Training with '{enc_name}' skipped: {e}")
            entry["accuracy"] = None

        results[enc_name] = entry
    return results


# ============================================================
# E9: NISQ noise robustness
# ============================================================

def run_e9_noise(cfg, device):
    """
    Accuracy vs depolarizing noise probability {0, 0.005, 0.01, 0.02}.
    Uses default.mixed device with noise channels when p > 0.
    """
    results = {}
    for noise_prob in [0.0, 0.005, 0.01, 0.02]:
        print(f"\n--- E9: noise_prob={noise_prob} ---")
        run_cfg = load_config_from(cfg)
        run_cfg.noise.depolarizing_prob = noise_prob
        set_seed(42)

        try:
            r = train_sentiment_only(run_cfg, device)
            results[str(noise_prob)] = {
                "accuracy": r["test"].get("accuracy", 0),
                "f1_macro": r["test"].get("f1_macro", 0),
            }
        except Exception as e:
            print(f"  Noise sim at p={noise_prob} failed: {e}")
            results[str(noise_prob)] = {"accuracy": None, "error": str(e)}

        print(f"E9 | p={noise_prob} | acc={results[str(noise_prob)]['accuracy']}")
    return results


# ============================================================
# E10: Parameter efficiency
# ============================================================

def run_e10_parameter_counting(cfg, device="cpu"):
    """
    E10: Quantum vs classical param counting + ansatz profile
    (GAP-6 / §2.7: expressibility & entangling capability).
    """
    from src.training.train import build_model

    results = {}
    for model_name in ["qmmf", "early_fusion", "late_fusion", "cross_attention",
                       "show_and_tell"]:
        try:
            task = "captioning" if model_name == "show_and_tell" else "sentiment"
            model = build_model(cfg, task=task, model_name=model_name)
            counts = count_parameters(model)
            results[model_name] = counts
            print(f"E10 | {model_name:<16} | total={counts['total']:>12,} "
                  f"| quantum={counts['quantum']:>8,} "
                  f"| classical={counts['classical']:>12,} "
                  f"| ratio={counts['ratio']:.6f}")
        except Exception as e:
            print(f"E10 | {model_name}: skipped ({e})")

    # GAP-6: ansatz profiling (expressibility + entangling capability)
    try:
        from src.quantum.ansatz_metrics import profile_ansatz
        mcfg = getattr(cfg, "model", cfg)
        ansatz_name = getattr(mcfg, "fusion_type", "tensor")
        print(f"\n--- E10 Ansatz Profile ({ansatz_name}) ---")
        profile = profile_ansatz(
            ansatz_names=[ansatz_name],
            qubits_list=[4, 6, 8, 10],
            depths=[1, 2, 3, 5],
            n_samples=120,
            seed=42,
        )
        results["ansatz_profile"] = profile
        for row in profile:
            kl = row.get("expressibility_kl")
            mw = row.get("entangling_capability")
            if mw is not None:
                print(f"  q={row['n_qubits']} L={row['depth']} "
                      f"KL={kl if kl is not None else 'N/A'} MW={mw:.3f}")
    except Exception as e:
        print(f"  Ansatz profile skipped: {e}")

    return results


def load_config_from(base_cfg):
    """Deep-copy a Config so runs are independent."""
    raw = {
        "model": vars(base_cfg.model),
        "training": vars(base_cfg.training),
        "data": vars(base_cfg.data),
    }
    if hasattr(base_cfg, "noise"):
        raw["noise"] = vars(base_cfg.noise)
    for extra in ["checkpoint_dir", "log_dir", "model_name"]:
        if hasattr(base_cfg, extra):
            raw[extra] = getattr(base_cfg, extra)
    return Config(raw)


ABLATIONS = {
    "E6": run_e6_qubit_scaling,
    "E7": run_e7_depth_scaling,
    "E8": run_e8_encoding,
    "E9": run_e9_noise,
    "E10": run_e10_parameter_counting,
}


def main():
    parser = argparse.ArgumentParser(description="Q-MMF Ablation Studies")
    parser.add_argument("--config", default="experiments/configs/qfl_tensor.yaml")
    parser.add_argument("--experiment", default="all",
                        choices=["all", "E6", "E7", "E8", "E9", "E10"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epochs for faster ablations")
    parser.add_argument("--output", default="experiments/results/ablation_results.json")
    args = parser.parse_args()

    device = get_device(args.device)
    cfg = load_config(args.config)
    if args.epochs:
        cfg.training.epochs = args.epochs

    experiments = ABLATIONS.keys() if args.experiment == "all" else [args.experiment]

    all_results = {}
    for exp_id in experiments:
        print(f"\n{'#' * 60}")
        print(f"# Running {exp_id}")
        print(f"{'#' * 60}")
        fn = ABLATIONS[exp_id]
        if exp_id == "E10":
            all_results[exp_id] = fn(cfg)
        else:
            all_results[exp_id] = fn(cfg, device)

    save_results(all_results, args.output)
    print("\nAll requested ablations complete.")


if __name__ == "__main__":
    main()
