"""
Task 1 experiments: Multimodal Sentiment Analysis (E1-E5, MSA part).

Runs classical baselines (B1-B3) and Q-MMF quantum variants on MVSA/MOSI.

Usage:
    python experiments/run_sentiment.py --config experiments/configs/qfl_tensor.yaml
    python experiments/run_sentiment.py --model early_fusion --epochs 30
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.training.train import (
    Config,
    load_config,
    set_seed,
    get_device,
    build_model,
    train_sentiment_only,
    save_results,
)
from src.evaluation.metrics import count_parameters


def run_single(cfg, model_name, device, run_name):
    """Train + evaluate one sentiment model."""
    print(f"\n{'=' * 60}")
    print(f"Experiment: {run_name}")
    print(f"Model: {model_name} | Fusion: {getattr(cfg.model, 'fusion_type', 'classical')}")
    print(f"{'=' * 60}")

    cfg.model_name = model_name
    t0 = time.time()
    results = train_sentiment_only(cfg, device)
    results["train_time_sec"] = time.time() - t0
    results["params"] = count_parameters(
        build_model(cfg, task="sentiment", model_name=model_name))
    return results


def main():
    parser = argparse.ArgumentParser(description="Q-MMF Sentiment Experiments")
    parser.add_argument("--config", default="experiments/configs/qfl_tensor.yaml")
    parser.add_argument("--models", nargs="+",
                        default=["qmmf"],
                        choices=["qmmf", "early_fusion", "late_fusion", "cross_attention"],
                        help="Which models to run (multiple allowed)")
    parser.add_argument("--fusion_types", nargs="+",
                        default=["tensor"],
                        choices=["tensor", "attention", "interference"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--n_qubits", type=int, default=None)
    parser.add_argument("--n_layers", type=int, default=None)
    parser.add_argument("--output", default="experiments/results/sentiment_results.json")
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Device: {device}")

    all_results = {}
    base_cfg = load_config(args.config)

    for model_name in args.models:
        if model_name == "qmmf":
            for fusion in args.fusion_types:
                cfg = load_config(args.config)  # fresh copy per run
                if args.epochs:
                    cfg.training.epochs = args.epochs
                if args.batch_size:
                    cfg.training.batch_size = args.batch_size
                if args.n_qubits:
                    cfg.model.n_qubits = args.n_qubits
                if args.n_layers:
                    cfg.model.n_q_layers = args.n_layers
                cfg.model.fusion_type = fusion

                set_seed(args.seed)
                run_name = f"E_{model_name}_{fusion}"
                all_results[run_name] = run_single(cfg, model_name, device, run_name)
        else:
            cfg = load_config(args.config)
            if args.epochs:
                cfg.training.epochs = args.epochs
            if args.batch_size:
                cfg.training.batch_size = args.batch_size

            set_seed(args.seed)
            run_name = f"E1_{model_name}"
            all_results[run_name] = run_single(cfg, model_name, device, run_name)

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY — Sentiment Analysis")
    print("=" * 70)
    print(f"{'Run':<32} {'Acc':>8} {'F1-macro':>10} {'MAE':>8}")
    print("-" * 70)
    for name, r in all_results.items():
        test = r.get("test", {})
        print(f"{name:<32} {test.get('accuracy', 0):>8.4f} "
              f"{test.get('f1_macro', 0):>10.4f} {test.get('mae', 0):>8.4f}")

    save_results(all_results, args.output)


if __name__ == "__main__":
    main()
