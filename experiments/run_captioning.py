"""
Task 2 experiments: Image Captioning (E1 captioning part, E5).

Runs Show-and-Tell baseline and Q-MMF quantum captioning on Flickr8k.

Usage:
    python experiments/run_captioning.py --config experiments/configs/qfl_tensor.yaml
    python experiments/run_captioning.py --model show_and_tell --epochs 30
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.training.train import (
    load_config,
    set_seed,
    get_device,
    build_dataloaders,
    train_epoch_caption,
    evaluate_caption_bleu,
    save_checkpoint,
    save_results,
)
from src.training.loss import MultiTaskLoss
from src.training.optimizer_utils import setup_optimizer
from src.evaluation.metrics import count_parameters


def build_caption_model(cfg, model_name):
    from src.models.q_mmf_model import QuantumMultimodalFramework
    from src.models.classical_models import ClassicalCaptioningModel

    if model_name == "qmmf":
        return QuantumMultimodalFramework(cfg)
    if model_name == "show_and_tell":
        return ClassicalCaptioningModel(cfg)
    raise ValueError(f"Unknown captioning model: {model_name}")


def run_caption_experiment(cfg, model_name, device, run_name):
    """Train + evaluate one captioning model."""
    print(f"\n{'=' * 60}")
    print(f"Experiment: {run_name}")
    print(f"{'=' * 60}")

    set_seed(42)
    model = build_caption_model(cfg, model_name).to(device)
    loaders = build_dataloaders(cfg, "captioning")

    criterion = MultiTaskLoss(lambda_sentiment=0.0, lambda_caption=1.0)
    optimizer, scheduler = setup_optimizer(
        model, lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        warmup_steps=cfg.training.warmup_steps,
        total_steps=max(cfg.training.epochs * len(loaders["train"]), 1),
    )

    history, best_bleu = [], 0.0
    for epoch in range(cfg.training.epochs):
        t0 = time.time()
        train_loss = train_epoch_caption(
            model, loaders["train"], criterion, optimizer, scheduler,
            device, cfg.training.max_grad_norm,
        )
        val_bleu = evaluate_caption_bleu(model, loaders["val"], device)["bleu_4"]

        history.append({
            "epoch": epoch + 1, "train_loss": train_loss,
            "val_bleu4": val_bleu, "time": time.time() - t0,
        })
        print(f"Epoch {epoch + 1}/{cfg.training.epochs} | "
              f"Loss: {train_loss:.4f} | Val BLEU-4: {val_bleu:.4f}")

        if val_bleu > best_bleu:
            best_bleu = val_bleu
            save_checkpoint(model, optimizer, epoch, val_bleu,
                            cfg.checkpoint_dir, name=f"best_{run_name}")

    test_results = evaluate_caption_bleu(model, loaders["test"], device)

    return {
        "history": history,
        "best_val_bleu4": best_bleu,
        "test": {"bleu_4": test_results["bleu_4"]},
        "params": count_parameters(build_caption_model(cfg, model_name)),
        "train_time_sec": sum(h["time"] for h in history),
    }


def main():
    parser = argparse.ArgumentParser(description="Q-MMF Captioning Experiments")
    parser.add_argument("--config", default="experiments/configs/qfl_tensor.yaml")
    parser.add_argument("--models", nargs="+", default=["qmmf", "show_and_tell"],
                        choices=["qmmf", "show_and_tell"])
    parser.add_argument("--fusion_types", nargs="+", default=["tensor"],
                        choices=["tensor", "attention", "interference"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--output", default="experiments/results/caption_results.json")
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Device: {device}")

    all_results = {}

    for model_name in args.models:
        if model_name == "qmmf":
            for fusion in args.fusion_types:
                cfg = load_config(args.config)
                if args.epochs:
                    cfg.training.epochs = args.epochs
                if args.batch_size:
                    cfg.training.batch_size = args.batch_size
                cfg.model.fusion_type = fusion

                set_seed(args.seed)
                run_name = f"E_{model_name}_{fusion}"
                all_results[run_name] = run_caption_experiment(
                    cfg, model_name, device, run_name)
        else:
            cfg = load_config(args.config)
            if args.epochs:
                cfg.training.epochs = args.epochs
            if args.batch_size:
                cfg.training.batch_size = args.batch_size

            set_seed(args.seed)
            run_name = f"E1_{model_name}"
            all_results[run_name] = run_caption_experiment(
                cfg, model_name, device, run_name)

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY — Image Captioning")
    print("=" * 70)
    print(f"{'Run':<32} {'BLEU-4':>10} {'#Params':>12}")
    print("-" * 70)
    for name, r in all_results.items():
        bleu = r.get("test", {}).get("bleu_4", 0)
        params = r.get("params", {}).get("total", 0)
        print(f"{name:<32} {bleu:>10.4f} {params:>12,}")

    save_results(all_results, args.output)


if __name__ == "__main__":
    main()
