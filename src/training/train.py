"""
Main training script for Q-MMF.

Supports 3 training modes:
    - sentiment : MSA only (MVSA / CMU-MOSI)
    - captioning: Image Captioning only (Flickr8k)
    - multitask : Alternate between MSA and Captioning batches

Usage (per plan Appendix A):
    python -m src.training.train --task sentiment --config src/training/configs/sentiment_config.yaml
    python -m src.training.train --task captioning --config src/training/configs/caption_config.yaml
    python -m src.training.train --task multitask --config src/training/configs/sentiment_config.yaml \
        --model qmmf --fusion_type tensor --n_qubits 8 --n_layers 3
"""

import torch
from torch.utils.data import DataLoader
import argparse
import json
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import yaml

from .loss import MultiTaskLoss
from .optimizer_utils import setup_optimizer, clip_gradients

logger = logging.getLogger(__name__)


# ============================================================
# Config: dot-access wrapper over nested YAML dict
# ============================================================

class Config:
    """Attribute-style access config that merges YAML sections."""

    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, Config(v) if isinstance(v, dict) else v)

    def get(self, key, default=None):
        return getattr(self, key, default)


def load_config(path):
    """Load YAML config file into a Config object."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return Config(raw)


# ============================================================
# Reproducibility & device
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_str="auto"):
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


# ============================================================
# Model builders (Q-MMF + classical baselines B1-B4)
# ============================================================

def build_model(cfg, task="sentiment", model_name="qmmf"):
    """
    Build model by name.
        qmmf           -> QuantumMultimodalFramework (shared quantum fusion)
        early_fusion   -> B1 classical baseline
        late_fusion    -> B2
        cross_attention-> B3
        show_and_tell  -> B4 classical captioning baseline
    """
    from ..models.q_mmf_model import QuantumMultimodalFramework
    from ..models.classical_models import ClassicalSentimentModel, ClassicalCaptioningModel

    # models nhận config flat (có d_model, vocab_size ở top level)
    # YAML nested → cfg.model chứa tất cả model hyperparams
    mcfg = getattr(cfg, "model", cfg)

    if model_name == "qmmf":
        return QuantumMultimodalFramework(mcfg)

    if task == "sentiment":
        fusion_map = {
            "early_fusion": "early",
            "late_fusion": "late",
            "cross_attention": "cross_attention",
        }
        if model_name not in fusion_map:
            raise ValueError(f"Unknown sentiment model: {model_name}")
        return ClassicalSentimentModel(mcfg, fusion=fusion_map[model_name])

    if task == "captioning":
        if model_name != "show_and_tell":
            raise ValueError(f"Unknown captioning model: {model_name}")
        return ClassicalCaptioningModel(mcfg)

    raise ValueError(f"Unknown task/model combination: {task}/{model_name}")


# ============================================================
# Data builders
# ============================================================

def collate_caption(batch):
    """Default collate works for all keys since tensors are pre-padded."""
    from torch.utils.data.dataloader import default_collate
    return default_collate(batch)


def build_dataloaders(cfg, task):
    """Create train/val/test loaders according to task."""
    data_cfg = cfg.data
    batch_size = cfg.training.batch_size
    num_workers = getattr(data_cfg, "num_workers", 2)

    if task in ("sentiment", "multitask"):
        from ..data.msa_dataset import MultimodalSentimentDataset

        def make_msa(split, shuffle):
            ds = MultimodalSentimentDataset(
                data_dir=data_cfg.msa_data_dir,
                split=split,
                tokenizer_name=getattr(cfg.model, "text_encoder", "bert-base-uncased"),
                max_length=data_cfg.max_text_length,
                image_size=data_cfg.image_size,
                dataset_name=data_cfg.msa_dataset,
            )
            return DataLoader(
                ds, batch_size=batch_size, shuffle=shuffle,
                num_workers=num_workers, pin_memory=True,
            )

        loaders = {
            "train": make_msa("train", True),
            "val": make_msa("val", False),
            "test": make_msa("test", False),
        }
        if task == "sentiment":
            return loaders
        # multitask: also need caption loaders
        cap = build_dataloaders(cfg, "captioning")
        return {**loaders, **{f"cap_{k}": v for k, v in cap.items()}}

    if task == "captioning":
        from ..data.caption_dataset import ImageCaptionDataset

        def make_cap(split, shuffle):
            ds = ImageCaptionDataset(
                data_dir=data_cfg.caption_data_dir,
                split=split,
                tokenizer_name=getattr(cfg.model, "text_encoder", "bert-base-uncased"),
                max_caption_length=data_cfg.max_caption_length,
                image_size=data_cfg.image_size,
            )
            return DataLoader(
                ds, batch_size=batch_size, shuffle=shuffle,
                num_workers=num_workers, pin_memory=True,
            )

        return {
            "train": make_cap("train", True),
            "val": make_cap("val", False),
            "test": make_cap("test", False),
        }

    raise ValueError(f"Unknown task: {task}")


# ============================================================
# Training loops
# ============================================================

def _move_to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def train_epoch_sentiment(model, loader, criterion, optimizer, scheduler, device, max_grad_norm=1.0):
    """Train one epoch for sentiment only."""
    model.train()
    total_loss, n_seen = 0.0, 0

    for step, batch in enumerate(loader):
        batch = _move_to_device(batch, device)
        optimizer.zero_grad()

        logits = model(task="msa", input_ids=batch["input_ids"],
                       attention_mask=batch["attention_mask"], images=batch["image"])
        loss = criterion(sentiment_logits=logits, sentiment_labels=batch["label"])
        loss.backward()
        clip_gradients(model, max_norm=max_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * batch["label"].size(0)
        n_seen += batch["label"].size(0)

    return total_loss / max(n_seen, 1)


def train_epoch_caption(model, loader, criterion, optimizer, scheduler, device, max_grad_norm=1.0):
    """Train one epoch for captioning only."""
    model.train()
    total_loss, n_seen = 0.0, 0

    for batch in loader:
        batch = _move_to_device(batch, device)
        optimizer.zero_grad()

        logits = model(task="caption", images=batch["image"],
                       decoder_input_ids=batch["decoder_input_ids"])
        loss = criterion(caption_logits=logits, caption_labels=batch["labels"])
        loss.backward()
        clip_gradients(model, max_norm=max_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        n_tokens = (batch["labels"] != -100).sum().item()
        total_loss += loss.item() * max(n_tokens, 1)
        n_seen += max(n_tokens, 1)

    return total_loss / max(n_seen, 1)


def train_epoch(model, msa_loader, caption_loader, criterion, optimizer, scheduler, device,
                max_grad_norm=1.0):
    """Train one epoch, alternating between MSA and Captioning batches."""
    model.train()
    total_loss = 0.0

    msa_iter = iter(msa_loader)
    cap_iter = iter(caption_loader)
    n_batches = min(len(msa_loader), len(caption_loader))

    for _ in range(n_batches):
        optimizer.zero_grad()
        loss = torch.tensor(0.0, device=device)

        try:
            msa_batch = next(msa_iter)
        except StopIteration:
            msa_iter = iter(msa_loader)
            msa_batch = next(msa_iter)
        msa_batch = _move_to_device(msa_batch, device)

        sentiment_logits = model(task="msa", input_ids=msa_batch["input_ids"],
                                 attention_mask=msa_batch["attention_mask"],
                                 images=msa_batch["image"])
        loss_msa = criterion(sentiment_logits=sentiment_logits,
                             sentiment_labels=msa_batch["label"])
        loss = loss + loss_msa

        try:
            cap_batch = next(cap_iter)
        except StopIteration:
            cap_iter = iter(caption_loader)
            cap_batch = next(cap_iter)
        cap_batch = _move_to_device(cap_batch, device)

        caption_logits = model(task="caption", images=cap_batch["image"],
                               decoder_input_ids=cap_batch["decoder_input_ids"])
        loss_cap = criterion(caption_logits=caption_logits,
                             caption_labels=cap_batch["labels"])
        loss = loss + loss_cap

        loss.backward()
        clip_gradients(model, max_norm=max_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

    return total_loss / max(n_batches, 1)


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def collect_msa_predictions(model, loader, device):
    """Run MSA inference, return (preds, labels) tensors."""
    model.eval()
    all_preds, all_labels = [], []

    for batch in loader:
        batch = _move_to_device(batch, device)
        logits = model(task="msa", input_ids=batch["input_ids"],
                       attention_mask=batch["attention_mask"], images=batch["image"])
        all_preds.append(logits.argmax(dim=-1).cpu())
        all_labels.append(batch["label"].cpu())

    return torch.cat(all_preds), torch.cat(all_labels)


@torch.no_grad()
def evaluate_msa(model, loader, device):
    """Evaluate MSA accuracy (kept for backward compat)."""
    preds, labels = collect_msa_predictions(model, loader, device)
    return (preds == labels).float().mean().item()


@torch.no_grad()
def generate_captions(model, loader, device, tokenizer=None):
    """Generate captions for all images in loader."""
    model.eval()
    hyps, refs = [], []

    for batch in loader:
        images = batch["image"].to(device)
        generated = model(task="caption", images=images)  # [B, L] token ids

        for i in range(generated.size(0)):
            ids = generated[i].tolist()
            if tokenizer is not None:
                text = tokenizer.decode(ids, skip_special_tokens=True)
                tokens = text.lower().split()
            else:
                tokens = [str(t) for t in ids]
            hyps.append(tokens)

        labels = batch.get("raw_references")
        if labels is not None:
            refs.extend(labels)
        else:
            for i in range(labels_len := batch["labels"].size(0)):
                ids = [t for t in batch["labels"][i].tolist() if t != -100]
                if tokenizer is not None:
                    text = tokenizer.decode(ids, skip_special_tokens=True)
                    tokens = text.lower().split()
                else:
                    tokens = [str(t) for t in ids]
                refs.append([tokens])

    return hyps, refs


@torch.no_grad()
def evaluate_caption_bleu(model, loader, device):
    """Quick BLEU evaluation using token-id sequences when tokenizer unavailable."""
    model.eval()
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

    smooth = SmoothingFunction().method1
    all_refs, all_hyps = [], []

    for batch in loader:
        images = batch["image"].to(device)
        generated = model(task="caption", images=images)
        for i in range(generated.size(0)):
            all_hyps.append([str(t) for t in generated[i].tolist()])
            ref_ids = [t for t in batch["labels"][i].tolist() if t != -100]
            all_refs.append([[str(t) for t in ref_ids]])

    bleu4 = corpus_bleu(all_refs, all_hyps, weights=(0.25, 0.25, 0.25, 0.25),
                        smoothing_function=smooth)
    return {"bleu_4": float(bleu4), "generated": all_hyps, "references": all_refs}


# ============================================================
# Checkpointing
# ============================================================

def save_checkpoint(model, optimizer, epoch, metric, checkpoint_dir, name="best"):
    """Save model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"{name}_epoch{epoch + 1}_{metric:.4f}.pt")
    torch.save({
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metric": metric,
    }, path)
    logger.info(f"Checkpoint saved: {path}")
    # Also save as latest symlink-style copy
    latest = os.path.join(checkpoint_dir, f"{name}_latest.pt")
    torch.save({
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metric": metric,
    }, latest)
    return path


def save_results(results, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {out_path}")


# ============================================================
# Full training pipelines
# ============================================================

def train_sentiment_only(cfg, device):
    """Full pipeline: train Q-MMF or baseline on MSA task."""
    from ..evaluation.metrics import sentiment_metrics

    model = build_model(cfg, task="sentiment", model_name=cfg.get("model_name", "qmmf"))
    model.to(device)
    loaders = build_dataloaders(cfg, "sentiment")

    criterion = MultiTaskLoss(lambda_sentiment=1.0, lambda_caption=0.0, lambda_reg=0.0)
    optimizer, scheduler = setup_optimizer(
        model, lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        warmup_steps=cfg.training.warmup_steps,
        total_steps=max(cfg.training.epochs * len(loaders["train"]), 1),
    )

    best_acc, history = 0.0, []
    for epoch in range(cfg.training.epochs):
        t0 = time.time()
        train_loss = train_epoch_sentiment(
            model, loaders["train"], criterion, optimizer, scheduler,
            device, cfg.training.max_grad_norm,
        )
        val_preds, val_labels = collect_msa_predictions(model, loaders["val"], device)
        val_metrics = sentiment_metrics(val_preds, val_labels)
        val_acc = val_metrics["accuracy"]

        history.append({
            "epoch": epoch + 1, "train_loss": train_loss,
            "val_accuracy": val_acc, "val_f1_macro": val_metrics["f1_macro"],
            "time": time.time() - t0,
        })
        logger.info(
            f"Epoch {epoch + 1}/{cfg.training.epochs} | "
            f"Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"F1(macro): {val_metrics['f1_macro']:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_acc,
                            cfg.checkpoint_dir, name="best_sentiment")

    test_preds, test_labels = collect_msa_predictions(model, loaders["test"], device)
    test_metrics = sentiment_metrics(test_preds, test_labels)
    logger.info(f"Test metrics: {json.dumps({k: v for k, v in test_metrics.items() if k != 'confusion_matrix'}, indent=2)}")

    return {"history": history, "best_val_acc": best_acc, "test": test_metrics}


def train_captioning_only(cfg, device):
    """Full pipeline: train on image captioning task."""
    model = build_model(cfg, task="captioning", model_name=cfg.get("model_name", "qmmf"))
    model.to(device)
    loaders = build_dataloaders(cfg, "captioning")

    criterion = MultiTaskLoss(lambda_sentiment=0.0, lambda_caption=1.0, lambda_reg=0.0)
    optimizer, scheduler = setup_optimizer(
        model, lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        warmup_steps=cfg.training.warmup_steps,
        total_steps=max(cfg.training.epochs * len(loaders["train"]), 1),
    )

    best_bleu, history = 0.0, []
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
        logger.info(f"Epoch {epoch + 1}/{cfg.training.epochs} | Loss: {train_loss:.4f} | BLEU-4: {val_bleu:.4f}")

        if val_bleu > best_bleu:
            best_bleu = val_bleu
            save_checkpoint(model, optimizer, epoch, val_bleu,
                            cfg.checkpoint_dir, name="best_caption")

    test_results = evaluate_caption_bleu(model, loaders["test"], device)
    return {"history": history, "best_val_bleu4": best_bleu,
            "test": {"bleu_4": test_results["bleu_4"]}}


def train_multitask(cfg, device):
    """Joint multi-task training: alternate MSA + Captioning batches."""
    model = build_model(cfg, task="sentiment", model_name=cfg.get("model_name", "qmmf"))
    model.to(device)
    loaders = build_dataloaders(cfg, "multitask")

    criterion = MultiTaskLoss(
        lambda_sentiment=cfg.training.lambda_sentiment,
        lambda_caption=cfg.training.lambda_caption,
        lambda_reg=cfg.training.lambda_reg,
    )
    steps_per_epoch = min(len(loaders["train"]), len(loaders["cap_train"]))
    optimizer, scheduler = setup_optimizer(
        model, lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        warmup_steps=cfg.training.warmup_steps,
        total_steps=max(cfg.training.epochs * steps_per_epoch, 1),
    )

    best_acc, history = 0.0, []
    for epoch in range(cfg.training.epochs):
        t0 = time.time()
        train_loss = train_epoch(
            model, loaders["train"], loaders["cap_train"], criterion,
            optimizer, scheduler, device, cfg.training.max_grad_norm,
        )
        val_preds, val_labels = collect_msa_predictions(model, loaders["val"], device)
        from ..evaluation.metrics import sentiment_metrics
        val_acc = sentiment_metrics(val_preds, val_labels)["accuracy"]

        history.append({
            "epoch": epoch + 1, "train_loss": train_loss,
            "val_accuracy": val_acc, "time": time.time() - t0,
        })
        logger.info(f"[MultiTask] Epoch {epoch + 1} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_acc,
                            cfg.checkpoint_dir, name="best_multitask")

    test_preds, test_labels = collect_msa_predictions(model, loaders["test"], device)
    from ..evaluation.metrics import sentiment_metrics
    test_metrics = sentiment_metrics(test_preds, test_labels)
    return {"history": history, "best_val_acc": best_acc, "test": test_metrics}


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Q-MMF Training Script")
    parser.add_argument("--task", required=True, choices=["sentiment", "captioning", "multitask"],
                        help="Which task(s) to train")
    parser.add_argument("--config", required=True, type=str,
                        help="Path to YAML config file")
    parser.add_argument("--model", default="qmmf",
                        choices=["qmmf", "early_fusion", "late_fusion",
                                 "cross_attention", "show_and_tell"],
                        help="Model architecture (default: qmmf)")
    parser.add_argument("--device", default="auto", help="cuda | cpu | auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", default=None, help="Experiment run name for outputs")

    # Common overrides (take precedence over config file)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--n_qubits", type=int, default=None)
    parser.add_argument("--n_layers", type=int, default=None)
    parser.add_argument("--fusion_type", default=None,
                        choices=["tensor", "attention", "interference"])
    parser.add_argument("--noise_prob", type=float, default=None,
                        help="Depolarizing noise probability (NISQ sim)")

    return parser.parse_args()


def apply_overrides(cfg, args):
    """Apply CLI overrides on top of YAML config."""
    if args.epochs is not None:
        cfg.training.epochs = args.epochs
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    if args.learning_rate is not None:
        cfg.training.learning_rate = args.learning_rate
    if args.n_qubits is not None:
        cfg.model.n_qubits = args.n_qubits
    if args.n_layers is not None:
        cfg.model.n_q_layers = args.n_layers
    if args.fusion_type is not None:
        cfg.model.fusion_type = args.fusion_type
    if args.noise_prob is not None:
        cfg.noise.depolarizing_prob = args.noise_prob
    return cfg


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    set_seed(args.seed)
    device = get_device(args.device)
    logger.info(f"Device: {device}")

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args)
    cfg.model_name = args.model

    run_name = args.run_name or f"{args.task}_{args.model}_{int(time.time())}"
    cfg.checkpoint_dir = str(Path(cfg.checkpoint_dir) / run_name)
    output_path = Path("experiments/results") / f"{run_name}.json"

    logger.info(f"Run: {run_name} | Task: {args.task} | Model: {args.model}")
    logger.info(f"Fusion: {getattr(cfg.model, 'fusion_type', 'n/a')} | "
                f"Qubits: {getattr(cfg.model, 'n_qubits', 'n/a')} | "
                f"Layers: {getattr(cfg.model, 'n_q_layers', 'n/a')}")

    if args.task == "sentiment":
        results = train_sentiment_only(cfg, device)
    elif args.task == "captioning":
        results = train_captioning_only(cfg, device)
    elif args.task == "multitask":
        results = train_multitask(cfg, device)
    else:
        raise ValueError(f"Unknown task: {args.task}")

    results["run_name"] = run_name
    results["args"] = vars(args)
    save_results(results, output_path)
    logger.info(f"Training complete. Best results logged; outputs at {output_path}")


if __name__ == "__main__":
    main()
