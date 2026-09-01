"""
GAP-4 (RESEARCH_GAP.md): NISQ noise robustness study — component sweep,
error propagation in autoregressive decoding, crossover threshold, and
residual mitigation (QMLSC-style).

Study targets:
    - Component-wise noise: fusion-PQC vs decoder-attention-PQC vs
      classifier-head-PQC → "compare noise sensitivity of three PQC positions"
    - Autoregressive error propagation under noise → "error propagation
      through autoregressive decode steps"
    - Per-step token entropy under noise → "per-step token entropy in
      decoding under noise"
    - Crossover threshold → "crossover noise threshold where quantum < classical"
    - Residual mitigation → "residual connection (QMLSC style) as mitigation"

Cách dùng:
    python experiments/run_noise_study.py --config experiments/configs/qfl_tensor.yaml --epochs 3
    python experiments/run_noise_study.py --skip_train --output experiments/results/noise_study.json
"""

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from src.training.train import (
    Config,
    load_config,
    set_seed,
    get_device,
    build_model,
    build_dataloaders,
    train_sentiment_only,
    train_captioning_only,
    train_epoch_caption,
    evaluate_msa,
    evaluate_caption_bleu,
    save_checkpoint,
    save_results,
    _move_to_device,
)
from src.training.loss import MultiTaskLoss
from src.training.optimizer_utils import setup_optimizer, clip_gradients
from src.evaluation.metrics import count_parameters
from src.quantum.noise_wrapper import (
    apply_component_noise,
    restore_component_noise,
    ResidualQuantumMitigation,
)
from src.evaluation.noise_analysis import (
    per_step_token_entropy,
    per_step_disagreement,
    divergence_onset,
    find_crossover_threshold,
    degradation_curve,
    relative_noise_sensitivity,
)

RESULTS_DIR = Path(__file__).parent / "results"
CKPT_DIR = Path("checkpoints/_noise_study")

NOISE_PROBS = [0.0, 0.005, 0.01, 0.02]


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


def _ckpt_path(variant):
    return CKPT_DIR / f"{variant}_latest.pt"


# ------------------------------------------------------------------
# A. Train reference models for each variant
# ------------------------------------------------------------------

def build_variant(base_cfg, variant, device):
    """
    variant: "qmmf_fusion" | "qmmf_quantum_head" | "qmmf_hybrid_decoder"
             | "classical_early"
    """
    cfg = _deep_copy_cfg(base_cfg)
    mcfg = getattr(cfg, "model", cfg)

    if variant == "qmmf_fusion":
        mcfg.use_quantum_head = False
        mcfg.decoder_type = "transformer"
    elif variant == "qmmf_quantum_head":
        mcfg.use_quantum_head = True
        mcfg.head_qubits = 4
        mcfg.head_q_layers = 2
        mcfg.decoder_type = "transformer"
    elif variant == "qmmf_hybrid_decoder":
        mcfg.use_quantum_head = False
        mcfg.decoder_type = "hybrid_quantum"
        mcfg.qam_heads = 1
        mcfg.qam_qubits = 4
    elif variant == "classical_early":
        return build_model(cfg, task="sentiment", model_name="early_fusion")
    else:
        raise ValueError(f"Unknown variant: {variant}")
    return build_model(cfg, task="sentiment", model_name="qmmf")


def train_variant(base_cfg, variant, device, skip_train=False):
    """Train variant (or load checkpoint). Returns model on device."""
    ckpt = _ckpt_path(variant)
    model = build_variant(base_cfg, variant, device).to(device)

    if skip_train and ckpt.exists():
        ckpt_data = torch.load(ckpt, map_location=device, weights_only=True)
        model.load_state_dict(ckpt_data["model_state_dict"])
        print(f"  Loaded checkpoint: {ckpt}")
        return model

    cfg = _deep_copy_cfg(base_cfg)
    set_seed(42)

    if variant == "classical_early":
        r = train_sentiment_only(cfg, device)
    elif variant in ("qmmf_fusion", "qmmf_quantum_head"):
        r = train_sentiment_only(cfg, device)
    elif variant == "qmmf_hybrid_decoder":
        # Train with multitask so decoder gets meaningful image features
        cfg.checkpoint_dir = str(CKPT_DIR)
        r = train_captioning_only(cfg, device)
    else:
        raise ValueError(variant)

    # Rebuild model and load best for evaluation
    model = build_variant(base_cfg, variant, device).to(device)
    best_ckpt = Path(cfg.checkpoint_dir) / f"best_*_latest.pt"
    import glob
    candidates = sorted(glob.glob(str(CKPT_DIR / f"{variant}_latest.pt")))
    # Also check standard naming
    candidates += sorted(glob.glob(str(Path(cfg.checkpoint_dir) / "*_latest.pt")))
    for c in candidates:
        try:
            data = torch.load(c, map_location=device, weights_only=True)
            model.load_state_dict(data["model_state_dict"])
            print(f"  Loaded: {c}")
            break
        except Exception:
            continue

    return model


# ------------------------------------------------------------------
# B. Component-wise noise sweep
# ------------------------------------------------------------------

COMPONENT_MAP = {
    "qmmf_fusion":         "fusion",
    "qmmf_quantum_head":   "quantum_head",
    "qmmf_hybrid_decoder": "decoder_attention",
}


def evaluate_model_acc(model, loaders, device, task="msa"):
    if task == "msa":
        return evaluate_msa(model, loaders["val"], device)
    elif task == "caption":
        return evaluate_caption_bleu(model, loaders["val"], device)["bleu_4"]
    return 0


def noise_sweep(base_cfg, device, skip_train=False, task_limit_n=64):
    """
    Sweep noise p across all applicable component/variant combos.
    Returns records: [{variant, component, p, metric_name, value}, ...].
    """
    records = []
    variants_to_test = ["qmmf_fusion", "qmmf_quantum_head", "qmmf_hybrid_decoder",
                        "classical_early"]
    loaders = build_dataloaders(base_cfg, "sentiment")
    cap_loaders = build_dataloaders(base_cfg, "captioning")

    for variant in variants_to_test:
        print(f"\n--- Noise sweep: {variant} ---")
        try:
            model = train_variant(base_cfg, variant, device, skip_train=skip_train)
        except Exception as e:
            print(f"  Training failed for {variant}: {e}")
            continue
        model.eval()
        component = COMPONENT_MAP.get(variant)

        for p in NOISE_PROBS:
            if variant == "classical_early" and p > 0:
                # Classical model — no quantum noise; just record clean once
                continue
            if component and p > 0:
                try:
                    apply_component_noise(model, component, p=p)
                except Exception as e:
                    print(f"  apply_noise({component}, p={p}) failed: {e}")
                    continue

            msa_acc = evaluate_model_acc(model, loaders, device, "msa")
            records.append({"variant": variant, "component": component,
                            "p": p, "metric": "accuracy", "value": msa_acc})

            if variant == "qmmf_hybrid_decoder":
                bleu = evaluate_model_acc(model, cap_loaders, device, "caption")
                records.append({"variant": variant, "component": component,
                                "p": p, "metric": "bleu_4", "value": bleu})
            print(f"  {variant} | p={p} | acc={msa_acc:.4f}")

            if p > 0:
                restore_component_noise(model)

    return records


# ------------------------------------------------------------------
# C. Error propagation & per-step entropy (hybrid decoder only)
# ------------------------------------------------------------------

@torch.no_grad()
def error_propagation_analysis(base_cfg, device, skip_train=False):
    """GAP-4: error propagation through autoregressive decode steps."""
    print("\n--- Error propagation analysis (hybrid decoder) ---")
    cfg = _deep_copy_cfg(base_cfg)
    mcfg = getattr(cfg, "model", cfg)
    mcfg.use_quantum_head = False
    mcfg.decoder_type = "hybrid_quantum"
    mcfg.qam_heads = 1
    mcfg.qam_qubits = 4

    try:
        model = train_variant(base_cfg, "qmmf_hybrid_decoder", device, skip_train=skip_train)
    except Exception as e:
        print(f"  Cannot train hybrid decoder: {e}")
        return {"error": str(e)}
    model.eval()

    cap_loaders = build_dataloaders(cfg, "captioning")
    batch = next(iter(cap_loaders["val"]))
    images = batch["image"][:8].to(device)  # small batch for speed

    results_clean = per_step_token_entropy(model, images, device=device)
    clean_ids = results_clean["token_ids"]

    propagation_curves = {}
    entropy_curves = {}

    for p in [0.005, 0.01, 0.02]:
        apply_component_noise(model, "decoder_attention", p=p)
        results_noisy = per_step_token_entropy(model, images, device=device)
        noisy_ids = results_noisy["token_ids"]

        onset = divergence_onset(clean_ids, noisy_ids)
        disagreement = per_step_disagreement(clean_ids, noisy_ids)
        propagation_curves[str(p)] = {
            "onset_mean": onset["mean_onset"],
            "frac_diverged": onset["frac_diverged"],
            "disagreement_curve": disagreement,
        }
        entropy_curves[str(p)] = results_noisy["entropies"].mean(dim=1).tolist()
        restore_component_noise(model)

    entropy_curves["clean"] = results_clean["entropies"].mean(dim=1).tolist()

    return {"propagation": propagation_curves, "entropy": entropy_curves}


# ------------------------------------------------------------------
# D. Crossover threshold
# ------------------------------------------------------------------

def crossover_analysis(noise_records):
    """Find crossover p where quantum accuracy < classical baseline."""
    classical_acc = None
    for r in noise_records:
        if r["variant"] == "classical_early" and r["metric"] == "accuracy":
            classical_acc = r["value"]
            break
    if classical_acc is None:
        return {"error": "classical baseline not found"}

    results = {}
    for variant in COMPONENT_MAP:
        pts = sorted([(r["p"], r["value"]) for r in noise_records
                       if r["variant"] == variant and r["metric"] == "accuracy"],
                      key=lambda x: x[0])
        if len(pts) < 2:
            continue
        ps = [p for p, _ in pts]
        accs = [a for _, a in pts]
        cross = find_crossover_threshold(ps, accs, [classical_acc] * len(accs))
        deg = degradation_curve(pts)
        results[variant] = {
            "crossover": cross,
            "degradation_curve_pct": deg,
            "sensitivity": relative_noise_sensitivity(pts),
        }
    return results


# ------------------------------------------------------------------
# E. Residual mitigation
# ------------------------------------------------------------------

def residual_mitigation_study(base_cfg, device, skip_train=False,
                              alphas=(0.0, 0.25, 0.5)):
    """Compare acc under noise with/without residual skip (QMLSC-style)."""
    print("\n--- Residual mitigation study ---")
    model = train_variant(base_cfg, "qmmf_fusion", device, skip_train=skip_train)
    model.eval()

    loaders = build_dataloaders(base_cfg, "sentiment")
    results = []

    for p in [0.0, 0.01, 0.02]:
        # Without mitigation
        if p > 0:
            apply_component_noise(model, "fusion", p=p)
        acc_clean_wrapper = evaluate_model_acc(model, loaders, device)
        results.append({"alpha": 0.0, "p": p, "acc": acc_clean_wrapper})
        if p > 0:
            restore_component_noise(model)

        # With ResidualQuantumMitigation at different alphas
        for alpha in alphas:
            if alpha == 0.0:
                continue
            wrapper = ResidualQuantumMitigation(
                model.shared_quantum.q_fusion,
                n_qubits=model.shared_quantum.q_fusion.n_qubits if hasattr(
                    model.shared_quantum.q_fusion, "n_qubits") else 8,
                alpha=alpha,
            )
            model.shared_quantum.q_fusion = wrapper
            if p > 0:
                # Apply noise on top of residual wrapper
                apply_component_noise(model, "fusion", p=p)
            acc_res = evaluate_model_acc(model, loaders, device)
            results.append({"alpha": alpha, "p": p, "acc": acc_res})
            if p > 0:
                restore_component_noise(model)
            # restore clean
            restore_component_noise(model)
    return results


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GAP-4 NISQ Noise Study")
    parser.add_argument("--config", default="experiments/configs/qfl_tensor.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_train", action="store_true",
                        help="Load existing checkpoints instead of training")
    parser.add_argument("--skip", nargs="+", default=[],
                        choices=["sweep", "propagation", "crossover", "mitigation"])
    parser.add_argument("--output", default=str(RESULTS_DIR / "noise_study.json"))
    args = parser.parse_args()

    device = get_device(args.device)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    base_cfg = load_config(args.config)
    if args.epochs:
        base_cfg.training.epochs = args.epochs

    results = {}

    noise_records = []
    if "sweep" not in args.skip:
        noise_records = noise_sweep(base_cfg, device, skip_train=args.skip_train)
        results["sweep"] = noise_records

    if "propagation" not in args.skip:
        results["error_propagation"] = error_propagation_analysis(
            base_cfg, device, skip_train=args.skip_train)

    if "crossover" not in args.skip and noise_records:
        results["crossover_analysis"] = crossover_analysis(noise_records)

    if "mitigation" not in args.skip:
        results["residual_mitigation"] = residual_mitigation_study(
            base_cfg, device, skip_train=args.skip_train)

    save_results(results, args.output)
    print("\nNoise study complete.")
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
