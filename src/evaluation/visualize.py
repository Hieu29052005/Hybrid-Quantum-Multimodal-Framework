"""
Visualization utilities for results analysis.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_quantum_vs_classical_accuracy(results_dict, save_path="figures/quantum_vs_classical.png"):
    """Bar chart: accuracy comparison quantum vs classical for each task."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    models = list(results_dict["sentiment"].keys())
    accs = [results_dict["sentiment"][m]["accuracy"] for m in models]
    colors = ["#2196F3" if "quantum" in m.lower() else "#FF9800" for m in models]
    ax.barh(models, accs, color=colors)
    ax.set_xlabel("Accuracy")
    ax.set_title("Sentiment Analysis: Quantum vs Classical")
    ax.set_xlim(0, 1)

    ax = axes[1]
    models = list(results_dict["captioning"].keys())
    bleu4 = [results_dict["captioning"][m]["bleu_4"] for m in models]
    colors = ["#2196F3" if "quantum" in m.lower() else "#FF9800" for m in models]
    ax.barh(models, bleu4, color=colors)
    ax.set_xlabel("BLEU-4")
    ax.set_title("Image Captioning: Quantum vs Classical")

    plt.tight_layout()
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_qubit_scaling(ablation_results, save_path="figures/qubit_scaling.png"):
    """Line plot: accuracy vs number of qubits."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for variant, data in ablation_results.items():
        qubits = sorted(data.keys())
        accs = [data[q]["accuracy"] for q in qubits]
        ax.plot(qubits, accs, "o-", label=variant, linewidth=2, markersize=8)

    ax.set_xlabel("Number of Qubits", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Quantum Resource Scaling: Accuracy vs Qubits", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_noise_robustness(noise_results, save_path="figures/noise_robustness.png"):
    """Line plot: accuracy vs noise level for each model."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for model_name, data in noise_results.items():
        noise_levels = sorted(data.keys())
        accs = [data[n]["accuracy"] for n in noise_levels]
        ax.plot(noise_levels, accs, "o-", label=model_name, linewidth=2, markersize=8)

    ax.set_xlabel("Depolarizing Noise Probability", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("NISQ Noise Robustness Analysis", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_parameter_efficiency(param_counts, save_path="figures/param_efficiency.png"):
    """Bar chart: parameter count comparison quantum vs classical."""
    fig, ax = plt.subplots(figsize=(10, 6))

    models = list(param_counts.keys())
    quantum = [param_counts[m]["quantum"] for m in models]
    classical = [param_counts[m]["classical"] for m in models]

    x = np.arange(len(models))
    width = 0.35
    ax.bar(x - width / 2, quantum, width, label="Quantum", color="#2196F3")
    ax.bar(x + width / 2, classical, width, label="Classical", color="#FF9800")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_ylabel("Number of Parameters")
    ax.set_title("Parameter Efficiency: Quantum vs Classical")
    ax.legend()
    ax.set_yscale("log")
    plt.tight_layout()
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_tsne_fused_representations(features, labels, title="t-SNE of Fused Representations",
                                     save_path="figures/tsne_fused.png"):
    """t-SNE visualization of quantum vs classical fused features."""
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(features.cpu().numpy())

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        embeddings_2d[:, 0], embeddings_2d[:, 1],
        c=labels.cpu().numpy(), cmap="RdYlBu", alpha=0.7, s=30,
    )
    plt.colorbar(scatter)
    plt.title(title, fontsize=14)
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.tight_layout()
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
