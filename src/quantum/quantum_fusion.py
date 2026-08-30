"""
Quantum Fusion Layer (QFL): learns entangled cross-modal representations
between text and image using Parameterized Quantum Circuits (PQC).

3 variants:
    1. QFL-Tensor:    Angle encoding + entangling layers (baseline)
    2. QFL-Attention: Quantum kernel attention for cross-modal scoring
    3. QFL-Interference: Density matrix + quantum interference term
"""

import torch
import torch.nn as nn
import pennylane as qml
import numpy as np


# ============================================================
# Variant 1: QFL-Tensor (Baseline)
# ============================================================

n_qubits = 8  # 4 for text, 4 for image
n_layers = 3

dev_tensor = qml.device("default.qubit", wires=n_qubits)


@qml.QNode(dev_tensor, interface="torch", diff_method="backprop")
def _tensor_circuit(text_emb, image_emb, weights):
    """
    Quantum fusion circuit for text + image.

    Encoding strategy: Angle encoding
    - Text features → qubits 0..3 (RY rotations)
    - Image features → qubits 4..7 (RY rotations)

    Entangling: CNOT cross-modal + intra-modal
    """
    # === Angle Encoding ===
    for i in range(n_qubits // 2):
        qml.RY(text_emb[i], wires=i)
        qml.RZ(text_emb[i + n_qubits // 2], wires=i)
    for i in range(n_qubits // 2, n_qubits):
        qml.RY(image_emb[i - n_qubits // 2], wires=i)
        qml.RZ(image_emb[i - n_qubits // 2 + n_qubits // 2], wires=i)

    # === Entangling Layers ===
    for layer in range(n_layers):
        # Cross-modal entanglement (text ↔ image)
        for i in range(n_qubits // 2):
            qml.CNOT(wires=[i, i + n_qubits // 2])

        # Intra-modal entanglement
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])

        # Parameterized rotations
        for i in range(n_qubits):
            qml.RY(weights[layer, i], wires=i)
            qml.RZ(weights[layer, i + n_qubits], wires=i)

    # === Measurement ===
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


class QuantumFusionTensor(nn.Module):
    """QFL-Tensor: angle encoding + entangling PQC."""

    def __init__(self, n_qubits=8, n_layers=3):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.weights = nn.Parameter(
            torch.randn(n_layers, n_qubits * 2) * 0.01
        )

    def forward(self, text_emb, image_emb):
        """
        Args:
            text_emb:  [batch, n_qubits//2] (4 features for text)
            image_emb: [batch, n_qubits//2] (4 features for image)
        Returns:
            fused: [batch, n_qubits] (8-dim quantum representation)
        """
        # Pad embeddings to qubit count if needed
        t = text_emb[:, :self.n_qubits // 2]
        i = image_emb[:, :self.n_qubits // 2]

        results = []
        for b in range(t.shape[0]):
            result = _tensor_circuit(t[b], i[b], self.weights)
            results.append(torch.stack(result))

        return torch.stack(results)  # [batch, n_qubits]


# ============================================================
# Variant 2: QFL-Attention (Quantum Kernel Attention)
# ============================================================

dev_attention = qml.device("default.qubit", wires=4)


@qml.QNode(dev_attention, interface="torch", diff_method="backprop")
def _attention_kernel(query, key, attn_weights):
    """
    Quantum kernel for computing attention similarity.
    sim(q, k) = <0|U†(q) · (Z⊗Z) · U(k)|0>
    """
    # Encode query
    for i in range(2):
        qml.RY(query[i], wires=i)
        qml.RZ(query[i + 2], wires=i)
    # Encode key (controlled rotation)
    for i in range(2):
        qml.CRY(key[i], wires=[i, i + 2])
    # Parameterized interaction
    for i in range(4):
        qml.RY(attn_weights[i], wires=i)
    # Entanglement
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[1, 2])

    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliZ(3))


class QuantumFusionAttention(nn.Module):
    """QFL-Attention: quantum kernel attention scores."""

    def __init__(self, n_qubits=8, d_proj=256):
        super().__init__()
        self.n_qubits = n_qubits
        self.query_proj = nn.Linear(d_proj, n_qubits // 2)
        self.key_proj = nn.Linear(d_proj, n_qubits // 2)
        self.value_proj = nn.Linear(d_proj, n_qubits)
        self.attn_weights = nn.Parameter(torch.randn(4) * 0.01)
        self.out_proj = nn.Linear(n_qubits, n_qubits)

    def forward(self, text_emb, image_emb):
        """
        Cross-attention: text queries, image keys/values.
        """
        q = self.query_proj(text_emb)
        k = self.key_proj(image_emb)
        v = self.value_proj(image_emb)

        # Compute quantum attention score per sample in batch
        scores = []
        for b in range(q.shape[0]):
            score = _attention_kernel(q[b], k[b], self.attn_weights)
            scores.append(score)
        score = torch.stack(scores)
        weight = torch.sigmoid(score).unsqueeze(-1)

        fused = weight * v[:, :self.n_qubits] + (1 - weight) * self.out_proj(
            v[:, :self.n_qubits]
        )
        return fused


# ============================================================
# Variant 3: QFL-Interference (Quantum Interference Fusion)
# ============================================================

dev_interference = qml.device("default.qubit", wires=n_qubits)


@qml.QNode(dev_interference, interface="torch", diff_method="backprop")
def _interference_circuit(text_state, image_state, phase_weights):
    """
    Quantum interference-inspired fusion.
    Based on double-slit experiment analogy:
    P(both) = |α ψ_text + β ψ_image|² = α²P_t + β²P_i + 2αβ√(P_t·P_i)cos(θ)
    """
    # Encode text state
    for i in range(n_qubits // 2):
        qml.RY(text_state[i], wires=i)
    # Encode image state
    for i in range(n_qubits // 2, n_qubits):
        qml.RY(image_state[i - n_qubits // 2], wires=i)
    # Phase shift for interference
    for i in range(n_qubits):
        qml.RZ(phase_weights[i], wires=i)
    # Controlled interference (cross terms)
    for i in range(n_qubits // 2):
        qml.CNOT(wires=[i, i + n_qubits // 2])
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


class QuantumFusionInterference(nn.Module):
    """QFL-Interference: quantum interference-inspired fusion."""

    def __init__(self, n_qubits=8):
        super().__init__()
        self.n_qubits = n_qubits
        self.phase_weights = nn.Parameter(torch.randn(n_qubits) * 0.01)
        self.alpha = nn.Parameter(torch.tensor(0.7))  # text weight
        self.beta = nn.Parameter(torch.tensor(0.3))   # image weight

    def forward(self, text_emb, image_emb):
        alpha_norm = torch.softmax(torch.stack([self.alpha, self.beta]), dim=0)[0]
        beta_norm = torch.softmax(torch.stack([self.alpha, self.beta]), dim=0)[1]

        results = []
        for b in range(text_emb.shape[0]):
            result = _interference_circuit(
                text_emb[b] * alpha_norm,
                image_emb[b] * beta_norm,
                self.phase_weights
            )
            results.append(torch.stack(result))

        return torch.stack(results)
