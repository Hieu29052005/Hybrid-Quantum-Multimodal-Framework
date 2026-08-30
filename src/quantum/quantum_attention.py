"""
Quantum Attention Module (QAM): quantum-enhanced cross-attention
for image captioning decoder.
"""

import torch
import torch.nn as nn
import pennylane as qml
import math

n_attn_qubits = 4
dev_attn = qml.device("default.qubit", wires=n_attn_qubits)


@qml.QNode(dev_attn, interface="torch", diff_method="backprop")
def quantum_attention_score(query_feat, key_feat, attn_weights):
    """
    Quantum kernel attention:
    sim(q, k) = <ψ_q|ψ_k>² via inner product measurement
    """
    for i in range(n_attn_qubits // 2):
        qml.RY(query_feat[i], wires=i)
    for i in range(n_attn_qubits // 2, n_attn_qubits):
        qml.RY(key_feat[i - n_attn_qubits // 2], wires=i)
    for i in range(n_attn_qubits):
        qml.RY(attn_weights[i], wires=i)
    qml.CNOT(wires=[0, 2])
    qml.CNOT(wires=[1, 3])
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[2, 3])

    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))


class QuantumMultiHeadAttention(nn.Module):
    """
    Quantum-enhanced multi-head cross-attention.
    Queries come from caption tokens, Keys/Values from image features.
    """

    def __init__(self, d_model=256, n_heads=4, n_qubits=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.n_qubits = n_qubits

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.q_attn_weights = nn.ParameterList([
            nn.Parameter(torch.randn(n_qubits) * 0.01)
            for _ in range(n_heads)
        ])

    def forward(self, query, key, value, mask=None, return_attention=False):
        """
        Args:
            query: [batch, seq_len_q, d_model]
            key:   [batch, seq_len_k, d_model]
            value: [batch, seq_len_v, d_model]
            return_attention: if True, also return attention maps
                              (GAP-3 evidence: quantum attention visualization)
        Returns:
            attn_output: [batch, seq_len_q, d_model]
            attn_weights: [batch, n_heads, Lq, Lk] (only if return_attention)
        """
        B, Lq, _ = query.shape
        _, Lk, _ = key.shape

        Q = self.q_proj(query).view(B, Lq, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(B, Lk, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(B, Lk, self.n_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.zeros(B, self.n_heads, Lq, Lk, device=query.device)
        for h in range(self.n_heads):
            for i in range(Lq):
                for j in range(Lk):
                    score = quantum_attention_score(
                        Q[:, h, i, : self.n_qubits],
                        K[:, h, j, : self.n_qubits],
                        self.q_attn_weights[h],
                    )
                    attn_scores[:, h, i, j] = score

        attn_scores = attn_scores / math.sqrt(self.head_dim)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=-1)

        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        out = self.out_proj(attn_output)
        if return_attention:
            return out, attn_weights
        return out
