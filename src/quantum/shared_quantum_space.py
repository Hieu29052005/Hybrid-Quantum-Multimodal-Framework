"""
Shared Quantum Feature Space: quantum layer used by both MSA and Captioning.
Forces both tasks to learn common quantum cross-modal representations.
"""

import torch.nn as nn
from .quantum_fusion import QuantumFusionTensor, QuantumFusionAttention, QuantumFusionInterference
from .quantum_attention import QuantumMultiHeadAttention


class SharedQuantumSpace(nn.Module):
    """
    Shared quantum layer between MSA and Captioning tasks.

    For MSA: uses QuantumFusion to produce fused sentiment representation
    For Captioning: uses QuantumAttention for cross-attention in decoder
    """

    def __init__(self, n_qubits=8, n_layers=3, d_model=256, fusion_type="tensor"):
        super().__init__()

        if fusion_type == "tensor":
            self.q_fusion = QuantumFusionTensor(n_qubits, n_layers)
        elif fusion_type == "attention":
            self.q_fusion = QuantumFusionAttention(n_qubits, d_model)
        elif fusion_type == "interference":
            self.q_fusion = QuantumFusionInterference(n_qubits)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

        self.q_proj = nn.Linear(n_qubits, d_model)

        self.caption_adapter = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(self, text_emb, image_emb, task="msa"):
        """
        Args:
            text_emb:  [batch, d_model]
            image_emb: [batch, d_model]
            task: "msa" or "caption"
        Returns:
            For MSA:   fused_repr [batch, d_model]
            For caption: adapted_repr [batch, d_model]
        """
        half = text_emb.shape[-1] // 2
        fused_q = self.q_fusion(
            text_emb[:, :half],
            image_emb[:, :half],
        )
        fused = self.q_proj(fused_q)

        if task == "caption":
            fused = self.caption_adapter(fused)

        return fused
