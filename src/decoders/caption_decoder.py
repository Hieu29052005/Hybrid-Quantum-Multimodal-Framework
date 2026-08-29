"""Transformer decoder cho image captioning.

GAP-3 (RESEARCH_GAP.md): quantum kernel cross-attention INSIDE the
autoregressive decoder — HybridQuantumCaptionDecoder thay thế cross-attention
cổ điển bằng QuantumMultiHeadAttention ở mỗi decoder layer.
"""

import torch
import torch.nn as nn
import math

from ..quantum.quantum_attention import QuantumMultiHeadAttention


def _causal_mask(seq_len, device):
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return mask.masked_fill(mask == 1, float("-inf"))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class TransformerCaptionDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=3, max_seq_len=50):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_seq_len)
        self.scale = math.sqrt(d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=512,
            dropout=0.1, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, memory, tgt_mask=None):
        """
        Args:
            tgt: [batch, seq_len] decoder input tokens
            memory: [batch, 1, d_model] image features (from quantum layer)
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        tgt_emb = self.pos_enc(self.embed(tgt) * self.scale)
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        return self.fc_out(output)

    @torch.no_grad()
    def generate(self, memory, max_len=50, bos_token_id=2, eos_token_id=3):
        """Autoregressive caption generation (greedy decoding)."""
        B = memory.shape[0]
        device = memory.device
        generated = torch.full((B, 1), bos_token_id, dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            logits = self.forward(generated, memory)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if (next_token == eos_token_id).all():
                break

        return generated


class QuantumTransformerDecoderLayer(nn.Module):
    """
    GAP-3: Decoder layer với cross-attention được thay bằng QAM
    (quantum kernel attention). Self-attention giữ cổ điển (nhanh,
    chỉ tương quan caption↔caption); quantum attention chuyên cho
    tương liên caption↔image.
    """

    def __init__(self, d_model=256, nhead=8, dim_feedforward=512, dropout=0.1,
                 qam_heads=1, qam_qubits=4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                               batch_first=True)
        self.q_cross_attn = QuantumMultiHeadAttention(d_model, n_heads=qam_heads,
                                                      n_qubits=qam_qubits)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.last_cross_attention = None  # lưu để visualize (GAP-3 evidence)

    def forward(self, tgt, memory, tgt_mask=None):
        x = tgt
        # masked self-attention (classical)
        sa, _ = self.self_attn(x, x, x, attn_mask=tgt_mask, need_weights=False)
        x = self.norm1(x + self.dropout(sa))
        # quantum kernel cross-attention (caption queries → image memory)
        ca, attn = self.q_cross_attn(x, memory, memory, return_attention=True)
        self.last_cross_attention = attn.detach()
        x = self.norm2(x + self.dropout(ca))
        ff = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.norm3(x + self.dropout(ff))


class HybridQuantumCaptionDecoder(nn.Module):
    """
    GAP-3: Autoregressive decoder dùng QuantumTransformerDecoderLayer.
    Cùng interface với TransformerCaptionDecoder (forward / generate).
    """

    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=3,
                 max_seq_len=50, qam_heads=1, qam_qubits=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_seq_len, dropout)
        self.scale = math.sqrt(d_model)

        self.layers = nn.ModuleList([
            QuantumTransformerDecoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=512,
                dropout=dropout, qam_heads=qam_heads, qam_qubits=qam_qubits,
            )
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, memory, tgt_mask=None):
        """
        Args:
            tgt: [batch, seq_len]
            memory: [batch, seq_len_k, d_model] (thường [B, 1, d])
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        if tgt_mask is None:
            tgt_mask = _causal_mask(tgt.size(1), tgt.device)
        x = self.pos_enc(self.embed(tgt) * self.scale)
        for layer in self.layers:
            x = layer(x, memory, tgt_mask=tgt_mask)
        return self.fc_out(x)

    @torch.no_grad()
    def generate(self, memory, max_len=50, bos_token_id=2, eos_token_id=3):
        B = memory.shape[0]
        device = memory.device
        generated = torch.full((B, 1), bos_token_id, dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            logits = self.forward(generated, memory)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if (next_token == eos_token_id).all():
                break

        return generated
