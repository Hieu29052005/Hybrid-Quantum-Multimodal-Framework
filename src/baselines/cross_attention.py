"""Cross-Attention baseline cho MSA."""

import torch
import torch.nn as nn
import math


class CrossAttentionMSA(nn.Module):
    def __init__(self, text_encoder, image_encoder, num_classes=3,
                 proj_dim=256, n_heads=4):
        super().__init__()
        self.text_enc = text_encoder
        self.image_enc = image_encoder

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=proj_dim, num_heads=n_heads, batch_first=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_ids, attention_mask, images):
        t = self.text_enc(input_ids, attention_mask).unsqueeze(1)  # [B, 1, d]
        i = self.image_enc(images).unsqueeze(1)  # [B, 1, d]

        attn_out, _ = self.cross_attn(query=t, key=i, value=i)
        fused = attn_out.squeeze(1)
        return self.classifier(fused)
