"""Early Fusion baseline cho MSA."""

import torch.nn as nn


class EarlyFusionMSA(nn.Module):
    def __init__(self, text_encoder, image_encoder, num_classes=3, proj_dim=256):
        super().__init__()
        self.text_enc = text_encoder
        self.image_enc = image_encoder
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim * 2, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_ids, attention_mask, images):
        t = self.text_enc(input_ids, attention_mask)
        i = self.image_enc(images)
        fused = torch.cat([t, i], dim=-1)
        return self.classifier(fused)
