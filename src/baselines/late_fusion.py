"""Late Fusion baseline cho MSA."""

import torch.nn as nn


class LateFusionMSA(nn.Module):
    def __init__(self, text_encoder, image_encoder, num_classes=3, proj_dim=256):
        super().__init__()
        self.text_enc = text_encoder
        self.image_enc = image_encoder
        
        # Separate classifiers for each modality
        self.text_classifier = nn.Sequential(
            nn.Linear(proj_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
        
        self.image_classifier = nn.Sequential(
            nn.Linear(proj_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_ids, attention_mask, images):
        t = self.text_enc(input_ids, attention_mask)
        i = self.image_enc(images)
        
        # Separate predictions
        text_logits = self.text_classifier(t)
        image_logits = self.image_classifier(i)
        
        # Average predictions (late fusion)
        return (text_logits + image_logits) / 2
