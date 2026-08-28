"""ResNet18/ViT-based image encoder → projection → quantum-ready embedding."""

import torch
import torch.nn as nn
import torchvision.models as models


class ImageEncoder(nn.Module):
    """
    Image encoder with ResNet18 backbone + projection.
    Input: image tensor [batch, 3, 224, 224]
    Output: image embedding (d=256)

    Architecture:
        ResNet18 (512-d) → Linear(512→512) → GELU → Dropout(0.1) → Linear(512→256)
    """

    def __init__(self, model_name="resnet18", proj_dim=256, dropout=0.1,
                 pretrained=True):
        super().__init__()

        if model_name == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            backbone_dim = self.backbone.fc.in_features  # 512
            self.backbone.fc = nn.Identity()  # remove classification head
        elif model_name == "resnet34":
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet34(weights=weights)
            backbone_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif model_name == "vit_small":
            self.backbone = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
            backbone_dim = 768
            self.backbone.heads = nn.Identity()
        else:
            raise ValueError(f"Unknown model: {model_name}")

        self.proj = nn.Sequential(
            nn.Linear(backbone_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, proj_dim),
            nn.LayerNorm(proj_dim),
        )

    def forward(self, images):
        """
        Args:
            images: [batch, 3, 224, 224]
        Returns:
            image_emb: [batch, proj_dim=256]
        """
        feat = self.backbone(images)
        return self.proj(feat)
