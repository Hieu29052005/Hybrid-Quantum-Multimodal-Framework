"""
Multi-task loss for Q-MMF.
L_total = λ₁·L_sentiment + λ₂·L_caption + λ₃·L_regularization
"""

import torch
import torch.nn as nn


class MultiTaskLoss(nn.Module):
    def __init__(self, lambda_sentiment=1.0, lambda_caption=1.0,
                 lambda_reg=0.1, label_smoothing=0.1):
        super().__init__()
        self.lambda1 = lambda_sentiment
        self.lambda2 = lambda_caption
        self.lambda3 = lambda_reg

        self.sentiment_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.caption_loss = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, sentiment_logits=None, caption_logits=None,
                sentiment_labels=None, caption_labels=None,
                quantum_features=None):
        """
        Compute total loss.
        """
        device = "cpu"
        if sentiment_logits is not None:
            device = sentiment_logits.device
        elif caption_logits is not None:
            device = caption_logits.device

        total_loss = torch.tensor(0.0, device=device)

        if sentiment_logits is not None and sentiment_labels is not None:
            loss_s = self.sentiment_loss(sentiment_logits, sentiment_labels)
            total_loss = total_loss + self.lambda1 * loss_s

        if caption_logits is not None and caption_labels is not None:
            loss_c = self.caption_loss(
                caption_logits.reshape(-1, caption_logits.size(-1)),
                caption_labels.reshape(-1),
            )
            total_loss = total_loss + self.lambda2 * loss_c

        if quantum_features is not None:
            loss_reg = torch.mean(torch.norm(quantum_features, p=2, dim=-1))
            total_loss = total_loss + self.lambda3 * loss_reg

        return total_loss
