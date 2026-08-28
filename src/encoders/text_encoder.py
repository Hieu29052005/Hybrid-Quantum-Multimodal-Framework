"""BERT-based text encoder → projection → quantum-ready embedding."""

import torch
import torch.nn as nn
from transformers import AutoModel


class TextEncoder(nn.Module):
    """
    BERT-base encoder with projection layer.
    Input: tokenized text (input_ids, attention_mask)
    Output: text embedding (d=256)

    Architecture:
        BERT-base (768-d) → Linear(768→512) → GELU → Dropout(0.1) → Linear(512→256)
    """

    def __init__(self, model_name="bert-base-uncased", proj_dim=256, dropout=0.1,
                 freeze_bert_layers=6):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size  # 768

        # Freeze bottom layers of BERT (optional, for efficiency)
        if freeze_bert_layers > 0:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
            for i in range(freeze_bert_layers):
                for param in self.bert.encoder.layer[i].parameters():
                    param.requires_grad = False

        # Projection head
        self.proj = nn.Sequential(
            nn.Linear(self.hidden_size, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, proj_dim),
            nn.LayerNorm(proj_dim),
        )

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
        Returns:
            text_emb: [batch, proj_dim=256]
        """
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = output.last_hidden_state[:, 0, :]  # [CLS] token
        return self.proj(cls_token)
