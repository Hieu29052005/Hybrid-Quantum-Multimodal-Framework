"""
Classical baselines wrapper for comparison with quantum models.
"""

import torch
import torch.nn as nn
from ..encoders.text_encoder import TextEncoder
from ..encoders.image_encoder import ImageEncoder


class ClassicalSentimentModel(nn.Module):
    """Classical sentiment model with configurable fusion strategy."""

    def __init__(self, config, fusion="early"):
        super().__init__()
        self.fusion = fusion
        self.text_enc = TextEncoder(
            model_name=getattr(config, "text_encoder", "bert-base-uncased"),
            proj_dim=config.d_model,
            freeze_bert_layers=getattr(config, "freeze_bert_layers", 6),
        )
        self.image_enc = ImageEncoder(
            model_name=getattr(config, "image_encoder", "resnet18"),
            proj_dim=config.d_model,
        )

        num_classes = getattr(config, "num_sentiment_classes", 3)
        d = config.d_model

        if fusion == "early":
            self.classifier = nn.Sequential(
                nn.Linear(d * 2, 512), nn.GELU(), nn.Dropout(0.3),
                nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(256, num_classes),
            )
        elif fusion == "late":
            self.text_classifier = nn.Sequential(
                nn.Linear(d, 256), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(256, num_classes),
            )
            self.image_classifier = nn.Sequential(
                nn.Linear(d, 256), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(256, num_classes),
            )
        elif fusion == "cross_attention":
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=d, num_heads=4, batch_first=True,
            )
            self.classifier = nn.Sequential(
                nn.Linear(d, 256), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(256, num_classes),
            )

    def forward(self, input_ids, attention_mask, images):
        t = self.text_enc(input_ids, attention_mask)
        i = self.image_enc(images)

        if self.fusion == "early":
            return self.classifier(torch.cat([t, i], dim=-1))
        elif self.fusion == "late":
            logits_t = self.text_classifier(t)
            logits_i = self.image_classifier(i)
            return (logits_t + logits_i) / 2
        elif self.fusion == "cross_attention":
            t_seq = t.unsqueeze(1)
            i_seq = i.unsqueeze(1)
            attn_out, _ = self.cross_attn(query=t_seq, key=i_seq, value=i_seq)
            return self.classifier(attn_out.squeeze(1))


class ClassicalCaptioningModel(nn.Module):
    """Classical Show-and-Tell captioning model."""

    def __init__(self, config):
        super().__init__()
        self.image_enc = ImageEncoder(
            model_name=getattr(config, "image_encoder", "resnet18"),
            proj_dim=config.d_model,
        )

        from ..decoders.caption_decoder import TransformerCaptionDecoder
        self.decoder = TransformerCaptionDecoder(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            nhead=getattr(config, "n_heads", 8),
            num_layers=getattr(config, "decoder_layers", 3),
            max_seq_len=getattr(config, "max_caption_length", 50),
        )

    def forward(self, images, decoder_input_ids=None):
        i_emb = self.image_enc(images)
        if decoder_input_ids is not None:
            return self.decoder(decoder_input_ids, memory=i_emb.unsqueeze(1))
        else:
            return self.decoder.generate(
                memory=i_emb.unsqueeze(1),
                max_len=getattr(self.config, "max_caption_length", 50) if hasattr(self, "config") else 50,
            )
