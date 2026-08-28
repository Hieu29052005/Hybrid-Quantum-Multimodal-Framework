"""
Quantum Multimodal Framework (Q-MMF): Full hybrid model.
Supports both Sentiment Analysis and Image Captioning tasks.
"""

import torch
import torch.nn as nn
from ..encoders.text_encoder import TextEncoder
from ..encoders.image_encoder import ImageEncoder
from ..quantum.shared_quantum_space import SharedQuantumSpace
from ..decoders.caption_decoder import TransformerCaptionDecoder, HybridQuantumCaptionDecoder
from ..decoders.sentiment_head import SentimentHead, QuantumSentimentHead


class QuantumMultimodalFramework(nn.Module):
    """
    Hybrid Quantum-Classical Multimodal Framework.

    Shared components:
        - TextEncoder (BERT)
        - ImageEncoder (ResNet18)
        - SharedQuantumSpace (PQC fusion)

    Task-specific heads:
        - SentimentHead (MLP classifier) for MSA
        - TransformerCaptionDecoder for Image Captioning
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.text_encoder = TextEncoder(
            model_name=getattr(config, "text_encoder", "bert-base-uncased"),
            proj_dim=config.d_model,
            freeze_bert_layers=getattr(config, "freeze_bert_layers", 6),
        )
        self.image_encoder = ImageEncoder(
            model_name=getattr(config, "image_encoder", "resnet18"),
            proj_dim=config.d_model,
        )

        self.shared_quantum = SharedQuantumSpace(
            n_qubits=getattr(config, "n_qubits", 8),
            n_layers=getattr(config, "n_q_layers", 3),
            d_model=config.d_model,
            fusion_type=getattr(config, "fusion_type", "tensor"),
        )

        # GAP-4: quantum classifier head variant (so sánh noise sensitivity)
        if getattr(config, "use_quantum_head", False):
            self.sentiment_head = QuantumSentimentHead(
                d_model=config.d_model,
                num_classes=getattr(config, "num_sentiment_classes", 3),
                n_qubits=getattr(config, "head_qubits", 4),
                n_layers=getattr(config, "head_q_layers", 2),
            )
        else:
            self.sentiment_head = SentimentHead(
                d_model=config.d_model,
                num_classes=getattr(config, "num_sentiment_classes", 3),
            )

        # GAP-3: hybrid quantum decoder (QAM cross-attention trong decoder)
        decoder_type = getattr(config, "decoder_type", "transformer")
        if decoder_type == "hybrid_quantum":
            self.caption_decoder = HybridQuantumCaptionDecoder(
                vocab_size=config.vocab_size,
                d_model=config.d_model,
                nhead=getattr(config, "n_heads", 8),
                num_layers=getattr(config, "decoder_layers", 3),
                max_seq_len=getattr(config, "max_caption_length", 50),
                qam_heads=getattr(config, "qam_heads", 1),
                qam_qubits=getattr(config, "qam_qubits", 4),
            )
        else:
            self.caption_decoder = TransformerCaptionDecoder(
                vocab_size=config.vocab_size,
                d_model=config.d_model,
                nhead=getattr(config, "n_heads", 8),
                num_layers=getattr(config, "decoder_layers", 3),
                max_seq_len=getattr(config, "max_caption_length", 50),
            )

    def forward_sentiment(self, input_ids, attention_mask, images):
        """Forward pass for sentiment analysis."""
        t_emb = self.text_encoder(input_ids, attention_mask)
        i_emb = self.image_encoder(images)
        fused = self.shared_quantum(t_emb, i_emb, task="msa")
        logits = self.sentiment_head(fused)
        return logits

    def forward_caption(self, images, decoder_input_ids=None, text_emb=None):
        """
        Forward pass for image captioning.
        During training: teacher forcing with decoder_input_ids
        During inference: autoregressive generation
        """
        i_emb = self.image_encoder(images)
        if text_emb is None:
            text_emb = torch.zeros_like(i_emb)
        q_feat = self.shared_quantum(text_emb, i_emb, task="caption")

        if decoder_input_ids is not None:
            logits = self.caption_decoder(
                decoder_input_ids, memory=q_feat.unsqueeze(1)
            )
            return logits
        else:
            return self.caption_decoder.generate(
                memory=q_feat.unsqueeze(1),
                max_len=getattr(self.config, "max_caption_length", 50),
                bos_token_id=getattr(self.config, "bos_token_id", 2),
                eos_token_id=getattr(self.config, "eos_token_id", 3),
            )

    def forward(self, task="msa", **kwargs):
        if task == "msa":
            return self.forward_sentiment(
                kwargs["input_ids"], kwargs["attention_mask"], kwargs["images"]
            )
        elif task == "caption":
            return self.forward_caption(
                kwargs["images"], kwargs.get("decoder_input_ids"),
                kwargs.get("text_emb"),
            )
        else:
            raise ValueError(f"Unknown task: {task}")
