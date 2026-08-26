"""Show & Tell baseline cho image captioning (LSTM decoder)."""

import torch
import torch.nn as nn
import math


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


class ShowAndTell(nn.Module):
    """
    Image captioning with LSTM decoder.
    Image features → LSTM → caption tokens
    """
    
    def __init__(self, vocab_size, d_model=256, n_layers=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        
        # Project image features to d_model
        self.image_proj = nn.Linear(512, d_model)  # ResNet18 outputs 512-d
        
        # LSTM decoder
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, image_features, decoder_input_ids, hidden=None):
        """
        Args:
            image_features: [batch, 512] image features from encoder
            decoder_input_ids: [batch, seq_len] decoder input tokens
            hidden: LSTM hidden state (h, c) tuple
        Returns:
            logits: [batch, seq_len, vocab_size]
            hidden: updated LSTM hidden state
        """
        B = image_features.shape[0]

        # Project image features
        img_emb = self.image_proj(image_features).unsqueeze(1)  # [B, 1, d_model]

        # Embed decoder tokens
        tok_emb = self.embed(decoder_input_ids)  # [B, seq_len, d_model]
        tok_emb = self.pos_enc(tok_emb)

        # Concatenate image embedding with token embeddings
        inputs = torch.cat([img_emb, tok_emb[:, :-1]], dim=1)  # [B, seq_len, d_model]

        # LSTM forward
        lstm_out, hidden = self.lstm(inputs, hidden)
        lstm_out = self.dropout(lstm_out)

        logits = self.fc_out(lstm_out)
        return logits, hidden

    @torch.no_grad()
    def generate(self, image_features, max_len=50, bos_token_id=2, eos_token_id=3):
        """Autoregressive caption generation (greedy decoding)."""
        B = image_features.shape[0]
        device = image_features.device

        generated = torch.full((B, 1), bos_token_id, dtype=torch.long, device=device)
        hidden = None

        for _ in range(max_len - 1):
            logits, hidden = self.forward(image_features, generated, hidden)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if (next_token == eos_token_id).all():
                break

        return generated
