"""Tests for decoders."""

import torch
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.decoders.sentiment_head import SentimentHead
from src.decoders.caption_decoder import TransformerCaptionDecoder


class TestSentimentHead:
    def test_forward_shape(self):
        model = SentimentHead(d_model=256, num_classes=3)
        x = torch.randn(2, 256)
        output = model(x)
        assert output.shape == (2, 3)


class TestTransformerCaptionDecoder:
    def test_forward_shape(self):
        model = TransformerCaptionDecoder(vocab_size=1000, d_model=256, nhead=8, num_layers=2)
        tgt = torch.randint(0, 1000, (2, 20))
        memory = torch.randn(2, 1, 256)
        output = model(tgt, memory)
        assert output.shape == (2, 20, 1000)

    def test_generate(self):
        model = TransformerCaptionDecoder(vocab_size=1000, d_model=256, nhead=8, num_layers=2)
        memory = torch.randn(2, 1, 256)
        generated = model.generate(memory, max_len=10, bos_token_id=2, eos_token_id=3)
        assert generated.shape[0] == 2
        assert generated.shape[1] <= 10
