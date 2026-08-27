"""Tests for the full Q-MMF model."""

import torch
import pytest
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.q_mmf_model import QuantumMultimodalFramework


@pytest.fixture
def config():
    return SimpleNamespace(
        d_model=256,
        n_qubits=8,
        n_q_layers=3,
        fusion_type="tensor",
        image_encoder="resnet18",
        text_encoder="bert-base-uncased",
        freeze_bert_layers=6,
        num_sentiment_classes=3,
        vocab_size=1000,
        max_caption_length=50,
        n_heads=8,
        decoder_layers=2,
    )


class TestQuantumMultimodalFramework:
    def test_sentiment_forward(self, config):
        model = QuantumMultimodalFramework(config)
        input_ids = torch.randint(0, 1000, (2, 128))
        attention_mask = torch.ones(2, 128)
        images = torch.randn(2, 3, 224, 224)
        logits = model(task="msa", input_ids=input_ids, attention_mask=attention_mask, images=images)
        assert logits.shape == (2, 3)

    def test_caption_forward(self, config):
        model = QuantumMultimodalFramework(config)
        images = torch.randn(2, 3, 224, 224)
        decoder_input_ids = torch.randint(0, 1000, (2, 20))
        logits = model(task="caption", images=images, decoder_input_ids=decoder_input_ids)
        assert logits.shape == (2, 20, 1000)

    def test_caption_generate(self, config):
        model = QuantumMultimodalFramework(config)
        images = torch.randn(2, 3, 224, 224)
        generated = model(task="caption", images=images)
        assert generated.shape[0] == 2

    def test_gradient_flow(self, config):
        model = QuantumMultimodalFramework(config)
        input_ids = torch.randint(0, 1000, (2, 128))
        attention_mask = torch.ones(2, 128)
        images = torch.randn(2, 3, 224, 224)
        logits = model(task="msa", input_ids=input_ids, attention_mask=attention_mask, images=images)
        loss = logits.sum()
        loss.backward()
        quantum_grads = [p.grad for n, p in model.named_parameters()
                        if "quantum" in n and p.grad is not None]
        assert len(quantum_grads) > 0
