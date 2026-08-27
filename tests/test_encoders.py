"""Tests for encoders."""

import torch
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.encoders.text_encoder import TextEncoder
from src.encoders.image_encoder import ImageEncoder


class TestTextEncoder:
    def test_forward_shape(self):
        model = TextEncoder(proj_dim=256)
        input_ids = torch.randint(0, 1000, (2, 128))
        attention_mask = torch.ones(2, 128)
        output = model(input_ids, attention_mask)
        assert output.shape == (2, 256)

    def test_gradient_flow(self):
        model = TextEncoder(proj_dim=256)
        input_ids = torch.randint(0, 1000, (2, 128))
        attention_mask = torch.ones(2, 128)
        output = model(input_ids, attention_mask)
        loss = output.sum()
        loss.backward()


class TestImageEncoder:
    def test_forward_shape(self):
        model = ImageEncoder(model_name="resnet18", proj_dim=256)
        images = torch.randn(2, 3, 224, 224)
        output = model(images)
        assert output.shape == (2, 256)

    def test_gradient_flow(self):
        model = ImageEncoder(model_name="resnet18", proj_dim=256)
        images = torch.randn(2, 3, 224, 224)
        output = model(images)
        loss = output.sum()
        loss.backward()
