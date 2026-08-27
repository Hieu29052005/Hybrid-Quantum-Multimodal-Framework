"""Tests for training components."""

import torch
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.loss import MultiTaskLoss
from src.training.optimizer_utils import setup_optimizer


class TestMultiTaskLoss:
    def test_sentiment_loss(self):
        criterion = MultiTaskLoss()
        logits = torch.randn(4, 3)
        labels = torch.randint(0, 3, (4,))
        loss = criterion(sentiment_logits=logits, sentiment_labels=labels)
        assert loss.item() > 0

    def test_caption_loss(self):
        criterion = MultiTaskLoss()
        logits = torch.randn(4, 20, 1000)
        labels = torch.randint(0, 1000, (4, 20))
        loss = criterion(caption_logits=logits, caption_labels=labels)
        assert loss.item() > 0

    def test_combined_loss(self):
        criterion = MultiTaskLoss()
        s_logits = torch.randn(4, 3)
        s_labels = torch.randint(0, 3, (4,))
        c_logits = torch.randn(4, 20, 1000)
        c_labels = torch.randint(0, 1000, (4, 20))
        loss = criterion(
            sentiment_logits=s_logits, sentiment_labels=s_labels,
            caption_logits=c_logits, caption_labels=c_labels,
        )
        assert loss.item() > 0


class TestOptimizerUtils:
    def test_setup_optimizer(self):
        import torch.nn as nn
        model = nn.Linear(256, 3)
        optimizer, scheduler = setup_optimizer(model, lr=1e-4, warmup_steps=10, total_steps=100)
        assert optimizer is not None
        assert scheduler is not None
