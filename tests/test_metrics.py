"""Tests for evaluation metrics."""

import torch
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import sentiment_metrics, caption_metrics, count_parameters


class TestSentimentMetrics:
    def test_basic_metrics(self):
        preds = torch.tensor([0, 1, 2, 0, 1])
        labels = torch.tensor([0, 1, 2, 1, 1])
        metrics = sentiment_metrics(preds, labels, num_classes=3)
        assert "accuracy" in metrics
        assert "f1_macro" in metrics
        assert "f1_weighted" in metrics
        assert metrics["accuracy"] == 0.8

    def test_confusion_matrix(self):
        preds = torch.tensor([0, 1, 2, 0, 1])
        labels = torch.tensor([0, 1, 2, 1, 1])
        metrics = sentiment_metrics(preds, labels, num_classes=3)
        assert "confusion_matrix" in metrics


class TestCaptionMetrics:
    def test_bleu_scores(self):
        references = [[["a", "cat", "is", "sleeping"]]]
        hypotheses = [["a", "cat", "is", "sleeping"]]
        metrics = caption_metrics(references, hypotheses)
        assert "bleu_1" in metrics
        assert "bleu_4" in metrics
        assert metrics["bleu_4"] > 0.9


class TestCountParameters:
    def test_count(self):
        import torch.nn as nn
        model = nn.Sequential(nn.Linear(256, 128), nn.Linear(128, 3))
        counts = count_parameters(model)
        assert counts["total"] > 0
        assert counts["quantum"] == 0
        assert counts["classical"] == counts["total"]
