"""Tests for quantum fusion layers."""

import torch
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.quantum.quantum_fusion import (
    QuantumFusionTensor,
    QuantumFusionAttention,
    QuantumFusionInterference,
)


class TestQuantumFusionTensor:
    def test_forward_shape(self):
        model = QuantumFusionTensor(n_qubits=8, n_layers=3)
        text_emb = torch.randn(2, 4)
        image_emb = torch.randn(2, 4)
        output = model(text_emb, image_emb)
        assert output.shape == (2, 8)

    def test_gradient_flow(self):
        model = QuantumFusionTensor(n_qubits=8, n_layers=2)
        text_emb = torch.randn(2, 4, requires_grad=True)
        image_emb = torch.randn(2, 4, requires_grad=True)
        output = model(text_emb, image_emb)
        loss = output.sum()
        loss.backward()
        assert model.weights.grad is not None

    def test_single_sample(self):
        model = QuantumFusionTensor(n_qubits=8, n_layers=3)
        text_emb = torch.randn(1, 4)
        image_emb = torch.randn(1, 4)
        output = model(text_emb, image_emb)
        assert output.shape == (1, 8)


class TestQuantumFusionInterference:
    def test_forward_shape(self):
        model = QuantumFusionInterference(n_qubits=8)
        text_emb = torch.randn(2, 4)
        image_emb = torch.randn(2, 4)
        output = model(text_emb, image_emb)
        assert output.shape == (2, 8)

    def test_gradient_flow(self):
        model = QuantumFusionInterference(n_qubits=8)
        text_emb = torch.randn(2, 4)
        image_emb = torch.randn(2, 4)
        output = model(text_emb, image_emb)
        loss = output.sum()
        loss.backward()
        assert model.phase_weights.grad is not None
