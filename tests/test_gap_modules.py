"""
Tests for GAP-2/3/4/5/6 modules.

Run: pytest tests/test_gap_modules.py -v
"""

import math
import pytest
import numpy as np
import torch
import torch.nn as nn
from types import SimpleNamespace

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ------------------------------------------------------------------
# Shared fixture
# ------------------------------------------------------------------

@pytest.fixture
def flat_config():
    return SimpleNamespace(
        d_model=64, n_qubits=6, n_q_layers=2, fusion_type="tensor",
        image_encoder="resnet18", text_encoder="bert-base-uncased",
        freeze_bert_layers=6, num_sentiment_classes=3,
        vocab_size=500, max_caption_length=20,
        n_heads=4, decoder_layers=1,
        use_quantum_head=False, decoder_type="transformer",
    )


# ==============================================================
# GAP-5: entanglement
# ==============================================================

class TestPartialTrace:
    def test_trace_preserves_diagonal(self):
        from src.quantum.entanglement import partial_trace
        rho = np.eye(8) / 8  # maximally mixed on 3 qubits
        kept = partial_trace(rho, [0, 1], 3)
        assert kept.shape == (4, 4)
        assert abs(np.trace(kept) - 1.0) < 1e-8

    def test_bell_state(self):
        """Bell state: |ψ⟩ = (|00⟩+|11⟩)/√2, reduced state is I/2."""
        from src.quantum.entanglement import partial_trace
        psi = np.zeros(4)
        psi[0] = psi[3] = 1.0 / math.sqrt(2)
        rho = np.outer(psi, psi.conj())
        rho_q0 = partial_trace(rho, [0], 2)
        assert rho_q0.shape == (2, 2)
        assert abs(np.trace(rho_q0) - 1.0) < 1e-8


class TestVonNeumannEntropy:
    def test_pure_state(self):
        from src.quantum.entanglement import von_neumann_entropy
        psi = np.array([1.0, 0.0])
        rho = np.outer(psi, psi)
        assert von_neumann_entropy(rho) < 1e-8

    def test_maximally_mixed(self):
        from src.quantum.entanglement import von_neumann_entropy
        rho = np.eye(4) / 4
        s = von_neumann_entropy(rho)
        assert abs(s - 2.0) < 1e-6  # log2(4)=2


class TestMutualInformation:
    def test_product_state(self):
        from src.quantum.entanglement import mutual_information
        rho = np.eye(4) / 4
        rho_a = rho_b = np.eye(2) / 2
        mi = mutual_information(rho, rho_a, rho_b)
        assert abs(mi) < 1e-6

    def test_non_negative(self):
        from src.quantum.entanglement import mutual_information, partial_trace
        # random density matrix
        H = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
        rho = (H @ H.conj().T)
        rho /= np.trace(rho)
        rho_a = partial_trace(rho, [0], 2)
        rho_b = partial_trace(rho, [1], 2)
        mi = mutual_information(rho, rho_a, rho_b)
        assert mi >= -1e-6  # MI is non-negative (numerical tolerance)


class TestMeyerWallach:
    def test_product_state_zero(self):
        from src.quantum.entanglement import meyer_wallach_from_rho
        rho = np.eye(4) / 4
        assert meyer_wallach_from_rho(rho, 2) < 1e-6


class TestCrossModalEntanglementAnalyzer:
    def test_analyze_sample_keys(self):
        from src.quantum.entanglement import CrossModalEntanglementAnalyzer
        a = CrossModalEntanglementAnalyzer(n_qubits=4, n_layers=1,
                                           weights=torch.randn(1, 8) * 0.01)
        r = a.analyze_sample(np.array([0.1, 0.2]), np.array([0.3, 0.4]))
        assert "MI_text_image_mean" in str(list(r.keys())) or "MI_text_image" in r
        assert r["MI_text_image"] >= -1e-6

    def test_analyze_batch(self):
        from src.quantum.entanglement import CrossModalEntanglementAnalyzer
        a = CrossModalEntanglementAnalyzer(n_qubits=4, n_layers=1,
                                           weights=torch.randn(1, 8) * 0.01)
        stats = a.analyze_batch(np.random.randn(2, 2), np.random.randn(2, 2))
        assert "MI_text_image_mean" in stats
        assert "per_sample" in stats


# ==============================================================
# GAP-6: ansatz metrics
# ==============================================================

class TestAnsatzMetrics:
    def test_meyer_wallach_state(self):
        from src.quantum.ansatz_metrics import meyer_wallach_state
        state = np.zeros(4)
        state[0] = 1.0  # |00⟩ product state
        q = meyer_wallach_state(state, 2)
        assert q < 1e-6

    def test_entangling_capability_range(self):
        from src.quantum.ansatz_metrics import entangling_capability
        ec = entangling_capability("tensor", n_qubits=4, n_layers=1,
                                   n_samples=30, seed=0)
        assert 0.0 <= ec <= 1.0 + 1e-6

    def test_expressibility_returns_float(self):
        from src.quantum.ansatz_metrics import expressibility
        kl = expressibility("tensor", n_qubits=4, n_layers=1,
                            n_samples=30, n_bins=20, seed=0)
        assert isinstance(kl, float)
        assert math.isfinite(kl)


# ==============================================================
# GAP-4: noise_wrapper
# ==============================================================

class TestNoiseWrapper:
    def test_identity_at_zero_noise(self):
        from src.quantum.noise_wrapper import DepolarizingNoiseWrapper
        base = nn.Linear(4, 4, bias=False)
        nn.init.ones_(base.weight)
        w = DepolarizingNoiseWrapper(base, p=0.0, n_noise_sites=1)
        x = torch.randn(2, 4)
        out = w(x)
        expected = base(x)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_scaling_factor(self):
        from src.quantum.noise_wrapper import DepolarizingNoiseWrapper
        base = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            base.weight.fill_(1.0)
        p = 0.1
        w = DepolarizingNoiseWrapper(base, p=p, n_noise_sites=2)
        x = torch.ones(1, 4)
        out = w(x)
        expected_scale = (1 - p) ** 2
        # output ≈ x @ W.T * (1-p)^2 = 4 * expected_scale per entry
        assert torch.allclose(out, x @ base.weight.t() * expected_scale, atol=1e-5)

    def test_restore(self):
        from src.quantum.noise_wrapper import apply_component_noise, restore_component_noise
        from src.models.q_mmf_model import QuantumMultimodalFramework
        cfg = SimpleNamespace(
            d_model=32, n_qubits=4, n_q_layers=1, fusion_type="tensor",
            image_encoder="resnet18", text_encoder="bert-base-uncased",
            freeze_bert_layers=6, num_sentiment_classes=3,
            vocab_size=200, max_caption_length=10,
            n_heads=4, decoder_layers=1,
            use_quantum_head=False, decoder_type="transformer",
        )
        model = QuantumMultimodalFramework(cfg)
        orig = model.shared_quantum.q_fusion
        apply_component_noise(model, "fusion", p=0.1)
        assert model.shared_quantum.q_fusion is not orig
        restore_component_noise(model)
        assert model.shared_quantum.q_fusion is orig


# ==============================================================
# GAP-4: noise_analysis
# ==============================================================

class TestNoiseAnalysis:
    def test_crossover_found(self):
        from src.evaluation.noise_analysis import find_crossover_threshold
        ps = [0.0, 0.01, 0.02]
        q_acc = [0.9, 0.85, 0.7]
        c_acc = [0.88, 0.88, 0.88]  # classical stable
        r = find_crossover_threshold(ps, q_acc, c_acc)
        assert r["crossover_p"] == 0.02
        assert r["index"] == 2

    def test_no_crossover(self):
        from src.evaluation.noise_analysis import find_crossover_threshold
        ps = [0.0, 0.01, 0.02]
        q = [0.9, 0.91, 0.89]
        c = [0.80, 0.80, 0.80]
        r = find_crossover_threshold(ps, q, c)
        assert r["crossover_p"] is None

    def test_degradation_curve(self):
        from src.evaluation.noise_analysis import degradation_curve
        pts = [(0.0, 90.0), (0.01, 80.0)]
        dc = degradation_curve(pts)
        assert abs(dc[1][1] - 100.0 * 10.0 / 90.0) < 1e-6

    def test_per_step_disagreement(self):
        from src.evaluation.noise_analysis import per_step_disagreement
        clean = torch.tensor([[1, 1, 2], [3, 3, 3]])  # [T=3, B=2]
        noisy = torch.tensor([[1, 2, 2], [3, 4, 3]])
        curve = per_step_disagreement(clean, noisy)
        # B=2: step0: 0/2=0 (only sample1 differs); step1: 1/2=0.5 (sample2 new div); step2: 0.5 (same)
        assert curve[0] == 0.5
        assert curve[1] == 0.5


# ==============================================================
# GAP-3: HybridQuantumCaptionDecoder
# ==============================================================

class TestHybridDecoder:
    def test_forward_shape(self):
        from src.decoders.caption_decoder import HybridQuantumCaptionDecoder
        dec = HybridQuantumCaptionDecoder(vocab_size=100, d_model=32,
                                          nhead=4, num_layers=1,
                                          max_seq_len=10, qam_heads=1,
                                          qam_qubits=4)
        tgt = torch.randint(0, 100, (2, 5))
        memory = torch.randn(2, 1, 32)
        logits = dec(tgt, memory)
        assert logits.shape == (2, 5, 100)

    def test_generate(self):
        from src.decoders.caption_decoder import HybridQuantumCaptionDecoder
        dec = HybridQuantumCaptionDecoder(vocab_size=100, d_model=32,
                                          nhead=4, num_layers=1,
                                          max_seq_len=10, qam_heads=1,
                                          qam_qubits=4)
        memory = torch.randn(2, 1, 32)
        gen = dec.generate(memory, max_len=5, bos_token_id=1, eos_token_id=2)
        assert gen.shape[0] == 2


# ==============================================================
# GAP-4: QuantumSentimentHead
# ==============================================================

class TestQuantumSentimentHead:
    def test_forward_shape(self):
        from src.decoders.sentiment_head import QuantumSentimentHead
        head = QuantumSentimentHead(d_model=32, num_classes=3,
                                    n_qubits=4, n_layers=1)
        x = torch.randn(2, 32)
        logits = head(x)
        assert logits.shape == (2, 3)


# ==============================================================
# GAP-2: gradient_conflict
# ==============================================================

class TestGradientConflict:
    def test_cosine_sim(self):
        from src.training.gradient_conflict import cosine_similarity
        a = torch.tensor([1.0, 0.0])
        b = torch.tensor([0.0, 1.0])
        assert abs(cosine_similarity(a, b)) < 1e-6
        assert abs(cosine_similarity(a, a) - 1.0) < 1e-6
        assert cosine_similarity(-a, a) < -0.9

    def test_shared_param_names(self):
        from src.training.gradient_conflict import shared_param_names
        from src.models.q_mmf_model import QuantumMultimodalFramework
        cfg = SimpleNamespace(
            d_model=32, n_qubits=4, n_q_layers=1, fusion_type="tensor",
            image_encoder="resnet18", text_encoder="bert-base-uncased",
            freeze_bert_layers=6, num_sentiment_classes=3,
            vocab_size=200, max_caption_length=10,
            n_heads=4, decoder_layers=1,
            use_quantum_head=False, decoder_type="transformer",
        )
        model = QuantumMultimodalFramework(cfg)
        names = shared_param_names(model)
        assert any("q_fusion" in n for n in names)


# ==============================================================
# GAP-4: q_mmf_model config wiring
# ==============================================================

class TestModelConfigWiring:
    def test_hybrid_decoder_instantiation(self):
        from src.models.q_mmf_model import QuantumMultimodalFramework
        from src.decoders.caption_decoder import HybridQuantumCaptionDecoder
        cfg = SimpleNamespace(
            d_model=32, n_qubits=4, n_q_layers=1, fusion_type="tensor",
            image_encoder="resnet18", text_encoder="bert-base-uncased",
            freeze_bert_layers=6, num_sentiment_classes=3,
            vocab_size=200, max_caption_length=10,
            n_heads=4, decoder_layers=1,
            decoder_type="hybrid_quantum",
            qam_heads=1, qam_qubits=4,
        )
        model = QuantumMultimodalFramework(cfg)
        assert isinstance(model.caption_decoder, HybridQuantumCaptionDecoder)

    def test_quantum_head_instantiation(self):
        from src.models.q_mmf_model import QuantumMultimodalFramework
        from src.decoders.sentiment_head import QuantumSentimentHead
        cfg = SimpleNamespace(
            d_model=32, n_qubits=4, n_q_layers=1, fusion_type="tensor",
            image_encoder="resnet18", text_encoder="bert-base-uncased",
            freeze_bert_layers=6, num_sentiment_classes=3,
            vocab_size=200, max_caption_length=10,
            n_heads=4, decoder_layers=1,
            use_quantum_head=True,
            head_qubits=4, head_q_layers=1,
        )
        model = QuantumMultimodalFramework(cfg)
        assert isinstance(model.sentiment_head, QuantumSentimentHead)
