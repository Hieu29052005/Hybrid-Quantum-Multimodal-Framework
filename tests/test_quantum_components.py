"""
Smoke tests for quantum components (Week 3 milestone).

Verifies:
    - Output shapes for all 3 QFL variants
    - Gradient flow through PQC parameters (backprop via torch)
    - Vectorized batched execution correctness & speed vs per-sample loop
    - QuantumMultiHeadAttention shapes, masking, gradients
    - SharedQuantumSpace forward for both tasks + param counting
    - Noise utilities (entropy on Bell state, noise sweep)

Run:  pytest tests/test_quantum_components.py -v
      python -m tests.test_quantum_components   (standalone)
"""

import math
import time

import numpy as np
import pennylane as qml
import pytest
import torch

from src.quantum import (
    QuantumFusionTensor,
    QuantumFusionAttention,
    QuantumFusionInterference,
    QuantumMultiHeadAttention,
    SharedQuantumSpace,
    entanglement_entropy,
    cross_modal_mutual_information,
    run_noise_sweep,
    apply_encoding,
)
from src.quantum.noise import make_noisy_fusion_circuit


BATCH = 8


def _rand_feats(batch, dim):
    g = torch.Generator().manual_seed(0)
    return torch.randn(batch, dim, generator=g)


# ----------------------------------------------------------------------
# Fusion variants
# ----------------------------------------------------------------------

@pytest.mark.parametrize("fusion_cls,kw", [
    (QuantumFusionTensor, dict(n_qubits=8, n_layers=2)),
    (QuantumFusionAttention, dict(n_qubits=8, d_proj=32)),
    (QuantumFusionInterference, dict(n_qubits=8)),
])
def test_fusion_shapes_and_gradients(fusion_cls, kw):
    torch.manual_seed(0)
    d_proj = kw.get("d_proj", 4)
    t = _rand_feats(BATCH, d_proj)
    i = _rand_feats(BATCH, d_proj)

    model = fusion_cls(**kw)
    out = model(t, i)

    assert out.shape == (BATCH, 8), f"{fusion_cls.__name__} shape {out.shape}"
    assert torch.isfinite(out).all()

    loss = out.sum()
    loss.backward()
    grads_ok = all(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in model.parameters() if p.requires_grad
    )
    assert grads_ok, f"{fusion_cls.__name__}: missing/zero gradient"


def test_fusion_batch_invariance():
    """Batching must not change individual results."""
    torch.manual_seed(1)
    model = QuantumFusionTensor(n_qubits=8, n_layers=2)
    t = _rand_feats(5, 4)
    i = _rand_feats(5, 4)

    batched = model(t, i)
    single = torch.cat([model(t[j:j+1], i[j:j+1]) for j in range(5)], dim=0)
    assert torch.allclose(batched, single, atol=1e-6)


def test_tensor_vectorized_is_fast():
    """Vectorized path should beat a naive per-sample loop comfortably."""
    torch.manual_seed(0)
    big = 64
    model = QuantumFusionTensor(n_qubits=8, n_layers=2)
    t, i = _rand_feats(big, 4), _rand_feats(big, 4)

    start = time.perf_counter()
    model(t, i)
    vectorized_s = time.perf_counter() - start

    # Per-sample loop using the same underlying circuit
    start = time.perf_counter()
    for b in range(8):  # subset: proportional cost check
        model(t[b:b+1], i[b:b+1])
    loop_8_s = time.perf_counter() - start
    est_loop_s = loop_8_s * (big / 8)

    assert vectorized_s < est_loop_s / 5.0, (
        f"vectorized {vectorized_s:.3f}s vs est.loop {est_loop_s:.3f}s"
    )


def test_noisy_fusion_runs():
    torch.manual_seed(0)
    model = QuantumFusionTensor(n_qubits=8, n_layers=2, noise_prob=0.01)
    out = model(_rand_feats(4, 4), _rand_feats(4, 4))
    assert out.shape == (4, 8) and torch.isfinite(out).all()


def test_circuit_draws():
    model = QuantumFusionTensor(n_qubits=8, n_layers=1)
    ascii_art = model.draw()
    assert "RY" in ascii_art and len(ascii_art) > 50


# ----------------------------------------------------------------------
# Quantum attention
# ----------------------------------------------------------------------

def test_quantum_attention_shapes_and_mask():
    torch.manual_seed(0)
    B, Lq, Lk, D, H = 2, 6, 4, 32, 2
    attn = QuantumMultiHeadAttention(d_model=D, n_heads=H, n_qubits=4,
                                     chunk_size=512)
    q = _rand_feats(B * Lq, D).view(B, Lq, D)
    k = _rand_feats(B * Lk, D).view(B, Lk, D)

    out = attn(q, k, k)
    assert out.shape == (B, Lq, D)
    assert torch.isfinite(out).all()

    # Padding mask: last key position masked out -> output changes
    mask = torch.ones(B, Lk, dtype=torch.bool)
    mask[:, -1] = False
    out_masked = attn(q, k, k, mask=mask)
    assert not torch.allclose(out, out_masked)


def test_quantum_attention_gradient():
    torch.manual_seed(0)
    attn = QuantumMultiHeadAttention(d_model=16, n_heads=1, n_qubits=4,
                                     chunk_size=256)
    q = _rand_feats(2 * 3, 16).view(2, 3, 16)
    k = _rand_feats(2 * 2, 16).view(2, 2, 16)
    out = attn(q, k, k).sum()
    out.backward()
    assert attn.kernel_weights.grad is not None
    assert torch.isfinite(attn.kernel_weights.grad).all()


# ----------------------------------------------------------------------
# Shared space
# ----------------------------------------------------------------------

@pytest.mark.parametrize("ftype", ["tensor", "attention", "interference"])
def test_shared_space_tasks(ftype):
    torch.manual_seed(0)
    space = SharedQuantumSpace(n_qubits=8, n_layers=2, d_model=32,
                               fusion_type=ftype)
    t = _rand_feats(BATCH, 32)
    i = _rand_feats(BATCH, 32)

    msa_out = space(t, i, task="msa")
    cap_out = space(t, i, task="caption")

    assert msa_out.shape == cap_out.shape == (BATCH, 32)
    assert not torch.allclose(msa_out, cap_out)  # adapter differs

    (msa_out.sum() + cap_out.sum()).backward()
    qcount = space.quantum_parameter_count()
    assert qcount > 0, "quantum params should be counted"
    if ftype in ("tensor", "interference"):
        # tensor: [layers, 16]; interference: phases[8] + alpha + beta
        assert qcount >= 8


# ----------------------------------------------------------------------
# Noise & diagnostics utilities
# ----------------------------------------------------------------------

def test_bell_state_entropy():
    """|Phi+> = (|00>+|11>)/sqrt2 has exactly 1 bit of entanglement."""
    bell = np.array([1, 0, 0, 1], dtype=complex) / math.sqrt(2)
    s = entanglement_entropy(bell, n_qubits=2)
    mi = cross_modal_mutual_information(bell, n_qubits=2)
    assert abs(s - 1.0) < 1e-6
    assert abs(mi - 2.0) < 1e-6

    prod = np.array([1, 0, 0, 0], dtype=complex)
    assert entanglement_entropy(prod, 2) < 1e-9


def test_noisy_circuit_drifts_from_clean():
    circuit = make_noisy_fusion_circuit(n_qubits=8, n_layers=2, noise_prob=0.05)
    clean = make_noisy_fusion_circuit(n_qubits=8, n_layers=2, noise_prob=0.0)
    rng = np.random.default_rng(3)
    t = torch.tensor(rng.normal(size=(4, 4)), dtype=torch.float32)
    i = torch.tensor(rng.normal(size=(4, 4)), dtype=torch.float32)
    w = torch.tensor(rng.normal(scale=0.05, size=(2, 16)), dtype=torch.float32)

    out_n = np.asarray(circuit(t, i, w)).ravel()
    out_c = np.asarray(clean(t, i, w)).ravel()
    assert np.abs(out_n - out_c).mean() > 1e-4, "noise had no effect?"


def test_noise_sweep_structure():
    res = run_noise_sweep(n_qubits=8, n_layers=2,
                          noise_levels=(0.0, 0.01), batch_size=4)
    assert set(res.keys()) == {0.0, 0.01}
    assert "fidelity_vs_clean" in res[0.0]
    assert 0.0 < res[0.01]["shadow_similarity"] <= 1.0


def test_encodings_apply():
    dev = qml.device("default.qubit", wires=4)

    @qml.qnode(dev, interface="torch", diff_method="best")
    def circ(x):
        apply_encoding("double_angle", x, wires=[0, 1, 2, 3])
        return qml.expval(qml.PauliZ(0))

    x = _rand_feats(3, 4)
    out = circ(x)
    assert out.shape == (3,) or tuple(np.shape(out)) == (3,)


# ----------------------------------------------------------------------
# Standalone runner
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
