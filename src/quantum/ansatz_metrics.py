"""
GAP-6 / §2.7 (RESEARCH_GAP.md): Ansatz profiling metrics.

    - expressibility(): KL divergence giữa fidelity distribution của ansatz
      và Haar-random distribution (Sim, Johnson & Aspuru-Guzik, Adv. Quantum
      Technol. 2019). KL thấp ⇒ ansatz "biểu đạt" tốt Hilbert space.
    - meyer_wallach_state(): global entanglement từ statevector.
    - entangling_capability(): trung bình Q(θ) trên random θ.

Được dùng trong E10 (run_ablation.py) để profile QFL-Tensor/QFL-Interference
ansatz theo depth/qubits — bổ sung góc nhìn "why quantum helps" ngoài accuracy.
"""

import itertools
import math

import numpy as np
import pennylane as qml


# ---------------------------------------------------------------------------
# Ansatz definitions (trainable phần của fusion circuits, không data encoding)
# ---------------------------------------------------------------------------

def qfl_tensor_ansatz(weights, wires):
    """Ansatz của QuantumFusionTensor: per-layer RY/RZ + CNOT ring."""
    n = len(wires)
    n_layers = weights.shape[0]
    for l in range(n_layers):
        for i in range(n):
            qml.RY(weights[l, i], wires=i)
            qml.RZ(weights[l, i + n], wires=i)
        for i in range(n - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[n - 1, 0])


def qfl_interference_ansatz(weights, wires):
    """Ansatz của QuantumFusionInterference: RY/RZ + all-to-all CNOT."""
    n = len(wires)
    n_layers = weights.shape[0]
    for l in range(n_layers):
        for i in range(n):
            qml.RY(weights[l, i], wires=i)
            qml.RZ(weights[l, i + n], wires=i)
        for i in range(n):
            for j in range(i + 1, n):
                qml.CNOT(wires=[i, j])
                qml.RZ(weights[l, (i * n + j) % (2 * n)], wires=j)
                qml.CNOT(wires=[i, j])


ANSATZ_REGISTRY = {
    "tensor": qfl_tensor_ansatz,
    "interference": qfl_interference_ansatz,
}


def _make_state_qnode(n_qubits, ansatz_fn, weight_shape):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.QNode(dev, interface="autograd", diff_method=None)
    def state_qnode(flat_params):
        weights = np.asarray(flat_params).reshape(weight_shape)
        ansatz_fn(weights, list(range(n_qubits)))
        return qml.state()

    return state_qnode


# ---------------------------------------------------------------------------
# Meyer-Wallach global entanglement
# ---------------------------------------------------------------------------

def meyer_wallach_state(state, n_qubits):
    """
    Q(|ψ⟩) = (2/n) Σ_j (1 - Tr ρ_j²) tính trực tiếp từ statevector
    (nhanh hơn đường density-matrix).
    """
    psi = np.asarray(state)
    # reshape: [2]*n (row subsystems); ρ_j qua trace các chỉ số còn lại
    tensor = psi.reshape([2] * n_qubits)
    total = 0.0
    for j in range(n_qubits):
        # reduced ρ_j elements: ρ_j[a,b] = Σ_{rest} ψ[rest,a] ψ*[rest,b]
        other_axes = tuple(i for i in range(n_qubits) if i != j)
        contracted = np.tensordot(tensor, tensor.conj(),
                                  axes=(other_axes, other_axes))
        rho_j = contracted  # [2, 2]
        purity = float(np.real(np.trace(rho_j @ rho_j)))
        total += 1.0 - purity
    return float(2.0 * total / n_qubits)


def entangling_capability(ansatz="tensor", n_qubits=8, n_layers=3,
                          n_samples=200, seed=0):
    """
    Trung bình Meyer–Wallach trên n_samples random parameter settings.
    Trả về scalar ∈ [0, 1].
    """
    rng = np.random.default_rng(seed)
    fn = ANSATZ_REGISTRY.get(ansatz)
    if fn is None:
        raise ValueError(f"Unknown ansatz: {ansatz}")
    weight_shape = (n_layers, 2 * n_qubits)
    state_qnode = _make_state_qnode(n_qubits, fn, weight_shape)

    qs = []
    for _ in range(n_samples):
        params = rng.uniform(-math.pi, math.pi, size=weight_shape.size)
        state = state_qnode(params)
        qs.append(meyer_wallach_state(state, n_qubits))
    return float(np.mean(qs))


# ---------------------------------------------------------------------------
# Expressibility (Sim et al. 2019)
# ---------------------------------------------------------------------------

def _haar_fidelity_pdf(x, dim):
    """P_F(x) = (N-1)(1-x)^{N-2} với N = dim Hilbert space."""
    return (dim - 1) * np.power(1.0 - x, dim - 2)


def expressibility(ansatz="tensor", n_qubits=8, n_layers=3,
                   n_samples=300, n_bins=60, seed=0):
    """
    KL( F_circuit || F_Haar ) giữa histogram fidelity của ansatz và
    phân phối Haar-random. Giá trị NHỎ hơn ⇒ biểu đạt tốt hơn.

    Args:
        n_samples: số state random (cặp pair được lấy toàn bộ C(n,2))
        n_bins: số bin histogram trên [0, 1]
    Returns:
        float: KL divergence (nats)
    """
    rng = np.random.default_rng(seed)
    fn = ANSATZ_REGISTRY.get(ansatz)
    if fn is None:
        raise ValueError(f"Unknown ansatz: {ansatz}")
    weight_shape = (n_layers, 2 * n_qubits)
    state_qnode = _make_state_qnode(n_qubits, fn, weight_shape)

    states = []
    for _ in range(n_samples):
        params = rng.uniform(-math.pi, math.pi, size=weight_shape.size)
        states.append(np.asarray(state_qnode(params)))

    fidelities = []
    for i, j in itertools.combinations(range(len(states)), 2):
        overlap = np.vdot(states[i].conj(), states[j])
        fidelities.append(abs(overlap) ** 2)
    fidelities = np.array(fidelities)

    hist_ckt, edges = np.histogram(fidelities, bins=n_bins, range=(0.0, 1.0),
                                   density=True)
    centers = (edges[:-1] + edges[1:]) / 2.0

    dim = 2 ** n_qubits
    pdf_haar = _haar_fidelity_pdf(centers, dim)

    mask = hist_ckt > 0
    kl = float(np.sum(
        hist_ckt[mask] * np.log(hist_ckt[mask] / pdf_haar[mask])
    ))
    return kl


def profile_ansatz(ansatz_names=("tensor", "interference"),
                   qubits_list=(4, 8), depths=(1, 2, 3, 5),
                   n_samples=200, seed=0):
    """
    Profile bảng ansatz cho E10/GAP-6: {ansatz, n_qubits, depth} →
    {expressibility_KL, entangling_capability}.
    """
    rows = []
    for name in ansatz_names:
        for n in qubits_list:
            for depth in depths:
                row = {
                    "ansatz": name,
                    "n_qubits": n,
                    "depth": depth,
                    "expressibility_kl": None,
                    "entangling_capability": None,
                }
                try:
                    row["expressibility_kl"] = expressibility(
                        name, n, depth, n_samples=min(n_samples, 150), seed=seed)
                    row["entangling_capability"] = entangling_capability(
                        name, n, depth, n_samples=min(n_samples, 100), seed=seed)
                except Exception as e:  # memory guard với n lớn
                    row["error"] = str(e)
                rows.append(row)
    return rows
