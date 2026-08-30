"""
GAP-5 (RESEARCH_GAP.md): Entanglement / circuit metrics as interpretable
cross-modal diagnostics.

Cung cấp:
    - partial_trace(rho, keep, n): trace out subsystems
    - von_neumann_entropy(rho): S(rho) = -Tr(rho log2 rho)
    - mutual_information(rho_full, rho_text, rho_image):
        I(text : image) = S(rho_text) + S(rho_image) - S(rho_full)
    - CrossModalEntanglementAnalyzer:
        * density matrix của state sau PQC fusion (text | image registers)
        * theo dõi MI theo epoch (training dynamics)
        * MI vs prediction-correctness (interpretability)
        * MI collapse under noise (default.mixed + DepolarizingChannel)

Diễn giải: MI cao ⇒ quantum fusion đã trộn (entangle) thông tin text-image;
MI ≈ 0 ⇒ hai register tách biệt (separable), fusion "chết".
"""

import math

import numpy as np
import pennylane as qml
import torch


# ---------------------------------------------------------------------------
# Linear-algebra helpers (pure NumPy, hoạt động trên density matrices)
# ---------------------------------------------------------------------------

def partial_trace(rho, keep, n_qubits):
    """
    Partial trace giữ lại các subsystem trong `keep`.

    Args:
        rho: [2^n, 2^n] density matrix
        keep: list chỉ số subsystem giữ lại (0-indexed)
        n_qubits: tổng số qubit n
    Returns:
        rho_reduced: [2^k, 2^k] với k = len(keep)
    """
    rho = np.asarray(rho)
    keep = sorted(keep)
    dim = 2 ** n_qubits
    if rho.shape != (dim, dim):
        raise ValueError(f"rho shape {rho.shape} != (2^{n_qubits},)² ")

    import string
    letters = iter(string.ascii_letters)
    row_labels = [None] * n_qubits
    col_labels = [None] * n_qubits
    keep_set = set(keep)

    for i in range(n_qubits):
        letter = next(letters)
        if i in keep_set:
            # giữ nguyên: hàng và cột dùng chữ khác nhau
            row_labels[i] = letter
            col_labels[i] = next(letters)
        else:
            # trace out: hàng và cột dùng CÙNG một chữ => được sum qua
            row_labels[i] = letter
            col_labels[i] = letter

    subscripts = "".join(row_labels) + "".join(col_labels)
    out_labels = "".join(row_labels[i] for i in keep) + \
                 "".join(col_labels[i] for i in keep)

    tensor = rho.reshape([2] * (2 * n_qubits))
    reduced = np.einsum(f"{subscripts}->{out_labels}", tensor)
    return reduced.reshape(2 ** len(keep), 2 ** len(keep))


def von_neumann_entropy(rho):
    """S(rho) = -Tr(rho log2 rho) (bits). Clamp eigenvalues âm do numerical error."""
    eigvals = np.linalg.eigvalsh(np.asarray(rho))
    eigvals = np.clip(eigvals.real, 0.0, 1.0)
    nz = eigvals[eigvals > 1e-12]
    return float(-np.sum(nz * np.log2(nz)))


def mutual_information(rho_full, rho_a, rho_b):
    """
    Quantum mutual information giữa partition A|B:
        I(A:B) = S(rho_A) + S(rho_B) - S(rho_AB)
    State toàn phần thuần khiết (không noise) ⇒ S(rho_AB)≈0 và I=2·S(rho_A).
    Với noise (mixed), công thức đầy đủ vẫn đúng.
    """
    return von_neumann_entropy(rho_a) + von_neumann_entropy(rho_b) \
        - von_neumann_entropy(rho_full)


def meyer_wallach_from_rho(rho, n_qubits):
    """
    Global entanglement Meyer–Wallach: Q = (2/n) Σ_j (1 - Tr ρ_j²).
    ρ_j = single-qubit reduced state của qubit j. Q ∈ [0, 1].
    """
    total = 0.0
    for j in range(n_qubits):
        rho_j = partial_trace(rho, [j], n_qubits)
        purity = float(np.real(np.trace(rho_j @ rho_j)))
        total += 1.0 - purity
    return float(2.0 * total / n_qubits)


# ---------------------------------------------------------------------------
# Density-matrix circuits cho cross-modal fusion
# ---------------------------------------------------------------------------

def make_dm_circuit(n_qubits, n_layers, noisy=False, depolarizing_p=0.01):
    """
    Xây dựng QNode trả về density matrix của state sau PQC fusion.
    Text encode trên wires [0, n/2), image trên wires [n/2, n).
    Ansatz giống QuantumFusionTensor (RY/RZ + CNOT ring per layer).

    Args:
        noisy: nếu True dùng default.mixed và chèn DepolarizingChannel(p)
               sau mỗi gate (mô phỏng NISQ, phục vụ MI-vs-noise curve).
    """
    wires = list(range(n_qubits))
    half = n_qubits // 2
    dev_name = "default.mixed" if noisy else "default.qubit"
    dev = qml.device(dev_name, wires=n_qubits)

    def _apply_noise():
        if noisy:
            for w in wires:
                qml.DepolarizingChannel(depolarizing_p, wires=w)

    def _ansatz(weights):
        for l in range(n_layers):
            for i in range(n_qubits):
                qml.RY(weights[l, i], wires=i)
                qml.RZ(weights[l, i + n_qubits], wires=i)
                _apply_noise()
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
                _apply_noise()
            qml.CNOT(wires=[n_qubits - 1, 0])
            _apply_noise()

    @qml.QNode(dev, interface="autograd", diff_method=None)
    def circuit(text_feats, image_feats, weights):
        # data encoding (angle)
        for i in range(half):
            qml.RY(text_feats[i], wires=i)
            qml.RY(image_feats[i], wires=i + half)
        _apply_noise()
        _ansatz(weights)
        return qml.density_matrix(wires=wires)

    return circuit


class CrossModalEntanglementAnalyzer:
    """
    Bộ phân tích entanglement chéo modal trên shared quantum space.

    Usage:
        analyzer = CrossModalEntanglementAnalyzer(n_qubits=8, n_layers=3,
                                                  weights=torch_weights)
        stats = analyzer.analyze_batch(text_embs, image_embs)
        # {'S_text': ..., 'S_image': ..., 'MI': ..., ...}
    """

    def __init__(self, n_qubits=8, n_layers=3, weights=None):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.half = n_qubits // 2
        self.text_wires = list(range(self.half))
        self.image_wires = list(range(self.half, n_qubits))

        if weights is None:
            weights = torch.randn(n_layers, n_qubits * 2) * 0.01
        self.weights = np.asarray(
            weights.detach().cpu().numpy() if isinstance(weights, torch.Tensor)
            else weights
        )

        self._circuit_clean = make_dm_circuit(n_qubits, n_layers, noisy=False)

    # ------------------------------------------------------------------
    def analyze_sample(self, text_half, image_half, weights=None):
        """
        Phân tích 1 mẫu.
        Args:
            text_half:  [n/2] features text (giống input của QFL-Tensor)
            image_half: [n/2] features image
            weights:    [L, 2n] PQC weights (mặc định dùng self.weights)
        Returns:
            dict: S_text, S_image, S_full, MI_text_image, MW_global
        """
        w = self.weights if weights is None else np.asarray(weights)
        rho = np.asarray(self._circuit_clean(
            np.asarray(text_half), np.asarray(image_half), w
        ))
        rho_text = partial_trace(rho, self.text_wires, self.n_qubits)
        rho_image = partial_trace(rho, self.image_wires, self.n_qubits)

        return {
            "S_text": von_neumann_entropy(rho_text),
            "S_image": von_neumann_entropy(rho_image),
            "S_full": von_neumann_entropy(rho),
            "MI_text_image": mutual_information(rho, rho_text, rho_image),
            "MW_global": meyer_wallach_from_rho(rho, self.n_qubits),
        }

    def analyze_batch(self, text_emb, image_emb, weights=None):
        """
        Args:
            text_emb:  [B, n/2] (torch hoặc numpy)
            image_emb: [B, n/2]
        Returns:
            dict các metric (mean ± std) trên batch.
        """
        if isinstance(text_emb, torch.Tensor):
            text_emb = text_emb.detach().cpu().numpy()
        if isinstance(image_emb, torch.Tensor):
            image_emb = image_emb.detach().cpu().numpy()

        per_sample = [
            self.analyze_sample(text_emb[b], image_emb[b], weights)
            for b in range(text_emb.shape[0])
        ]
        stats = {}
        for key in per_sample[0]:
            vals = np.array([s[key] for s in per_sample])
            stats[f"{key}_mean"] = float(vals.mean())
            stats[f"{key}_std"] = float(vals.std())
        stats["per_sample"] = per_sample
        return stats

    # ------------------------------------------------------------------
    def mi_vs_correctness(self, text_embs, image_embs, correct_flags):
        """
        GAP-5 evidence: tương quan MI với prediction correctness.
        Args:
            correct_flags: [N] bool array
        Returns:
            dict: mi_correct_mean, mi_incorrect_mean, delta, point_biserial_r
        """
        stats = self.analyze_batch(text_embs, image_embs)
        per = stats.pop("per_sample")
        correct_flags = np.asarray(correct_flags, dtype=bool)

        mi = np.array([s["MI_text_image"] for s in per])
        mi_c = mi[correct_flags]
        mi_i = mi[~correct_flags]

        result = {
            "mi_correct_mean": float(mi_c.mean()) if len(mi_c) else None,
            "mi_incorrect_mean": float(mi_i.mean()) if len(mi_i) else None,
            "delta": float(mi_c.mean() - mi_i.mean())
            if len(mi_c) and len(mi_i) else None,
        }
        # point-biserial correlation ≈ Pearson(correct_flag, MI)
        if len(set(correct_flags.tolist())) > 1:
            result["point_biserial_r"] = float(np.corrcoef(
                correct_flags.astype(float), mi)[0, 1])
        return result

    # ------------------------------------------------------------------
    def mi_under_noise(self, text_half, image_half, noise_levels=(0.0, 0.005, 0.01, 0.02)):
        """
        GAP-5 evidence: MI collapse dưới depolarizing noise.
        Dùng default.mixed + DepolarizingChannel (true density-matrix evolution).
        Trả về list [(p, MI)].
        """
        results = []
        for p in noise_levels:
            if p == 0.0:
                circ = self._circuit_clean
            else:
                circ = make_dm_circuit(self.n_qubits, self.n_layers,
                                       noisy=True, depolarizing_p=p)
            rho = np.asarray(circ(np.asarray(text_half),
                                  np.asarray(image_half), self.weights))
            results.append((p, von_neumann_entropy(rho),
                            mutual_information(rho,
                                               partial_trace(rho, self.text_wires, self.n_qubits),
                                               partial_trace(rho, self.image_wires, self.n_qubits))))
        return results
