"""
Quantum data encoding strategies.

Supports 3 encoding schemes for experiment E8:
    - "angle":     Angle encoding (RY rotations) — default, fast
    - "amplitude": Amplitude embedding — dense (2^n amplitudes)
    - "iqp":       Instantaneous Quantum Polynomial — feature interactions

Each encoder is a callable: features [n_features] → quantum state on `wires`.
"""

import pennylane as qml
import numpy as np


def angle_encoding(features, wires):
    """
    Angle encoding: each feature → RY rotation on one qubit.
    Requires len(features) <= len(wires).
    """
    n = min(len(features), len(wires))
    for i in range(n):
        qml.RY(features[i], wires=wires[i])


def amplitude_encoding(features, wires):
    """
    Amplitude encoding: encode features into state amplitudes.
    Encodes up to 2^len(wires) features into the state vector.
    """
    qml.AmplitudeEmbedding(
        features=features,
        wires=wires,
        normalize=True,
        pad_with=0.0,
    )


def iqp_encoding(features, wires):
    """
    Instantaneous Quantum Polynomial (IQP) encoding:
      H on all wires → RZ(x_i) → CNOT ladder with RZ(x_i * x_j) terms.
    Captures pairwise feature interactions in the kernel.
    """
    n = min(len(features), len(wires))
    for i in range(n):
        qml.Hadamard(wires=wires[i])
    for i in range(n):
        qml.RZ(features[i], wires=wires[i])
    # Pairwise interaction terms x_i * x_j (diagonal ZZ)
    for i in range(n - 1):
        qml.CNOT(wires=[wires[i], wires[i + 1]])
        qml.RZ(features[i] * features[i + 1], wires=wires[i + 1])
        qml.CNOT(wires=[wires[i], wires[i + 1]])
    # Ring closure for stronger entanglement
    if n > 2:
        qml.CNOT(wires=[wires[n - 1], wires[0]])
        qml.RZ(features[n - 1] * features[0], wires=wires[0])
        qml.CNOT(wires=[wires[n - 1], wires[0]])


# Registry for experiment E8 (encoding comparison)
ENCODINGS = {
    "angle": angle_encoding,
    "amplitude": amplitude_encoding,
    "iqp": iqp_encoding,
}


def get_encoding(name: str):
    """Get an encoding function by name."""
    if name not in ENCODINGS:
        raise ValueError(
            f"Unknown encoding '{name}'. Available: {list(ENCODINGS.keys())}"
        )
    return ENCODINGS[name]


def max_features_for(encoding_name: str, n_wires: int) -> int:
    """Maximum number of classical features an encoding can accept."""
    if encoding_name == "amplitude":
        return 2 ** n_wires
    return n_wires


def estimate_encoding_cost(encoding_name: str, n_features: int) -> dict:
    """Estimate gate counts for each encoding scheme (E8 analysis)."""
    costs = {
        "angle": {"gates": n_features, "depth": n_features, "2q_gates": 0},
        "amplitude": {
            "gates": int(np.log2(max(n_features, 1))),
            "depth": int(np.log2(max(n_features, 1))),
            "2q_gates": max(n_features - 1, 0),
        },
        "iqp": {"gates": 3 * n_features - 1, "depth": 3 * n_features - 1, "2q_gates": 2 * (n_features - 1)},
    }
    return costs.get(encoding_name, {})
