"""Reusable PQC building blocks for quantum layers."""

import pennylane as qml


def angle_encoding(features, wires):
    """Angle encoding: each feature → RY rotation on one qubit."""
    for i, f in enumerate(features):
        qml.RY(f, wires=wires[i])


def amplitude_encoding(features, wires):
    """Amplitude encoding: encode features into amplitudes."""
    qml.AmplitudeEmbedding(features, wires=wires, normalize=True)


def iqp_encoding(features, wires):
    """Instantaneous Quantum Polynomial (IQP) encoding."""
    for _ in features:
        qml.Hadamard(wires=wires[0])
    for i, f in enumerate(features):
        qml.RZ(f, wires=wires[i])
    for i in range(len(features) - 1):
        qml.CNOT(wires=[wires[i], wires[i + 1]])


def strongly_entangling_layer(weights, wires, n_layers):
    """Strongly entangling layer from PennyLane templates."""
    for l in range(n_layers):
        for i, w in enumerate(wires):
            qml.RX(weights[l, i, 0], wires=w)
            qml.RY(weights[l, i, 1], wires=w)
            qml.RZ(weights[l, i, 2], wires=w)
        for i in range(len(wires) - 1):
            qml.CNOT(wires=[wires[i], wires[i + 1]])
        if len(wires) > 2:
            qml.CNOT(wires=[wires[-1], wires[0]])


def basic_entangler_layer(weights, wires):
    """Basic entangling layer: RY + CNOT chain."""
    for i, w in enumerate(wires):
        qml.RY(weights[i], wires=w)
    for i in range(len(wires) - 1):
        qml.CNOT(wires=[wires[i], wires[i + 1]])
