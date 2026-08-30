"""
NISQ noise simulation for quantum circuits.
Adds realistic noise models to QNodes.
"""

import pennylane as qml


def add_depolarizing_noise(prob=0.01):
    """
    Add per-gate depolarizing noise channel.
    Simulates NISQ hardware noise.
    """
    def noise_fn():
        for wire in range(qml.active_context().num_wires):
            qml.DepolarizingChannel(prob, wires=wire)
    return noise_fn


def create_noisy_device(n_qubits=8, noise_prob=0.01):
    """Create PennyLane device with depolarizing noise."""
    dev = qml.device(
        "default.mixed",
        wires=n_qubits,
        noise=add_depolarizing_noise(noise_prob),
    )
    return dev


def compute_fidelity(rho, sigma):
    """Compute fidelity between two quantum states (density matrices)."""
    import torch
    if rho.dim() == 2:
        return torch.abs(torch.trace(rho @ sigma)) ** 2
    return torch.tensor(0.0)
