"""Sentiment classification heads.

GAP-4 (RESEARCH_GAP.md): QuantumSentimentHead đặt PQC ở classification head
để so sánh độ nhạy noise giữa 3 vị trí: fusion-PQC / decoder-attention-PQC /
classifier-head-PQC.
"""

import torch
import torch.nn as nn
import pennylane as qml


class SentimentHead(nn.Module):
    def __init__(self, d_model=256, num_classes=3, dropout=0.2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.head(x)


class QuantumSentimentHead(nn.Module):
    """
    GAP-4: Classification head dùng PQC.
    fused [B, d_model] → proj → n_qubits → angle encode → L entangler layers
    → <Z> per qubit → Linear(n_qubits → num_classes).
    """

    def __init__(self, d_model=256, num_classes=3, n_qubits=4, n_layers=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        self.in_proj = nn.Sequential(
            nn.Linear(d_model, n_qubits),
            nn.Tanh(),  # giới hạn [-1, 1] cho angle encoding
        )
        self.weights = nn.Parameter(torch.randn(n_layers, n_qubits * 2) * 0.01)
        self.out = nn.Linear(n_qubits, num_classes)

        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.QNode(dev, interface="torch", diff_method="backprop")
        def _circuit(x, w):
            for i in range(n_qubits):
                qml.RY(x[i], wires=i)
            for l in range(n_layers):
                for i in range(n_qubits):
                    qml.RY(w[l, i], wires=i)
                    qml.RZ(w[l, i + n_qubits], wires=i)
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[n_qubits - 1, 0])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self._circuit = _circuit

    def forward(self, x):
        x = self.in_proj(x)  # [B, n_qubits]
        outs = []
        for b in range(x.shape[0]):
            outs.append(torch.stack(self._circuit(x[b], self.weights)))
        z = torch.stack(outs)  # [B, n_qubits]
        return self.out(z)
