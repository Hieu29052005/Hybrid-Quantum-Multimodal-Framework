"""
GAP-4 (RESEARCH_GAP.md): Component-wise NISQ noise injection & mitigation.

Chiến lược mô phỏng noise nhanh (global depolarizing approximation):
    Với depolarizing channel  E_p(ρ) = (1-p)·ρ + p·I/d , mọi expectation
    giá trị Pauli-Z biến đổi TUYẾN TÍNH:  <Z>_noisy = (1-p)·<Z>_clean.
    Kếp lại k lần ứng dụng độc lập ⇒ nhân tử (1-p)^k.
    → Cho phép đánh giá noise trên FULL model mà không cần default.mixed
      (chậm ~100×). Đường default.mixed "thật" có sẵn trong entanglement.py
      để verify trên mạch nhỏ.

Cung cấp:
    - DepolarizingNoiseWrapper: bọc bất kỳ module quantum nào, scale output
    - apply_component_noise(model, component, p): bọc đúng vị trí
        "fusion"           → shared_quantum.q_fusion
        "decoder_attention"→ mỗi QuantumMultiHeadAttention trong caption_decoder
        "quantum_head"     → sentiment_head (nếu là QuantumSentimentHead)
      restore_component_noise(model): trả model về trạng thái sạch
    - ResidualQuantumMitigation: skip-connection cổ điển song song
      (QMLSC-style): y = q_noisy + α · W_classical(x)
"""

import torch
import torch.nn as nn

from ..decoders.sentiment_head import QuantumSentimentHead


class DepolarizingNoiseWrapper(nn.Module):
    """
    Bọc module quantum; forward = base(*args) * (1 - p_eff), với tuỳ chọn
    readout Gaussian noise. `n_noise_sites` mô phỏng số điểm lỗi tích luỹ.
    """

    def __init__(self, base_module, p=0.0, n_noise_sites=1, readout_std=0.0):
        super().__init__()
        self.base = base_module
        self.p = float(p)
        self.n_noise_sites = int(n_noise_sites)
        self.readout_std = float(readout_std)

    def forward(self, *args, **kwargs):
        out = self.base(*args, **kwargs)
        scale = (1.0 - self.p) ** self.n_noise_sites
        out = out * scale
        if self.readout_std > 0:
            # noise vật lý áp cả lúc train lẫn eval (không phải augmentation)
            out = out + torch.randn_like(out) * self.readout_std
        return out


# ---------------------------------------------------------------------------
# Component targeting
# ---------------------------------------------------------------------------

def _iter_qam_modules(module):
    """Sinh tất cả QuantumMultiHeadAttention con (cho decoder hybrid)."""
    from ..quantum.quantum_attention import QuantumMultiHeadAttention
    for m in module.modules():
        if isinstance(m, QuantumMultiHeadAttention):
            yield m


def apply_component_noise(model, component="fusion", p=0.01,
                          n_noise_sites=None, readout_std=0.0):
    """
    Bọc noise vào ĐÚNG thành phần PQC của model (in-place).

    Args:
        component: "fusion" | "decoder_attention" | "quantum_head"
        p: depolarizing probability per site
        n_noise_sites: hệ số k trong (1-p)^k; mặc định theo component:
            fusion → n_layers (mỗi layer 1 vòng noise),
            decoder_attention → 1,
            quantum_head → head_q_layers.
    Returns:
        list các wrapper đã gắn (để restore).
    """
    wrappers = []
    if component == "fusion":
        base = model.shared_quantum.q_fusion
        k = n_noise_sites or getattr(base, "n_layers", 3)
        wrapped = DepolarizingNoiseWrapper(base, p=p, n_noise_sites=k,
                                           readout_std=readout_std)
        model.shared_quantum.q_fusion = wrapped
        wrappers.append(wrapped)
    elif component == "decoder_attention":
        k = n_noise_sites or 1
        for qam in list(_iter_qam_modules(model.caption_decoder)):
            wrapped = DepolarizingNoiseWrapper(qam, p=p, n_noise_sites=k,
                                               readout_std=readout_std)
            _replace_module(model.caption_decoder, qam, wrapped)
            wrappers.append(wrapped)
    elif component == "quantum_head":
        if not isinstance(model.sentiment_head, QuantumSentimentHead):
            raise ValueError("use_quantum_head=False: không có PQC ở head")
        k = n_noise_sites or model.sentiment_head.n_layers
        # head chạy QNode trực tiếp — bọc ở tầng vector <Z> cho nhất quán
        wrapped_head = _NoisyQuantumHead(model.sentiment_head, p=p,
                                         n_noise_sites=k,
                                         readout_std=readout_std)
        model.sentiment_head = wrapped_head
        wrappers.append(wrapped_head)
    else:
        raise ValueError(f"Unknown component: {component}")
    return wrappers


def restore_component_noise(model):
    """Gỡ mọi wrapper, trả model về trạng thái sạch."""
    # fusion
    fq = model.shared_quantum.q_fusion
    if isinstance(fq, DepolarizingNoiseWrapper):
        model.shared_quantum.q_fusion = fq.base
    # decoder attention
    dec = model.caption_decoder
    if hasattr(dec, "layers"):
        for layer in dec.layers:
            qa = getattr(layer, "q_cross_attn", None)
            if qa is not None and isinstance(qa, DepolarizingNoiseWrapper):
                layer.q_cross_attn = qa.base
    # quantum head
    sh = model.sentiment_head
    if isinstance(sh, _NoisyQuantumHead):
        model.sentiment_head = sh.base
    return model


def _replace_module(root, target, replacement):
    """Thay module `target` bằng `replacement` tại vị trí đầu tiên tìm thấy."""
    for name, child in root.named_children():
        if child is target:
            setattr(root, name, replacement)
            return True
        if _replace_module(child, target, replacement):
            return True
    return False


class _NoisyQuantumHead(nn.Module):
    """Bọc QuantumSentimentHead: áp depolarizing lên vector <Z> trước Linear cuối."""

    def __init__(self, base_head, p=0.0, n_noise_sites=1, readout_std=0.0):
        super().__init__()
        self.base = base_head
        self.p = float(p)
        self.k = int(n_noise_sites)
        self.readout_std = float(readout_std)

    def forward(self, x):
        z = self.base.in_proj(x)
        outs = []
        for b in range(z.shape[0]):
            outs.append(torch.stack(self.base._circuit(z[b], self.base.weights)))
        zv = torch.stack(outs)
        zv = zv * ((1.0 - self.p) ** self.k)
        if self.readout_std > 0:
            zv = zv + torch.randn_like(zv) * self.readout_std
        return self.base.out(zv)


# ---------------------------------------------------------------------------
# Residual mitigation (QMLSC-style skip connection)
# ---------------------------------------------------------------------------

class ResidualQuantumMitigation(nn.Module):
    """
    Mitigation candidate (RESEARCH_GAP.md §2.5 / li2025qmlsc):
        y = q_noisy + α · W_cl(concat(text_half, image_half))
    Nhánh cổ điển W_cl học được từ dữ liệu sạch, đóng vai trò skip-path
    khi PQC bị nhiễu. α điều khiển mức trộn (α=0 ⇒ thuần quantum).
    """

    def __init__(self, base_fusion, n_qubits=8, alpha=0.25):
        super().__init__()
        self.base = base_fusion
        self.alpha = alpha
        self.classical_skip = nn.Linear(n_qubits, n_qubits)

    def forward(self, text_half, image_half):
        q_out = self.base(text_half, image_half)
        skip = self.classical_skip(torch.cat([text_half, image_half], dim=-1))
        return q_out + self.alpha * skip
