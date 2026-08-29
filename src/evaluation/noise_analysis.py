"""
GAP-4 (RESEARCH_GAP.md): Noise-robustness analysis cho multimodal fusion
+ generation.

Cung cấp:
    - per_step_token_entropy(): entropy phân bố token ở MỖI bước decode
      dưới noise → "per-step token entropy in decoding under noise"
    - divergence_onset(): bước đầu tiên chuỗi noisy lệch khỏi clean
      → "error propagation through autoregressive decode steps"
    - per_step_disagreement(): % sample diverged tại mỗi vị trí t
    - find_crossover_threshold(): mức noise đầu tiên mà quantum < classical
    - summarize_noise_sweep(): gom kết quả sweep thành bảng JSON-ready

Lưu ý: các hàm decode thủ công tái dùng forward() của decoder nên hoạt động
với cả TransformerCaptionDecoder lẫn HybridQuantumCaptionDecoder.
"""

import math

import torch


# ---------------------------------------------------------------------------
# Per-step token entropy under noise
# ---------------------------------------------------------------------------

def _token_entropy(logits):
    """Entropy của phân bố softmax tại 1 vị trí: [B, V] → [B]."""
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    return -(probs * log_probs).sum(dim=-1)


@torch.no_grad()
def per_step_token_entropy(model, images, text_emb=None, device="cpu",
                           max_len=50, bos_token_id=2, eos_token_id=3):
    """
    Greedy-decode từng bước và ghi entropy + token được chọn mỗi step.

    Args:
        model: QuantumMultimodalFramework (đã có encoder/quantum/decoder)
        images: [B, 3, H, W]
    Returns:
        dict: {
            'entropies':  [T, B]  entropy mỗi bước,
            'token_ids':  [T, B],
            'finished':   [B]     đã dừng trước max_len chưa,
        }
        T = số bước thực tế (dừng khi tất cả hit EOS hoặc max_len).
    """
    model.eval()
    images = images.to(device)
    i_emb = model.image_encoder(images)
    if text_emb is None:
        text_emb = torch.zeros_like(i_emb)
    q_feat = model.shared_quantum(text_emb, i_emb, task="caption")
    memory = q_feat.unsqueeze(1)

    B = images.shape[0]
    generated = torch.full((B, 1), bos_token_id, dtype=torch.long, device=device)

    entropies, token_ids = [], []
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(max_len - 1):
        logits = model.caption_decoder(generated, memory)[:, -1, :]
        entropies.append(_token_entropy(logits).cpu())
        next_token = logits.argmax(dim=-1)
        token_ids.append(next_token.cpu())
        finished = finished | (next_token == eos_token_id)
        generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
        if finished.all():
            break

    return {
        "entropies": torch.stack(entropies),          # [T, B]
        "token_ids": torch.stack(token_ids),          # [T, B]
        "finished": finished.cpu(),
    }


# ---------------------------------------------------------------------------
# Error propagation through autoregressive decoding
# ---------------------------------------------------------------------------

def per_step_disagreement(clean_tokens, noisy_tokens, eos_token_id=3):
    """
    % sample mà chuỗi noisy khác clean tại vị trí t (cumulative-first-divergence).

    Args:
        clean_tokens / noisy_tokens: [T, B] từ per_step_token_entropy
    Returns:
        list[float] độ dài T
    """
    T = min(clean_tokens.shape[0], noisy_tokens.shape[0])
    B = clean_tokens.shape[1]
    diverged = torch.zeros(B, dtype=torch.bool)
    curve = []
    for t in range(T):
        diverged = diverged | (clean_tokens[t] != noisy_tokens[t])
        curve.append(float(diverged.float().mean()))
    return curve


def divergence_onset(clean_tokens, noisy_tokens):
    """
    Bước đầu tiên mỗi sample bắt đầu lệch khỏi clean (None nếu không lệch).
    Trả về stats: mean/min/max onset + % samples ever diverged.
    """
    T, B = clean_tokens.shape
    onsets = []
    for b in range(B):
        diff = (clean_tokens[:, b] != noisy_tokens[:, b]).nonzero()
        onsets.append(int(diff[0]) if len(diff) else None)

    valid = [o for o in onsets if o is not None]
    return {
        "onset_per_sample": onsets,
        "n_diverged": len(valid),
        "frac_diverged": len(valid) / B,
        "mean_onset": float(sum(valid) / len(valid)) if valid else None,
        "min_onset": min(valid) if valid else None,
        "max_onset": max(valid) if valid else None,
    }


# ---------------------------------------------------------------------------
# Crossover quantum vs classical under noise
# ---------------------------------------------------------------------------

def find_crossover_threshold(noise_levels, acc_quantum, acc_classical,
                             strictly_worse=True):
    """
    GAP-4 evidence: "crossover noise threshold where quantum < classical".

    Args:
        noise_levels: list p tăng dần (ví dụ [0, 0.005, 0.01, 0.02])
        acc_quantum / acc_classical: metric tương ứng mỗi level
    Returns:
        dict: {'crossover_p': float|None, 'index': int|None,
               'gaps': list (quantum − classical mỗi level)}
    """
    gaps = [float(q) - float(c) for q, c in zip(acc_quantum, acc_classical)]
    crossover_p, idx = None, None
    for i, g in enumerate(gaps):
        worse = (g < 0) if strictly_worse else (g <= 0)
        if worse and i > 0:  # bỏ qua p=0 (điểm khởi tạo bằng nhau là bình thường)
            crossover_p = float(noise_levels[i])
            idx = i
            break
    return {"crossover_p": crossover_p, "index": idx, "gaps": gaps}


# ---------------------------------------------------------------------------
# Sweep summarization
# ---------------------------------------------------------------------------

def summarize_noise_sweep(records):
    """
    Gom danh sách record từ component-noise sweep thành cấu trúc báo cáo.
    Mỗi record: {variant, component, p, metric_name, value}
    Returns:
        {variant: {component: [(p, value), ...]}} đã sort theo p.
    """
    table = {}
    for r in records:
        v = table.setdefault(r["variant"], {}).setdefault(r["component"], [])
        v.append((r["p"], r["value"]))
    for variant in table:
        for comp in table[variant]:
            table[variant][comp].sort(key=lambda x: x[0])
    return table


def degradation_curve(values_by_level):
    """% suy giảm so với p=0 (level đầu). values: list[(p, value)]."""
    if not values_by_level:
        return []
    base = values_by_level[0][1]
    if base == 0:
        return [(p, 0.0) for p, _ in values_by_level]
    return [(p, 100.0 * (base - v) / abs(base)) for p, v in values_by_level]


def relative_noise_sensitivity(values_by_level):
    """
    Độ dốc suy giảm trung bình trên đơn vị noise (%metric / unit-p).
    Dùng để xếp hạng sensitivity: fusion-PQC vs attention-PQC vs head-PQC.
    """
    if len(values_by_level) < 2:
        return None
    (p0, v0), (_, v_last) = values_by_level[0], values_by_level[-1]
    p_last = values_by_level[-1][0]
    dp = p_last - p0
    if dp <= 0 or v0 == 0:
        return None
    return float(100.0 * (v0 - v_last) / abs(v0) / dp)
