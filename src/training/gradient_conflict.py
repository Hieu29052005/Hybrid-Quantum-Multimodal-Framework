"""
GAP-2 (RESEARCH_GAP.md): Gradient conflict / transfer analysis giữa
MSA (discriminative) và Captioning (generative) trên SHARED PQC weights.

Cung cấp:
    - shared_param_names(): tham số quantum dùng chung cần phân tích
    - task_gradients(): gradient vector của từng loss trên các param đó
    - gradient_cosine_similarity(): cos(g_msa, g_cap) — <0 ⇒ xung đột
      ("phân tích gradient conflict giữa 2 tasks trên shared PQC weights")
    - pcgrad_backward(): PCGrad (Yu et al. 2020) — project gradient task ra
      mặt phẳng vuông góc với task khác trước khi cộng
    - analyze_conflict_over_batches(): thống kê cos-sim qua nhiều batch

Kỳ vọng: nếu cos-sim trung bình > 0, sharing có lợi (transfer dương);
nếu nhiều batch cos<0 ⇒ cần PCGrad hoặc tách PQC.
"""

import torch


# ---------------------------------------------------------------------------
# Parameter selection
# ---------------------------------------------------------------------------

def shared_param_names(model, patterns=("shared_quantum", "q_fusion")):
    """Tên các parameter thuộc thành phần SHARED (ưu tiên PQC fusion)."""
    names = []
    for name, p in model.named_parameters():
        if p.requires_grad and any(pat in name for pat in patterns):
            names.append(name)
    return names


def _flatten_grads(model, names):
    vecs = []
    lookup = dict(model.named_parameters())
    for n in names:
        g = lookup[n].grad
        if g is not None:
            vecs.append(g.reshape(-1))
        else:
            vecs.append(torch.zeros(lookup[n].numel(),
                                    device=lookup[n].device))
    return torch.cat(vecs) if vecs else None


# ---------------------------------------------------------------------------
# Per-task gradients & cosine conflict
# ---------------------------------------------------------------------------

def task_gradients(model, msa_loss_fn, cap_loss_fn, patterns=("shared_quantum",)):
    """
    Tính gradient vector của từng task trên shared params.
    Args:
        msa_loss_fn / cap_loss_fn: zero-arg callables trả về scalar loss
            (caller tự forward trước; hàm chỉ backward).
        LƯU Ý: cap_loss_fn cần retain_graph=True ở graph chung nếu hai loss
        share subgraph — caller đảm bảo bằng cách forward riêng cho mỗi task.
    Returns:
        (g_msa, g_cap): tensor 1-D hoặc None nếu không có grad.
    """
    model.zero_grad(set_to_none=False)
    msa_loss_fn().backward()
    g_msa = _flatten_grads(model, shared_param_names(model, patterns))

    model.zero_grad(set_to_none=False)
    cap_loss_fn().backward()
    g_cap = _flatten_grads(model, shared_param_names(model, patterns))

    model.zero_grad()  # dọn sạch để optimizer.step() ngoài không bị nhiễu
    return g_msa, g_cap


def cosine_similarity(g1, g2, eps=1e-12):
    if g1 is None or g2 is None:
        return None
    denom = g1.norm() * g2.norm()
    if denom < eps:
        return None
    return float((g1 @ g2) / denom)


@torch.no_grad()
def gradient_cosine_similarity(model, msa_loss_fn, cap_loss_fn,
                               patterns=("shared_quantum",)):
    """
    Wrapper tiện dụng: trả về {'cos': float|None,
                               'g_norms': [||g_msa||, ||g_cap||],
                               'conflict': bool} .
    """
    model.zero_grad(set_to_none=False)
    msa_loss_fn().backward(retain_graph=True)
    g_msa = _flatten_grads(model, shared_param_names(model, patterns))

    model.zero_grad(set_to_none=False)
    cap_loss_fn().backward()
    g_cap = _flatten_grads(model, shared_param_names(model, patterns))
    model.zero_grad()

    cos = cosine_similarity(g_msa, g_cap)
    norms = [float(g.norm()) if g is not None else None for g in (g_msa, g_cap)]
    return {"cos": cos, "g_norms": norms,
            "conflict": bool(cos is not None and cos < 0)}


# ---------------------------------------------------------------------------
# PCGrad
# ---------------------------------------------------------------------------

def pcgrad_backward(model, losses, patterns=("shared_quantum",)):
    """
    PCGrad (Yu et al., NeurIPS 2020) cho K losses bất kỳ.

    1) autograd.grad từng loss (retain_graph=True) → vector per-task.
    2) Nếu cos(g_i, g_j) < 0: g_i ← g_i − (g_i·ĝ_j)·ĝ_j  (projection).
    3) Cộng tất cả vector đã project vào .grad của params.

    Args:
        losses: list scalar tensors (cùng graph chung của forward multitask)
    Returns:
        list các cos-sim cặp (để logging).
    """
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    name_to_p = dict(params)

    # 1) per-task grads trên TOÀN BỘ param (projection chỉ áp lên shared;
    #    task-specific params không xung đột nên giữ nguyên)
    task_vecs = []
    for loss in losses:
        grads = torch.autograd.grad(loss, [p for _, p in params],
                                    retain_graph=True, allow_unused=True)
        task_vecs.append(
            {name: (g if g is not None else torch.zeros_like(p))
             for (name, p), g in zip(params, grads)}
        )

    # 2) projection pairwise trên shared subset
    shared = [n for n in shared_param_names(model, patterns)]
    cos_log = []
    for i in range(len(task_vecs)):
        for j in range(len(task_vecs)):
            if i == j:
                continue
            gi = torch.cat([task_vecs[i][n].reshape(-1) for n in shared])
            gj = torch.cat([task_vecs[j][n].reshape(-1) for n in shared])
            dot = gi @ gj
            norm_j = gj.norm().clamp_min(1e-12)
            cos_log.append(float(dot / (gi.norm().clamp_min(1e-12) * norm_j)))
            if dot < 0:
                proj = (dot / norm_j ** 2)
                with torch.no_grad():
                    for n in shared:
                        task_vecs[i][n] = task_vecs[i][n] - proj * task_vecs[j][n]

    # 3) accumulate
    model.zero_grad()
    for name, p in params:
        agg = sum(task_vecs[k][name] for k in range(len(task_vecs)))
        p.grad = agg
    return cos_log


# ---------------------------------------------------------------------------
# Batch-level statistics
# ---------------------------------------------------------------------------

def analyze_conflict_over_batches(model, make_losses_fn, n_batches=20):
    """
    Chạy phân tích conflict qua nhiều batch.

    Args:
        make_losses_fn(batch_idx): callable nhận index batch, trả về
            (msa_loss, cap_loss) đã forward trên batch đó.
    Returns:
        {'cos_values': [...], 'mean_cos': float, 'frac_conflict': float}
    """
    cos_values = []
    for b in range(n_batches):
        msa_loss, cap_loss = make_losses_fn(b)
        res = gradient_cosine_similarity(model,
                                         lambda l=msa_loss: l,
                                         lambda l=cap_loss: l)
        if res["cos"] is not None:
            cos_values.append(res["cos"])

    return {
        "cos_values": cos_values,
        "mean_cos": float(sum(cos_values) / len(cos_values)) if cos_values else None,
        "frac_conflict": (sum(1 for c in cos_values if c < 0) / len(cos_values))
        if cos_values else None,
    }
