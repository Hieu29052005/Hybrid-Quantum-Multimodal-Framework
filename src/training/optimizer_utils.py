"""
Optimizer utilities: LR scheduling, gradient clipping, warmup.
"""

import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """Cosine schedule with linear warmup."""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item()))
    return LambdaLR(optimizer, lr_lambda)


def setup_optimizer(model, lr=1e-4, weight_decay=0.01, warmup_steps=500, total_steps=10000):
    """Setup AdamW optimizer with cosine warmup schedule."""
    no_decay = ["bias", "LayerNorm.weight"]
    param_groups = [
        {
            "params": [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=lr)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    return optimizer, scheduler


def clip_gradients(model, max_norm=1.0):
    """Clip gradients by global norm."""
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
