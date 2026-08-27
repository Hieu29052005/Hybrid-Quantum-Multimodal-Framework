"""
End-to-end smoke tests for Week 4: full model + training loop.

Hermetic: uses tiny mock encoders and synthetic DataLoaders — no network,
no pretrained weights. Run: pytest tests/test_models_training.py -v
"""

import torch
import torch.nn as nn
import pytest
from torch.utils.data import DataLoader, Dataset

from src.models import (
    QuantumMultimodalFramework,
    QMMFConfig,
    build_classical_baseline,
)
from src.training import (
    MultiTaskLoss,
    fit,
    train_step,
    evaluate_msa,
    evaluate_caption,
    build_optimizer,
)


# ----------------------------------------------------------------------
# Mocks & fixtures
# ----------------------------------------------------------------------

class MockTextEncoder(nn.Module):
    """Same interface as TextEncoder; random tiny transformer-free."""

    def __init__(self, d_model=32, vocab=1000):
        super().__init__()
        self.emb = nn.Embedding(vocab, 16)
        self.proj = nn.Linear(16, d_model)
        self.d_out_val = d_model

    def forward(self, input_ids, attention_mask):
        x = self.emb(input_ids)                       # [B, L, 16]
        m = attention_mask.unsqueeze(-1).float()
        pooled = (x * m).sum(1) / m.sum(1).clamp(min=1)
        return self.proj(pooled)

    @property
    def d_out(self):
        return self.d_out_val


class MockImageEncoder(nn.Module):
    def __init__(self, d_model=32):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(3, d_model)

    def forward(self, images):
        v = self.pool(images).flatten(1)              # [B, 3]
        return self.proj(v)


def tiny_config(**kw):
    defaults = dict(
        d_model=32, n_qubits=6, n_q_layers=2, fusion_type="tensor",
        noise_prob=0.0, num_sentiment_classes=3, vocab_size=50,
        bos_token_id=1, eos_token_id=2, pad_token_id=0,
        max_caption_length=10, decoder_layers=2, decoder_heads=2,
    )
    defaults.update(kw)
    return QMMFConfig(**defaults)


def make_model(**kw):
    cfg = tiny_config(**kw)
    return QuantumMultimodalFramework(
        cfg, text_encoder=MockTextEncoder(cfg.d_model),
        image_encoder=MockImageEncoder(cfg.d_model))


class DummyMSA(Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, idx):
        g = torch.Generator().manual_seed(idx)
        return {
            "input_ids": torch.randint(3, 999, (6,), generator=g),
            "attention_mask": torch.ones(6, dtype=torch.long),
            "images": torch.randn(3, 32, 32),
            "labels": torch.randint(0, 3, (1,), generator=g).squeeze(0),
        }


class DummyCaption(Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, idx):
        g = torch.Generator().manual_seed(idx + 100)
        seq = torch.randint(3, 49, (7,), generator=g)      # target tokens
        cap_in = torch.cat([torch.tensor([1]), seq[:-1]])   # shifted right
        return {
            "images": torch.randn(3, 32, 32),
            "cap_in": cap_in,
            "cap_out": seq,
        }


@pytest.fixture(scope="module")
def model():
    return make_model()


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

def test_config_yaml_roundtrip(tmp_path):
    cfg = tiny_config()
    p = tmp_path / "cfg.yaml"
    cfg.save_yaml(p)
    loaded = QMMFConfig.from_yaml(p)
    assert loaded == cfg

    # unknown keys are ignored gracefully
    p.write_text("model:\n  n_qubits: 4\n  bogus_key: 123\n")
    assert QMMFConfig.from_yaml(p).n_qubits == 4


# ----------------------------------------------------------------------
# Full-model forwards
# ----------------------------------------------------------------------

def test_forward_sentiment(model):
    b = next(iter(DataLoader(DummyMSA(), batch_size=4)))
    logits = model.forward_sentiment(
        b["input_ids"], b["attention_mask"], b["images"])
    assert logits.shape == (4, 3)


def test_gradients_reach_quantum_params(model):
    b = next(iter(DataLoader(DummyMSA(), batch_size=4)))
    logits = model.forward_sentiment(
        b["input_ids"], b["attention_mask"], b["images"])
    logits.sum().backward()

    qgrads = [p.grad for n, p in model.named_parameters()
              if n.startswith("shared_quantum.q_fusion")]
    assert qgrads and all(g is not None for g in qgrads)


def test_forward_caption_shapes(model):
    b = next(iter(DataLoader(DummyCaption(), batch_size=4)))
    logits = model.forward_caption(b["images"], b["cap_in"])
    assert logits.shape == (4, 7, 50)


def test_causal_mask_no_leak(model):
    """Changing a FUTURE input token must not alter past-position logits."""
    model.eval()
    cap_in_a = torch.tensor([[1, 5, 6, 7, 8, 9, 10]])
    cap_in_b = cap_in_a.clone()
    cap_in_b[0, 4] = 42                      # perturb future position only
    ctx = torch.randn(1, 32)

    with torch.no_grad():
        la = model.caption_decoder(cap_in_a, ctx)
        lb = model.caption_decoder(cap_in_b, ctx)

    assert torch.allclose(la[:, :4], lb[:, :4], atol=1e-6), \
        "causal mask leaked future information"


def test_generate_caption(model):
    imgs = torch.randn(3, 3, 32, 32)
    out = model.generate_caption(imgs, max_len=6)
    assert out.shape[0] == 3 and out.shape[1] <= 6
    assert out.dtype == torch.long


# ----------------------------------------------------------------------
# Loss / optimizer
# ----------------------------------------------------------------------

def test_multitask_loss_math():
    crit = MultiTaskLoss(w_msa=2.0, w_cap=3.0, ignore_index=0)
    msa_logits = torch.randn(4, 3)
    msa_labels = torch.tensor([0, 1, 2, 1])
    cap_logits = torch.randn(2, 5, 10)
    cap_labels = torch.randint(1, 10, (2, 5))

    total = crit(msa_logits=msa_logits, msa_labels=msa_labels,
                 cap_logits=cap_logits, cap_labels=cap_labels)

    ce_msa = torch.nn.functional.cross_entropy(msa_logits, msa_labels)
    ce_cap = torch.nn.functional.cross_entropy(
        cap_logits.view(-1, 10), cap_labels.view(-1))
    assert torch.allclose(total, 2.0 * ce_msa + 3.0 * ce_cap)


def test_optimizer_quantum_lr_group():
    m = make_model(fusion_type="interference")
    opt = build_optimizer(m, lr=1e-3, quantum_lr_mult=10.0)
    lrs = sorted({g["lr"] for g in opt.param_groups})
    assert lrs == [pytest.approx(1e-3), pytest.approx(1e-2)]


# ----------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------

def test_train_step_and_eval(tmp_path):
    m = make_model()
    loader = DataLoader(DummyMSA(), batch_size=4)
    batch = next(iter(loader))
    crit = MultiTaskLoss(ignore_index=0)
    opt = build_optimizer(m, lr=1e-3)

    loss = train_step(m, batch, crit, opt)
    assert torch.isfinite(torch.tensor(loss))

    acc, _ = evaluate_msa(m, loader)
    assert 0.0 <= acc <= 1.0


def test_fit_single_and_multitask(tmp_path):
    # --- single-task sentiment ---
    m1 = make_model()
    hist1 = fit(
        m1, {"msa": DataLoader(DummyMSA(), batch_size=4)},
        val_loaders={"msa": DataLoader(DummyMSA(), batch_size=4)},
        mode="sentiment", epochs=2, lr=1e-3, warmup_steps=2,
        device="cpu", out_dir=str(tmp_path / "msa"), patience=3,
        log_every=1000, log_fn=None,
    )
    assert len(hist1["train_loss"]) == 2
    ckpt = tmp_path / "msa" / "best_model.pt"
    assert ckpt.exists()

    # --- multitask ---
    m2 = make_model(fusion_type="attention")
    tl = {"msa": DataLoader(DummyMSA(), batch_size=4),
          "caption": DataLoader(DummyCaption(), batch_size=4)}
    vl = {"caption": DataLoader(DummyCaption(), batch_size=4)}
    hist2 = fit(
        m2, tl, val_loaders=vl, mode="multitask", epochs=2, lr=1e-3,
        warmup_steps=2, device="cpu", out_dir=str(tmp_path / "joint"),
        patience=3, log_every=1000, log_fn=None,
    )
    assert len(hist2["train_loss"]) == 2
    assert hist2["quantum_params"] > 0


def test_caption_eval_runs():
    m = make_model()
    loss = evaluate_caption(m, DataLoader(DummyCaption(), batch_size=4))
    assert torch.isfinite(torch.tensor(loss))


# ----------------------------------------------------------------------
# Classical baselines
# ----------------------------------------------------------------------

def test_classical_fusion_mlp():
    from src.models import ClassicalFusionMLP
    from src.models.classical_models import (
        ClassicalMSAModel, ClassicalCaptionModel)

    # Fusion MLP standalone: matches quantum layer's role
    fus = ClassicalFusionMLP(d_model=32, n_qubits=6, n_layers=2)
    out = fus(torch.randn(4, 32), torch.randn(4, 32))
    assert out.shape == (4, 32)

    # Full MSA model with injected mocks
    msa = ClassicalMSAModel(text_encoder=MockTextEncoder(32),
                            image_encoder=MockImageEncoder(32),
                            d_model=32, n_classes=3)
    b = next(iter(DataLoader(DummyMSA(), batch_size=4)))
    logits = msa(b["input_ids"], b["attention_mask"], b["images"])
    assert logits.shape == (4, 3)

    # Caption model with injected mock image encoder
    cap = ClassicalCaptionModel(
        vocab_size=50, d_model=32, decoder_layers=1, decoder_heads=2,
        max_len=8, bos_token_id=1, eos_token_id=2, pad_token_id=0,
        image_encoder=MockImageEncoder(32))
    bc = next(iter(DataLoader(DummyCaption(), batch_size=4)))
    cl = cap(bc["images"], bc["cap_in"])
    assert cl.shape == (4, 7, 50)
    gen = cap.generate_caption(bc["images"], max_len=5)
    assert gen.shape[0] == 4 and gen.shape[1] <= 5
