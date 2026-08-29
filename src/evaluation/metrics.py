"""
Comprehensive metrics for both tasks.
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, confusion_matrix
from scipy.stats import pearsonr, spearmanr
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction


def sentiment_metrics(preds, labels, num_classes=3):
    """Compute all sentiment analysis metrics."""
    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()

    metrics = {
        "accuracy": accuracy_score(labels_np, preds_np),
        "f1_macro": f1_score(labels_np, preds_np, average="macro", zero_division=0),
        "f1_weighted": f1_score(labels_np, preds_np, average="weighted", zero_division=0),
        "mae": mean_absolute_error(labels_np, preds_np),
        "confusion_matrix": confusion_matrix(labels_np, preds_np).tolist(),
    }

    if len(np.unique(labels_np)) > 1:
        metrics["pearson_r"], _ = pearsonr(preds_np.astype(float), labels_np.astype(float))
        metrics["spearman_rho"], _ = spearmanr(preds_np.astype(float), labels_np.astype(float))

    return metrics


def caption_metrics(references, hypotheses):
    """
    Compute captioning metrics.
    references: list of list of reference captions (tokenized)
    hypotheses: list of hypothesis captions (tokenized)
    """
    smooth = SmoothingFunction().method1

    metrics = {}
    for n in [1, 2, 3, 4]:
        weights = tuple([1.0 / n] * n)
        metrics[f"bleu_{n}"] = corpus_bleu(
            references, hypotheses, weights=weights, smoothing_function=smooth,
        )

    return metrics


def meteor_score_safe(references_tokenized, hypothesis_tokens):
    """
    GAP-3 evidence: METEOR (Banerjee & Lavie 2005) qua NLTK.
    Trả về None nếu wordnet chưa download / nltk thiếu — không làm crash eval.
    references_tokenized: [[w1, w2, ...], ...] (nhiều reference mỗi sample)
    """
    try:
        from nltk.translate.meteor_score import meteor_score as nltk_meteor
    except ImportError:
        return None
    try:
        scores = [
            nltk_meteor([list(r) for r in refs], list(hyp))
            for refs, hyp in zip(references_tokenized, hypothesis_tokens)
        ]
        return float(sum(scores) / len(scores))
    except LookupError:
        # nltk.download('wordnet') chưa chạy
        return None


def rouge_l_corpus(references_text, hypotheses_text, use_stemmer=True):
    """
    GAP-3 evidence: ROUGE-L (Lin 2004) qua rouge_score.
    Args:
        references_text: list[list[str]] — mỗi sample nhiều reference câu
        hypotheses_text: list[str]
    Returns:
        dict {'rouge_l_p', 'rouge_l_r', 'rouge_l_f'} trung bình corpus.
    """
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        return None

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=use_stemmer)
    agg = {"rouge_l_p": 0.0, "rouge_l_r": 0.0, "rouge_l_f": 0.0}
    n = len(hypotheses_text)
    if n == 0:
        return None
    for refs, hyp in zip(references_text, hypotheses_text):
        best = {"rouge_l_p": 0.0, "rouge_l_r": 0.0, "rouge_l_f": 0.0}
        for ref in refs:
            s = scorer.score(ref, hyp)["rougeL"]
            if s.fmeasure > best["rouge_l_f"]:
                best = {"rouge_l_p": s.precision,
                        "rouge_l_r": s.recall,
                        "rouge_l_f": s.fmeasure}
        for k in agg:
            agg[k] += best[k]
    return {k: v / n for k, v in agg.items()}


def caption_metrics_extended(references, hypotheses, detokenizer=None):
    """
    Bộ metric đầy đủ cho GAP-3: BLEU-1..4 + METEOR + ROUGE-L.
    Args:
        references: list of list of tokenized refs (giống caption_metrics)
        hypotheses: list of tokenized hyps
        detokenizer: callable(tokens)->str (tùy chọn, cho ROUGE-L)
    Returns:
        dict metrics; METEOR/ROUGE-L có thể là None nếu thiếu dependency.
    """
    metrics = caption_metrics(references, hypotheses)

    met = meteor_score_safe(references, hypotheses)
    if met is not None:
        metrics["meteor"] = met

    if detokenizer is not None:
        refs_text = [[detokenizer(r) for r in refs] for refs in references]
        hyps_text = [detokenizer(h) for h in hypotheses]
        rouge = rouge_l_corpus(refs_text, hyps_text)
        if rouge is not None:
            metrics.update(rouge)

    return metrics


def count_parameters(model):
    """Count trainable parameters (total, quantum, classical)."""
    quantum_params = 0
    classical_params = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            if "quantum" in name or "q_" in name:
                quantum_params += param.numel()
            else:
                classical_params += param.numel()

    return {
        "total": quantum_params + classical_params,
        "quantum": quantum_params,
        "classical": classical_params,
        "ratio": quantum_params / max(classical_params, 1),
    }
