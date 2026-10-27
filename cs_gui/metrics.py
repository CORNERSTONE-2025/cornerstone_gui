"""Evaluation metrics computed from per-sample predictions.

Definitions (binary task, positive class = flare ``YES``):

- Sensitivity (Recall, TPR) = TP / (TP + FN)
- Specificity (TNR)         = TN / (TN + FP)
- **TSS = Sensitivity + Specificity - 1** (= TPR - FPR, same as ``cornerstone.ml.utils.metrics``)
- HSS = 2 (TP·TN - FP·FN) / [(TP+FN)(FN+TN) + (TP+FP)(FP+TN)]
- ROC-AUC and PR-AUC (average precision) are threshold-free and need both classes.

Confidence intervals: non-parametric percentile bootstrap over samples (default 1000
resamples, fixed seed). Samples of the same Active Region are correlated, so the
sample-level bootstrap can underestimate the uncertainty; ``group`` enables a
cluster bootstrap by Active Region (HARPNUM) which is the recommended choice.
"""
from __future__ import annotations

import math

import numpy as np

THRESHOLD_FREE = ("ROC_AUC", "PR_AUC")


def confusion(y: np.ndarray, pred: np.ndarray) -> tuple[int, int, int, int]:
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    return tp, tn, fp, fn


def _div(a: float, b: float) -> float:
    return a / b if b else float("nan")


def point_metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    """All threshold-dependent and threshold-free metrics for one set of predictions."""
    y = np.asarray(y).astype(int)
    score = np.nan_to_num(np.asarray(score, dtype=float), nan=0.0)
    pred = (score >= threshold).astype(int)
    tp, tn, fp, fn = confusion(y, pred)
    sens = _div(tp, tp + fn)
    spec = _div(tn, tn + fp)
    prec = _div(tp, tp + fp)
    denom_hss = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    out = {
        "threshold": float(threshold),
        "n": int(len(y)), "n_pos": int(y.sum()), "n_neg": int(len(y) - y.sum()),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "Sensitivity": sens, "Specificity": spec,
        "TSS": (sens + spec - 1) if not (math.isnan(sens) or math.isnan(spec)) else float("nan"),
        "HSS": _div(2 * (tp * tn - fp * fn), denom_hss),
        "Precision": prec, "Recall": sens,
        "F1": _div(2 * tp, 2 * tp + fp + fn),
        "Accuracy": _div(tp + tn, len(y)),
        "FPR": _div(fp, fp + tn), "FNR": _div(fn, fn + tp),
        "MCC": _div(tp * tn - fp * fn, math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) or 0),
    }
    if 0 < y.sum() < len(y):
        from sklearn.metrics import average_precision_score, roc_auc_score
        out["ROC_AUC"] = float(roc_auc_score(y, score))
        out["PR_AUC"] = float(average_precision_score(y, score))
    else:
        out["ROC_AUC"] = out["PR_AUC"] = float("nan")
    return out


def bootstrap_ci(y, score, threshold: float, metrics=("TSS", "HSS", "F1", "ROC_AUC"),
                 n_boot: int = 1000, alpha: float = 0.05, seed: int = 0,
                 group: np.ndarray | None = None) -> dict:
    """Percentile bootstrap CI. With ``group`` resamples whole groups (cluster bootstrap)."""
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    rng = np.random.default_rng(seed)
    vals: dict[str, list[float]] = {m: [] for m in metrics}
    if group is not None:
        group = np.asarray(group)
        uniq = np.unique(group)
        members = {g: np.flatnonzero(group == g) for g in uniq}
    for _ in range(n_boot):
        if group is not None:
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([members[g] for g in pick])
        else:
            idx = rng.integers(0, len(y), len(y))
        pm = point_metrics(y[idx], score[idx], threshold)
        for m in metrics:
            vals[m].append(pm[m])
    out = {}
    for m, v in vals.items():
        arr = np.asarray(v, dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size < max(10, n_boot // 10):
            out[m] = {"low": float("nan"), "high": float("nan"), "n_valid": int(arr.size)}
        else:
            out[m] = {"low": float(np.quantile(arr, alpha / 2)), "high": float(np.quantile(arr, 1 - alpha / 2)),
                      "n_valid": int(arr.size)}
    out["_method"] = (f"percentile bootstrap, {n_boot} resamples, {'cluster by HARPNUM' if group is not None else 'per-sample'}, "
                      f"{int((1 - alpha) * 100)}% CI, seed={seed}")
    return out


def roc_curve(y, score) -> dict:
    from sklearn.metrics import roc_curve as _roc
    fpr, tpr, thr = _roc(np.asarray(y).astype(int), np.asarray(score, dtype=float))
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thr.tolist()}


def pr_curve(y, score) -> dict:
    from sklearn.metrics import precision_recall_curve
    p, r, thr = precision_recall_curve(np.asarray(y).astype(int), np.asarray(score, dtype=float))
    return {"precision": p.tolist(), "recall": r.tolist(), "thresholds": thr.tolist()}


def tss_vs_threshold(y, score, n: int = 101) -> dict:
    ths = np.linspace(0, 1, n)
    return {"threshold": ths.tolist(), "TSS": [point_metrics(y, score, t)["TSS"] for t in ths]}
