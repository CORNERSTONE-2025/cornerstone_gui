import math

import numpy as np
from sklearn.metrics import confusion_matrix, recall_score, roc_auc_score

from cs_gui import metrics as gm


def test_tss_definition_matches_sensitivity_plus_specificity_minus_one():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    s = np.clip(y * 0.3 + rng.random(500) * 0.7, 0, 1)
    m = gm.point_metrics(y, s, 0.5)
    pred = (s >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    sens, spec = tp / (tp + fn), tn / (tn + fp)
    assert math.isclose(m["TSS"], sens + spec - 1)
    assert math.isclose(m["Recall"], recall_score(y, pred))
    assert math.isclose(m["ROC_AUC"], roc_auc_score(y, s))
    assert math.isclose(m["FPR"], 1 - spec) and math.isclose(m["FNR"], 1 - sens)
    assert (m["TP"], m["FP"], m["TN"], m["FN"]) == (tp, fp, tn, fn)


def test_same_tss_as_cornerstone_formula():
    # cornerstone.ml.utils.metrics.compute_metrics: tss = tp/(tp+fn) - fp/(fp+tn)
    y = np.array([1, 1, 0, 0, 0, 1, 0])
    s = np.array([.9, .2, .1, .7, .3, .8, .4])
    tp, fn, fp, tn = 2, 1, 1, 3
    assert math.isclose(gm.point_metrics(y, s, 0.5)["TSS"], tp / (tp + fn) - fp / (fp + tn))


def test_perfect_and_single_class():
    y = np.array([0, 0, 1, 1])
    assert gm.point_metrics(y, np.array([.1, .2, .8, .9]), .5)["TSS"] == 1.0
    one = gm.point_metrics(np.zeros(5), np.linspace(0, 1, 5), .5)
    assert math.isnan(one["ROC_AUC"]) and math.isnan(one["TSS"])     # no ground-truth positives


def test_bootstrap_ci_brackets_point_estimate_and_cluster_mode():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 400)
    s = np.clip(y * 0.4 + rng.random(400) * 0.6, 0, 1)
    pt = gm.point_metrics(y, s, 0.5)["TSS"]
    ci = gm.bootstrap_ci(y, s, 0.5, n_boot=300, seed=0)
    assert ci["TSS"]["low"] <= pt <= ci["TSS"]["high"]
    groups = np.repeat(np.arange(40), 10)
    cci = gm.bootstrap_ci(y, s, 0.5, n_boot=300, seed=0, group=groups)
    assert "cluster" in cci["_method"] and cci["TSS"]["n_valid"] > 0
    assert gm.bootstrap_ci(y, s, 0.5, n_boot=300, seed=0)["TSS"] == ci["TSS"]   # seeded => reproducible


def test_curves():
    y = np.array([0, 1, 0, 1, 1])
    s = np.array([.1, .9, .3, .6, .4])
    assert gm.roc_curve(y, s)["tpr"][-1] == 1.0
    assert len(gm.tss_vs_threshold(y, s, n=11)["TSS"]) == 11
