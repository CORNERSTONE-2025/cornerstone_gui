"""Parameter validation (no package needed: summaries are synthetic)."""
import copy

from cs_gui.pipeline import (DEFAULT_TRAINING_CONFIG, validate_inference, validate_model_config, validate_split,
                             validate_training)

DATASET = {"summary": {"feature_names": ["A", "B", "CMASK"], "timesteps": 121,
                       "time_start": "2012-01-01 00:00:00", "time_end": "2014-12-31 00:00:00"}}
INFO = {"trainable_models": ["LSTM", "XGBOOST", "ROCKET"], "sk_models": ["XGBOOST", "ROCKET"],
        "gpu_only_models": ["ROCKET"], "cuda_available": False,
        "model_configs": {"LSTM": {"input_dim": None, "model_dim": 128, "bidirectional": True, "dropout_prob": 0.3},
                          "XGBOOST": {"max_depth": 3}},
        "feature_groups": {"g": ["A", "ZZZ"]}}


def split_cfg(**kw):
    c = {"FEATURE_SELECTION_MODE": "direct", "DIRECT_FEATURES": ["A", "B"], "TIMESTEPS": 121,
         "MISSINGNESS_FILTER_MODE": "multi_window_empty", "TIME_WINDOWS": [1, 16, 121], "TEMPORAL_FILTER_MODE": None,
         "TEMPORAL_WINDOW": None, "MULTI_PERIOD_FILTER": False, "PROBLEMATIC_PERIODS": [], "SPLITTING_STRATEGY": "random",
         "TEST_SIZE": 0.2, "N_FOLDS": 5, "RANDOM_STATE": 42, "TEST_CUTOFF_DATE": None, "TEST_TIME_WINDOW": None,
         "CMASK_FILTER": False, "MULTI_CLASS": False}
    c.update(kw)
    return c


def test_valid_split_config():
    assert validate_split(split_cfg(), DATASET, INFO) == []


def test_split_errors():
    assert any("not in the dataset" in e for e in validate_split(split_cfg(DIRECT_FEATURES=["Q"]), DATASET, INFO))
    assert any("missing from this dataset" in e for e in validate_split(split_cfg(FEATURE_SELECTION_MODE="criterion",
                                                                FEATURE_CRITERION="g"), DATASET, INFO))
    assert any("TIMESTEPS" in e for e in validate_split(split_cfg(TIMESTEPS=500), DATASET, INFO))
    assert any("chronological" in e for e in validate_split(split_cfg(TEST_CUTOFF_DATE="2013-01-01"), DATASET, INFO))
    errs = validate_split(split_cfg(SPLITTING_STRATEGY="chronological", TEST_CUTOFF_DATE="2020-01-01"), DATASET, INFO)
    assert any("outside the dataset range" in e for e in errs)
    errs = validate_split(split_cfg(TEMPORAL_FILTER_MODE="exclude", TEMPORAL_WINDOW=["2013-05-01", "2013-01-01"]), DATASET, INFO)
    assert any("start < end" in e for e in errs)
    assert any("MULTI_CLASS" in e for e in validate_split(split_cfg(MULTI_CLASS=True), DATASET, INFO))
    assert any("invalid date" in e for e in validate_split(
        split_cfg(SPLITTING_STRATEGY="chronological", TEST_CUTOFF_DATE="13/01/2013"), DATASET, INFO))


def mc_cfg(model="LSTM", **kw):
    c = {"model": model, "model_params": {}, "training_config": copy.deepcopy(DEFAULT_TRAINING_CONFIG),
         "regularization_params": None, "seed": 1}
    c.update(kw)
    return c


def test_model_config_validation():
    assert validate_model_config(mc_cfg(), INFO) == []
    assert validate_model_config(mc_cfg(model="NOPE"), INFO)
    assert any("integer" in e for e in validate_model_config(mc_cfg(model_params={"model_dim": 1.5}), INFO))
    assert any("boolean" in e for e in validate_model_config(mc_cfg(model_params={"bidirectional": 1}), INFO))
    assert any("regularization" in e for e in validate_model_config(mc_cfg(model="XGBOOST"), INFO))
    assert validate_model_config(mc_cfg(model="XGBOOST", regularization_params=[1.0, 10.0]), INFO) == []
    bad = mc_cfg()
    bad["training_config"]["early_stopping"] = {"monitor": "val_loss", "mode": "max"}
    assert any("val_loss" in e for e in validate_model_config(bad, INFO))


SPLIT = {"summary": {"timesteps": 61, "leakage": {"n_folds": 3, "test_vs_trainval_shared_harps": 0,
                                                  "test_vs_trainval_shared_samples": 0}}}


def test_training_validation():
    ok = {"windows": ["3h"], "window_map": {"3h": 16}, "n_folds": 3, "device": "auto"}
    assert validate_training(ok, SPLIT, {"config": {"model": "LSTM"}}, INFO) == []
    assert any("longer than" in e for e in validate_training({**ok, "windows": ["24h"], "window_map": {"24h": 121}},
                                                           SPLIT, {"config": {"model": "LSTM"}}, INFO))
    assert any("n_folds" in e for e in validate_training({**ok, "n_folds": 5}, SPLIT, {"config": {"model": "LSTM"}}, INFO))
    assert any("GPU" in e for e in validate_training(ok, SPLIT, {"config": {"model": "ROCKET"}}, INFO))
    leaky = {"summary": {**SPLIT["summary"], "leakage": {**SPLIT["summary"]["leakage"], "test_vs_trainval_shared_harps": 2}}}
    assert any("leakage" in e for e in validate_training(ok, leaky, {"config": {"model": "LSTM"}}, INFO))


def test_inference_validation_blocks_trainval_of_own_split():
    tr = {"summary": {"windows": {"3h": 16}, "n_folds": 3}, "parents": {"split": "spl-1"}}
    src = {"id": "spl-1", "type": "split", "summary": {"files": {"test": {}, "cv_0_val": {}}}}
    base = {"windows": ["3h"], "threshold_mode": "cv", "model_choice": "best", "source_file": "test"}
    assert validate_inference(base, tr, src) == []
    assert any("leakage" in e for e in validate_inference({**base, "source_file": "cv_0_val"}, tr, src))
    assert any("threshold" in e for e in validate_inference({**base, "threshold_mode": "fixed", "threshold": 1.5}, tr, src))
    assert any("was not trained" in e for e in validate_inference({**base, "windows": ["24h"]}, tr, src))
