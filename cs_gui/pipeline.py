"""Application logic: parameter specs and validation, artifact creation, variants, reproduction.

All the option lists below mirror what the CORNERSTONE package actually implements; the
source location is given next to each one so it can be checked when the package changes.
"""
from __future__ import annotations

import copy
import datetime as dt
import json
import re
import subprocess
from pathlib import Path

from . import datastore
from .jobs import runner_env
from .settings import Settings
from .store import ArtifactStore, StoreError

# --------------------------------------------------------------------------- package options
SERIES = datastore.SERIES                                                             # SHARP series of the store
THRESHOLDS = {"C": "C (flux ≥ 1e-6 W/m²)", "M": "M (≥ 1e-5)", "X": "X (≥ 1e-4)"}       # labeling.threshold_flux
FEATURE_SETS = {"clean": "clean (17 features, _N1/_N2 normalized)", "all": "all (22 raw SHARP keywords)"}  # processing.features
CALIBRATIONS = {"science": "science (GOES-8…15 fluxes / 0.7, consistent across satellites)",
                "operational": "operational (as reported by SWPC)"}                    # processing.FlareConfig
AMBIGUOUS_RULES = {"position_then_flux": "patch containing the flare position, else largest USFLUX",
                   "position_then_all": "patch containing the flare position, else every candidate",
                   "flux": "largest USFLUX", "all": "every candidate HARP"}
QUALITY_KEYS = ("require_position", "quality_mask", "quality_values", "cmask_gt", "meangbt_gt", "max_abs_lon",
                "exclude_years")                                                       # processing.QualityFilter
FLARE_DEFAULTS = {"calibration": "science", "peak_fallback": "start", "ambiguous": "position_then_flux",
                  "position_margin_deg": 2.0, "max_gap_hours": 24.0}
_THRESHOLD_RE = re.compile(r"^[ABCMX](\d+(\.\d+)?)?$")


def quality_preset(name: str, series: str) -> dict:
    """``QualityFilter`` parameters of the presets (same values as the package)."""
    q = {"require_position": True, "quality_mask": None, "quality_values": None, "cmask_gt": None,
         "meangbt_gt": None, "max_abs_lon": None, "exclude_years": []}
    if name == "legacy_clean":          # QualityFilter.legacy_clean: rows of the former SHARPC_clean tables
        q.update(quality_values=[0x400] if series == "NRT" else [0], cmask_gt=5000.0, meangbt_gt=0.0,
                 exclude_years=[2013] if series == "NRT" else [])
    elif name != "default":
        raise ValueError(f"unknown quality preset {name!r}")
    return q


def default_label_config(series: str = "DEF", threshold: str = "M", **overrides) -> dict:
    """Labeling configuration with the package defaults (unlimited before/after windows)."""
    cfg = {"series": series, "threshold": threshold, "flare_window_hours": 24.0, "step_hours": 24.0,
           "strat_threshold": None, "strat_window_hours": None, "start": None, "end": None,
           "quality": quality_preset("default", series), "flares": dict(FLARE_DEFAULTS)}
    cfg.update(overrides)
    return cfg


def label_key(cfg: dict) -> str:
    """Name of the label in the HDF5 files (``cornerstone.dataset.label_name``), e.g. ``Mplus_delta24``."""
    return f"{cfg['threshold']}plus_delta{float(cfg['flare_window_hours']):g}"
MISSINGNESS_MODES = ["multi_window_empty", "multi_window", "single_window", "none"]   # create_dataset_splits step 8
TEMPORAL_FILTER_MODES = [None, "select", "filter", "exclude"]                          # step 9
CMASK_STRATEGIES = ["any", "majority", "average", "minimum"]                          # CMASKFilter
SPLIT_STRATEGIES = ["random", "chronological"]                                        # SharpDatasetSplitter
LOSS_TYPES = ["bce", "focal", "asymmetric_focal", "curriculum"]                       # losses.create_loss
SCHEDULERS = ["CosineAnnealingLR", "ReduceLROnPlateau", None]                         # training.train_model
THRESHOLD_METRICS = ["TSS", "F1", "MCC", "precision", "recall", "precision_at_recall", "f1_at_precision"]
MONITORS = ["tss", "val_loss", "auc", "f1", "mcc", "hss"]                             # early stopping
CALIBRATION_TYPES = ["platt", "temperature", "auto"]                                  # calibration.create_calibrator

DEFAULT_TRAINING_CONFIG = {
    "batch_size": 64, "learning_rate": 1e-4, "weight_decay": 1e-4, "num_epochs": 100, "patience": 20,
    "n_folds": 5,
    "lr_scheduler": {"type": "CosineAnnealingLR", "T_max": 100, "eta_min": 1e-6},
    "threshold_optimization": {"enabled": True, "metric": "TSS", "n_thresholds": 50,
                               "threshold_range": [0.0, 1.0], "min_recall": 0.6, "min_precision": 0.3,
                               "optimize_every_epoch": True},
    "calibration": {"enabled": False, "type": "platt", "few_pos_threshold": 25, "n_bins": 10},
    "early_stopping": {"monitor": "tss", "mode": "max", "min_delta": 1e-3, "patience": 20},
    "loss_function": {"type": "bce"},
}


def default_split_config(settings: Settings) -> dict:
    p = Path(settings.cornerstone_root) / "cornerstone" / "configs" / "split_config.example.json"
    cfg = json.loads(p.read_text()) if p.exists() else {}
    for k in ("INPUT_H5_FILE", "OUTPUT_BASE_DIR", "EXPERIMENT_NAME", "LABEL_TYPE"):
        cfg.pop(k, None)               # set by the runner from the parent artifact
    cfg.setdefault("SAVE_PLOTS", True)
    cfg["VERBOSE"] = True
    cfg["MULTI_CLASS"] = False         # training/evaluation in the package are binary (YES vs NO*)
    return cfg


# --------------------------------------------------------------------------- package introspection
def package_info(settings: Settings, refresh: bool = False, timeout: int = 600) -> dict:
    out = settings.active_workspace / "package_info.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if refresh or not out.exists():
        r = subprocess.run([settings.python, "-m", "cs_gui.runner", "--introspect", str(out),
                            "--cornerstone-root", settings.cornerstone_root],
                           env=runner_env(settings), capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            raise RuntimeError("Package introspection failed:\n" + r.stdout[-3000:] + r.stderr[-3000:])
    return json.loads(out.read_text())


# --------------------------------------------------------------------------- validation helpers
def _date(s, field) -> dt.date | None:
    if s in (None, ""):
        return None
    try:
        return dt.date.fromisoformat(str(s)[:10])
    except ValueError:
        raise ValueError(f"{field}: invalid date '{s}' (format YYYY-MM-DD)")


def _check_range(errs, name, value, lo=None, hi=None, integer=False):
    if value is None:
        errs.append(f"{name}: value required")
        return
    if integer and (not isinstance(value, int) or isinstance(value, bool)):
        errs.append(f"{name}: must be an integer")
        return
    if lo is not None and value < lo:
        errs.append(f"{name}: must be ≥ {lo}")
    if hi is not None and value > hi:
        errs.append(f"{name}: must be ≤ {hi}")


def _require_completed(store: ArtifactStore, aid: str | None, expected_type, role: str) -> dict:
    if not aid:
        raise StoreError(f"Select the '{role}' artifact")
    m = store.get(aid)
    types = expected_type if isinstance(expected_type, (list, tuple)) else [expected_type]
    if m["type"] not in types:
        raise StoreError(f"'{role}' must be of type {types}, found {m['type']}")
    if m["status"] != "completed":
        raise StoreError(f"Artifact {aid} ({role}) is not completed (status: {m['status']})")
    return m


# --------------------------------------------------------------------------- validators
def _check_threshold(errs, name, v):
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if v <= 0:
            errs.append(f"{name}: a flux must be > 0")
    elif not (isinstance(v, str) and _THRESHOLD_RE.match(v.strip().upper())):
        errs.append(f"{name}: a GOES class such as C, M, X, M5 or C2.5, or a flux in W/m²")


def validate_labels(cfg: dict, settings: Settings) -> list[str]:
    errs = []
    if cfg.get("series") not in SERIES:
        errs.append(f"series: one of {list(SERIES)}")
    _check_threshold(errs, "threshold", cfg.get("threshold"))
    if cfg.get("strat_threshold") is not None:
        _check_threshold(errs, "strat_threshold", cfg.get("strat_threshold"))
    _check_range(errs, "flare_window_hours", cfg.get("flare_window_hours"), 0.2, 24 * 30)
    _check_range(errs, "step_hours", cfg.get("step_hours"), 0.2, 24 * 30)
    if cfg.get("strat_window_hours") is not None:
        _check_range(errs, "strat_window_hours", cfg.get("strat_window_hours"), 0.2, 24 * 365)
    try:
        a, b = _date(cfg.get("start"), "start"), _date(cfg.get("end"), "end")
        if a and b and a >= b:
            errs.append("start must be before end")
    except ValueError as e:
        errs.append(str(e))
    q = cfg.get("quality") or {}
    if set(q) != set(QUALITY_KEYS):
        errs.append(f"quality: parameters {list(QUALITY_KEYS)}")
    else:
        for k in ("cmask_gt", "meangbt_gt", "max_abs_lon"):
            if q[k] is not None and not isinstance(q[k], (int, float)):
                errs.append(f"quality.{k}: number or empty")
        if q["max_abs_lon"] is not None and not 0 < q["max_abs_lon"] <= 90:
            errs.append("quality.max_abs_lon: between 0 and 90 degrees")
    f = cfg.get("flares") or {}
    if f.get("calibration") not in CALIBRATIONS:
        errs.append(f"flares.calibration: one of {list(CALIBRATIONS)}")
    if f.get("ambiguous") not in AMBIGUOUS_RULES:
        errs.append(f"flares.ambiguous: one of {list(AMBIGUOUS_RULES)}")
    if f.get("peak_fallback") not in ("start", "drop"):
        errs.append("flares.peak_fallback: 'start' or 'drop'")
    _check_range(errs, "flares.max_gap_hours", f.get("max_gap_hours"), 0, 24 * 30)
    _check_range(errs, "flares.position_margin_deg", f.get("position_margin_deg"), 0, 30)
    if errs:
        return errs
    root = settings.active_data_root
    if not datastore.has_series(root, cfg["series"]):
        errs.append(f"No SHARP {cfg['series']} data in the store: download it from the Data page")
    st = datastore.status(root)
    if "goes/swpc_events" not in set(st["key"]):
        errs.append("No GOES flare events in the store: download them from the Data page")
    if cfg["series"] == "DEF" and "harp_noaa" not in set(st["key"]):
        errs.append("The HARP ↔ NOAA map (needed for DEF) is missing: download it from the Data page")
    return errs


def validate_dataset(cfg: dict, labels: dict, settings: Settings) -> list[str]:
    errs = []
    if "series" not in labels["config"]:
        return ["These labels were made by an earlier GUI version (SQLite database): create new labels"]
    if cfg.get("features") not in FEATURE_SETS:
        errs.append(f"features: one of {list(FEATURE_SETS)}")
    if cfg.get("quality_source") not in ("feature_set", "labels"):
        errs.append("quality_source: 'feature_set' or 'labels'")
    _check_range(errs, "window_hours", cfg.get("window_hours"), 1, 240, integer=True)
    _check_range(errs, "sampling_minutes", cfg.get("sampling_minutes"), 1, 120, integer=True)
    if cfg.get("subsample"):
        _check_range(errs, "subsample_size", cfg.get("subsample_size"), 1, labels["summary"].get("n_rows"), integer=True)
    if errs:
        return errs
    sm = cfg["sampling_minutes"]
    if 1440 % sm:
        errs.append("sampling_minutes must divide a day (e.g. 6, 12, 15, 30, 60)")
    if (cfg["window_hours"] * 60) % sm:
        errs.append("window_hours × 60 must be a multiple of sampling_minutes")
    step_min = float(labels["config"].get("step_hours", 24)) * 60
    if step_min % sm:
        errs.append(f"The t_X grid of the labels (every {step_min:g} min) is not on the {sm}-minute sampling grid")
    if not datastore.has_series(settings.active_data_root, labels["config"].get("series", "")):
        errs.append(f"No SHARP {labels['config'].get('series')} data in the store")
    return errs


def validate_split(cfg: dict, dataset: dict, info: dict | None = None) -> list[str]:
    errs = []
    s = dataset["summary"]
    feats = s.get("feature_names", [])
    mode = cfg.get("FEATURE_SELECTION_MODE")
    if mode == "direct":
        sel = cfg.get("DIRECT_FEATURES") or []
        if not sel:
            errs.append("Select at least one feature")
        bad = [f for f in sel if f not in feats]
        if bad:
            errs.append(f"Features not in the dataset: {bad}")
    elif mode == "criterion":
        groups = (info or {}).get("feature_groups", {})
        crit = cfg.get("FEATURE_CRITERION")
        if crit not in groups:
            errs.append(f"Unknown FEATURE_CRITERION: {crit}")
        else:
            miss = [f for f in groups[crit] if f not in feats]
            if miss:
                errs.append(f"Group '{crit}' contains features missing from this dataset: {miss}. "
                            "Use direct selection.")
    else:
        errs.append("FEATURE_SELECTION_MODE: 'direct' or 'criterion'")
    ts = cfg.get("TIMESTEPS")
    _check_range(errs, "TIMESTEPS", ts, 1, s.get("timesteps"), integer=True)
    mm = cfg.get("MISSINGNESS_FILTER_MODE")
    if mm not in MISSINGNESS_MODES:
        errs.append(f"MISSINGNESS_FILTER_MODE: one of {MISSINGNESS_MODES}")
    if mm in ("single_window", "multi_window"):
        _check_range(errs, "MAX_NAN_RATIO", cfg.get("MAX_NAN_RATIO"), 0.0, 1.0)
    if mm in ("multi_window", "multi_window_empty"):
        tw = cfg.get("TIME_WINDOWS") or []
        if not tw or any((not isinstance(w, int)) or w < 1 or (ts and w > ts) for w in tw):
            errs.append(f"TIME_WINDOWS: integers between 1 and TIMESTEPS ({ts})")
    tfm = cfg.get("TEMPORAL_FILTER_MODE")
    if tfm not in TEMPORAL_FILTER_MODES:
        errs.append(f"TEMPORAL_FILTER_MODE: one of {TEMPORAL_FILTER_MODES}")
    try:
        if tfm:
            a, b = (cfg.get("TEMPORAL_WINDOW") or [None, None])
            da, db_ = _date(a, "TEMPORAL_WINDOW"), _date(b, "TEMPORAL_WINDOW")
            if not (da and db_) or da >= db_:
                errs.append("TEMPORAL_WINDOW: start < end")
        if cfg.get("MULTI_PERIOD_FILTER"):
            for a, b in cfg.get("PROBLEMATIC_PERIODS") or []:
                if _date(a, "PROBLEMATIC_PERIODS") >= _date(b, "PROBLEMATIC_PERIODS"):
                    errs.append(f"Invalid excluded period: {a} → {b}")
        strat = cfg.get("SPLITTING_STRATEGY")
        if strat not in SPLIT_STRATEGIES:
            errs.append(f"SPLITTING_STRATEGY: one of {SPLIT_STRATEGIES}")
        cut, win = cfg.get("TEST_CUTOFF_DATE"), cfg.get("TEST_TIME_WINDOW")
        if strat == "random" and (cut or win):
            errs.append("TEST_CUTOFF_DATE/TEST_TIME_WINDOW only apply to the chronological split")
        if cut and win:
            errs.append("Use TEST_CUTOFF_DATE or TEST_TIME_WINDOW, not both")
        lo, hi = _date(s.get("time_start"), "dataset start"), _date(s.get("time_end"), "dataset end")
        if cut:
            c = _date(cut, "TEST_CUTOFF_DATE")
            if lo and hi and not (lo < c < hi):
                errs.append(f"TEST_CUTOFF_DATE outside the dataset range ({lo} → {hi})")
        if win:
            a, b = _date(win[0], "TEST_TIME_WINDOW"), _date(win[1], "TEST_TIME_WINDOW")
            if a >= b:
                errs.append("TEST_TIME_WINDOW: start < end")
            elif lo and hi and (b <= lo or a >= hi):
                errs.append(f"TEST_TIME_WINDOW does not overlap the dataset ({lo} → {hi})")
    except ValueError as e:
        errs.append(str(e))
    if cfg.get("SPLITTING_STRATEGY") == "random":
        _check_range(errs, "TEST_SIZE", cfg.get("TEST_SIZE"), 0.05, 0.5)
    _check_range(errs, "N_FOLDS", cfg.get("N_FOLDS"), 2, 10, integer=True)
    _check_range(errs, "RANDOM_STATE", cfg.get("RANDOM_STATE"), 0, None, integer=True)
    if cfg.get("CMASK_FILTER"):
        if cfg.get("CMASK_STRATEGY") not in CMASK_STRATEGIES:
            errs.append(f"CMASK_STRATEGY: one of {CMASK_STRATEGIES}")
        _check_range(errs, "CMASK_THRESHOLD", cfg.get("CMASK_THRESHOLD"), 0, None)
        _check_range(errs, "CMASK_MAJORITY_THRESHOLD", cfg.get("CMASK_MAJORITY_THRESHOLD"), 0.0, 1.0)
        if "CMASK" not in feats:
            errs.append("The CMASK filter requires the CMASK feature in the dataset")
    if cfg.get("MULTI_CLASS"):
        errs.append("MULTI_CLASS is not supported by the package training (binary YES/NO)")
    return errs


def validate_model_config(cfg: dict, info: dict) -> list[str]:
    errs = []
    model = cfg.get("model")
    if model not in info.get("trainable_models", []):
        return [f"Model not available in the package: {model}"]
    defaults = info["model_configs"].get(model, {})
    for k, v in (cfg.get("model_params") or {}).items():
        d = defaults.get(k)
        if d is None or v is None:
            continue
        if isinstance(d, bool) and not isinstance(v, bool):
            errs.append(f"{k}: boolean expected")
        elif isinstance(d, int) and not isinstance(d, bool) and (not isinstance(v, int) or isinstance(v, bool)):
            errs.append(f"{k}: integer expected")
        elif isinstance(d, float) and not isinstance(v, (int, float)):
            errs.append(f"{k}: number expected")
        elif isinstance(v, (int, float)) and not isinstance(v, bool) and v < 0:
            errs.append(f"{k}: must be ≥ 0")
    t = cfg.get("training_config") or {}
    _check_range(errs, "batch_size", t.get("batch_size"), 1, 65536, integer=True)
    _check_range(errs, "learning_rate", t.get("learning_rate"), 1e-9, 10)
    _check_range(errs, "weight_decay", t.get("weight_decay"), 0, 10)
    _check_range(errs, "num_epochs", t.get("num_epochs"), 1, 10000, integer=True)
    _check_range(errs, "patience", t.get("patience"), 1, 10000, integer=True)
    sch = (t.get("lr_scheduler") or {}).get("type")
    if sch not in SCHEDULERS:
        errs.append(f"lr_scheduler.type: one of {SCHEDULERS}")
    if sch == "ReduceLROnPlateau":
        for k in ("mode", "factor", "patience", "min_lr"):
            if k not in t["lr_scheduler"]:
                errs.append(f"ReduceLROnPlateau requires '{k}'")
    th = t.get("threshold_optimization") or {}
    if th.get("enabled") and th.get("metric") not in THRESHOLD_METRICS:
        errs.append(f"threshold_optimization.metric: one of {THRESHOLD_METRICS}")
    es = t.get("early_stopping") or {}
    monitor = str(es.get("monitor")).lower()            # the package lower-cases it (training.train_model)
    if es and monitor not in MONITORS:
        errs.append(f"early_stopping.monitor: one of {MONITORS}")
    if es and es.get("mode") not in ("min", "max"):
        errs.append("early_stopping.mode: 'min' or 'max'")
    if monitor == "val_loss" and es.get("mode") == "max":
        errs.append("early_stopping: with monitor 'val_loss' the mode must be 'min'")
    if (t.get("loss_function") or {}).get("type") not in LOSS_TYPES:
        errs.append(f"loss_function.type: one of {LOSS_TYPES}")
    cal = t.get("calibration") or {}
    if cal.get("enabled") and cal.get("type", "platt") not in CALIBRATION_TYPES:
        errs.append(f"calibration.type: one of {CALIBRATION_TYPES}")
    reg = cfg.get("regularization_params")
    if model in info.get("sk_models", []):
        if not reg or any((not isinstance(x, (int, float))) or x <= 0 for x in reg):
            errs.append("regularization_params: list of values > 0 (grid explored in CV)")
    _check_range(errs, "seed", cfg.get("seed"), 0, 2**32 - 1, integer=True)
    return errs


def validate_training(cfg: dict, split: dict, model_cfg: dict, info: dict) -> list[str]:
    errs = []
    model = model_cfg["config"]["model"]
    ws = cfg.get("windows") or []
    wm = cfg.get("window_map") or {}
    if not ws:
        errs.append("Select at least one time window")
    for w in ws:
        steps = wm.get(w)
        if not isinstance(steps, int) or steps < 1:
            errs.append(f"Window {w}: invalid number of steps")
        elif steps > split["summary"]["timesteps"]:
            errs.append(f"Window {w} ({steps} steps) longer than the split TIMESTEPS ({split['summary']['timesteps']})")
    nf = split["summary"]["leakage"]["n_folds"]
    _check_range(errs, "n_folds", cfg.get("n_folds"), 1, nf, integer=True)
    files = split["summary"].get("files", {})
    for k in range(min(int(cfg.get("n_folds") or 0), nf)):
        for part in ("train", "val"):
            f = files.get(f"cv_{k}_{part}")
            if f and f["n_positive"] in (0, f["n_samples"]):
                errs.append(f"Fold {k} {part} contains a single class ({f['n_positive']} positives out of "
                            f"{f['n_samples']} samples): the model cannot be trained/validated. Use more "
                            "events (no subsample, or a larger one) or a more frequent flare class.")
    if cfg.get("device") not in ("auto", "cpu", "cuda"):
        errs.append("device: auto, cpu or cuda")
    if cfg.get("device") == "cuda" and not info.get("cuda_available"):
        errs.append("CUDA not available on this machine")
    lk = split["summary"]["leakage"]
    if lk["test_vs_trainval_shared_harps"] or lk["test_vs_trainval_shared_samples"]:
        errs.append(f"The split shares Active Regions/samples between test and train/val ({lk}): "
                    "risk of data leakage, create a split separated by HARP")
    return errs


def validate_inference(cfg: dict, training: dict, source: dict) -> list[str]:
    errs = []
    tw = training["summary"]["windows"]
    for w in cfg.get("windows") or []:
        if w not in tw:
            errs.append(f"Window {w} was not trained (available: {list(tw)})")
    if not cfg.get("windows"):
        errs.append("Select at least one window")
    if cfg.get("threshold_mode") not in ("cv", "fixed"):
        errs.append("threshold_mode: 'cv' or 'fixed'")
    if cfg.get("threshold_mode") == "fixed":
        _check_range(errs, "threshold", cfg.get("threshold"), 0.0, 1.0)
    if cfg.get("model_choice") == "fold":
        _check_range(errs, "fold", cfg.get("fold"), 0, training["summary"]["n_folds"] - 1, integer=True)
    if source["type"] == "split":
        names = list(source["summary"]["files"])
        if cfg.get("source_file", "test") not in names:
            errs.append(f"Unknown split file: {cfg.get('source_file')} (available {names})")
        elif cfg.get("source_file", "test") != "test" and source["id"] == training["parents"]["split"]:
            errs.append("You are using train/validation data from the same split as the model: "
                        "these are not unseen events (data leakage)")
    try:
        _date(cfg.get("time_start"), "time_start")
        _date(cfg.get("time_end"), "time_end")
    except ValueError as e:
        errs.append(str(e))
    return errs


# --------------------------------------------------------------------------- creation API
class Pipeline:
    def __init__(self, settings: Settings, store: ArtifactStore):
        self.settings = settings
        self.store = store
        self._info: dict | None = None

    @property
    def info(self) -> dict:
        if self._info is None:
            self._info = package_info(self.settings)
        return self._info

    def package_info_refresh(self) -> dict:
        self._info = package_info(self.settings, refresh=True)
        return self._info

    def _create(self, kind, name, config, parents, description="", derived_from=None, status="pending"):
        return self.store.create(kind, name, config, parents=parents, description=description,
                                 derived_from=derived_from, status=status, mode=self.settings.mode)

    def _raise(self, errs):
        if errs:
            raise StoreError("Invalid parameters:\n- " + "\n- ".join(errs))

    def create_labels(self, name, config, description="", derived_from=None):
        self._raise(validate_labels(config, self.settings))
        return self._create("labels", name, config, {}, description, derived_from)

    def create_dataset(self, name, labels_id, config, description="", derived_from=None):
        lab = _require_completed(self.store, labels_id, "labels", "labels")
        self._raise(validate_dataset(config, lab, self.settings))
        return self._create("dataset", name, config, {"labels": labels_id}, description, derived_from)

    def create_split(self, name, dataset_id, config, description="", derived_from=None):
        ds = _require_completed(self.store, dataset_id, "dataset", "dataset")
        self._raise(validate_split(config, ds, self.info))
        return self._create("split", name, config, {"dataset": dataset_id}, description, derived_from)

    def create_model_config(self, name, config, description="", derived_from=None):
        self._raise(validate_model_config(config, self.info))
        return self._create("model_config", name, config, {}, description, derived_from, status="completed")

    def create_training(self, name, split_id, model_config_id, config, description="", derived_from=None):
        sp = _require_completed(self.store, split_id, "split", "split")
        mc = _require_completed(self.store, model_config_id, "model_config", "model_config")
        self._raise(validate_training(config, sp, mc, self.info))
        return self._create("training", name, config, {"split": split_id, "model_config": model_config_id},
                            description, derived_from)

    def create_inference(self, name, training_id, source_id, config, description="", derived_from=None):
        tr = _require_completed(self.store, training_id, "training", "training")
        src = _require_completed(self.store, source_id, ["split", "dataset"], "source")
        self._raise(validate_inference(config, tr, src))
        return self._create("inference", name, config, {"training": training_id, "source": source_id},
                            description, derived_from)

    # ------------------------------------------------------------------ variants & reproduction
    def variant(self, artifact_id: str, config: dict | None = None, name: str | None = None,
                parents: dict | None = None, description: str = "") -> dict:
        """New artifact with modified config/parents; the original is never altered."""
        m = self.store.get(artifact_id)
        cfg = copy.deepcopy(config if config is not None else m["config"])
        par = dict(parents if parents is not None else m["parents"])
        name = name or f"{m['name']} (variant)"
        kind = m["type"]
        creators = {
            "labels": lambda: self.create_labels(name, cfg, description, artifact_id),
            "dataset": lambda: self.create_dataset(name, par["labels"], cfg, description, artifact_id),
            "split": lambda: self.create_split(name, par["dataset"], cfg, description, artifact_id),
            "model_config": lambda: self.create_model_config(name, cfg, description, artifact_id),
            "training": lambda: self.create_training(name, par["split"], par["model_config"], cfg, description, artifact_id),
            "inference": lambda: self.create_inference(name, par["training"], par["source"], cfg, description, artifact_id),
        }
        return creators[kind]()

    def reproduce_step(self, artifact_id: str) -> dict:
        m = self.store.get(artifact_id)
        return self.variant(artifact_id, name=f"{m['name']} (reproduction)",
                            description=f"Reproduction of {artifact_id} with the saved configuration and inputs")

    def reproduce_chain(self, artifact_id: str) -> list[str]:
        """Re-create the whole chain (ancestors + artifact) as *new* artifacts, wired to each other.
        Returns the ids of the new runnable artifacts in execution order (model configs are static)."""
        order = self.store.ancestors(artifact_id) + [artifact_id]
        mapping: dict[str, str] = {}
        runnable = []
        for old in order:
            m = self.store.get(old)
            parents = {r: mapping.get(p, p) for r, p in m["parents"].items()}
            new = self.store.create(m["type"], f"{m['name']} (chain reproduction)", copy.deepcopy(m["config"]),
                                    parents=parents, derived_from=old, mode=self.settings.mode,
                                    status="completed" if m["type"] == "model_config" else "pending",
                                    description=f"Reproduction of the chain of {artifact_id}")
            mapping[old] = new["id"]
            if m["type"] != "model_config":
                runnable.append(new["id"])
        return runnable

    def resume_plan(self, artifact_id: str) -> list[str]:
        """Steps still to run to obtain ``artifact_id`` (restart from the last valid artifact)."""
        order = self.store.ancestors(artifact_id) + [artifact_id]
        return [a for a in order if self.store.get(a)["type"] != "model_config"
                and self.store.get(a)["status"] != "completed"]

    # ------------------------------------------------------------------ comparison & export
    def diff(self, a: str, b: str) -> list[dict]:
        ma, mb = self.store.get(a), self.store.get(b)
        fa, fb = _flatten({"config": ma["config"], "parents": ma["parents"]}), \
            _flatten({"config": mb["config"], "parents": mb["parents"]})
        rows = []
        for k in sorted(set(fa) | set(fb)):
            if fa.get(k) != fb.get(k):
                rows.append({"key": k, a: fa.get(k, "—"), b: fb.get(k, "—")})
        return rows

    def export(self, artifact_id: str) -> dict:
        chain = self.store.ancestors(artifact_id) + [artifact_id]
        return {"exported_at": dt.datetime.now().isoformat(timespec="seconds"),
                "artifact": artifact_id,
                "chain": [{k: v for k, v in self.store.get(x).items() if k != "history"} for x in chain]}

    def lineage_dot(self, artifact_id: str | None = None) -> str:
        ids = (self.store.ancestors(artifact_id) + [artifact_id] + self.store.descendants(artifact_id)) \
            if artifact_id else [m["id"] for m in self.store.list()]
        colors = {"completed": "#2e7d32", "failed": "#c62828", "running": "#1565c0", "pending": "#9e9e9e",
                  "cancelled": "#6d4c41", "draft": "#9e9e9e"}
        lines = ["digraph G {", 'rankdir=LR; node [shape=box, style="rounded,filled", fontname="Helvetica", '
                 'fontsize=10, fillcolor="#ffffff"];']
        for aid in ids:
            if not self.store.exists(aid):
                continue
            m = self.store.get(aid)
            pen = "3" if aid == artifact_id else "1.2"
            label = f"{m['type']}\\n{m['name'][:32]}\\n{aid}"
            lines.append(f'"{aid}" [label="{label}", color="{colors.get(m["status"], "#000")}", penwidth={pen}];')
            for role, pid in m["parents"].items():
                if pid in ids:
                    lines.append(f'"{pid}" -> "{aid}" [label="{role}", fontsize=8];')
            if m.get("derived_from") in ids:
                lines.append(f'"{m["derived_from"]}" -> "{aid}" [style=dashed, color="#888888", label="variant"];')
        lines.append("}")
        return "\n".join(lines)


def _flatten(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(_flatten(v, f"{prefix}{k}."))
    else:
        out[prefix[:-1]] = json.dumps(d) if isinstance(d, (list, tuple)) else d
    return out
