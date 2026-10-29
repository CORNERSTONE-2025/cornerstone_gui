"""Pipeline steps executed by the runner. Each step calls the real CORNERSTONE API.

step(ctx) -> summary dict (stored in the artifact manifest). Files written in ``ctx.dir``
are registered (size + sha256) by the runner after the step returns.
"""
from __future__ import annotations

import glob
import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import h5info
from .. import metrics as gm
from ..store import ArtifactStore
from . import adapters

DEFAULT_WINDOW_DESCRIPTIONS = {"3h": "3-Hour Window", "6h": "6-Hour Window",
                               "12h": "12-Hour Window", "24h": "24-Hour Window"}
SPLIT_PREFIX = "split"          # EXPERIMENT_NAME passed to create_dataset_splits => data prefix


@dataclass
class Ctx:
    store: ArtifactStore
    m: dict
    db: str
    mode: str
    cornerstone_root: str

    @property
    def dir(self) -> Path:
        return self.store.path(self.m["id"])

    @property
    def config(self) -> dict:
        return self.m["config"]

    def parent(self, role: str) -> dict:
        return self.store.get(self.m["parents"][role])

    def parent_dir(self, role: str) -> Path:
        return self.store.path(self.m["parents"][role])


def _cornerstone(ctx: Ctx, db: str | None = None, output_folder: Path | None = None):
    from cornerstone import Cornerstone
    out = output_folder or (ctx.dir / "_cornerstone_tmp")
    return Cornerstone({"local_db_path": db or ctx.db}, output_folder=str(out), project_name="cornerstone-gui")


def _rmdir_if_empty(p: Path) -> None:
    try:
        p.rmdir()
    except OSError:
        pass


# ======================================================================================
# 1. Labeling
# ======================================================================================
def summarize_labels(db: Path, table: str) -> dict:
    import pandas as pd
    con = sqlite3.connect(db)
    try:
        df = pd.read_sql_query(f'SELECT t_X, HARPn, label, label_u FROM "{table}"', con)
    finally:
        con.close()
    t = pd.to_datetime(df["t_X"], errors="coerce")
    yes = df["label"] == "YES"
    per_year = df.assign(year=t.dt.year, yes=yes).groupby("year")["yes"].agg(["sum", "count"])
    return {
        "table": table,
        "n_rows": int(len(df)),
        "n_active_regions": int(df["HARPn"].nunique()),
        "t_X_start": str(t.min()), "t_X_end": str(t.max()),
        "label_counts": {str(k): int(v) for k, v in df["label"].value_counts().items()},
        "label_u_counts": {str(k): int(v) for k, v in df["label_u"].value_counts().items()},
        "positive_rate": float(yes.mean()) if len(df) else 0.0,
        "per_year": {str(int(y)): {"YES": int(r["sum"]), "n": int(r["count"])} for y, r in per_year.iterrows()},
    }


def step_labels(ctx: Ctx) -> dict:
    c = ctx.config
    shtyp, cls, delta = c["table_type"], c["label_type"], int(c["delta_hours"])
    table = adapters.label_table_name(shtyp, cls, delta)
    out_db = ctx.dir / "labels.sqlite"
    if c["source"] == "existing":
        print(f"[cs-gui] snapshot of existing label table {table} (main DB opened read-only)")
        info = {"copied_rows": adapters.snapshot_label_table(ctx.db, str(out_db), table)}
    else:
        print(f"[cs-gui] preparing a private labeling DB from {ctx.db} (read-only)")
        info = adapters.build_labeling_db(ctx.db, str(out_db), shtyp)
        cs = _cornerstone(ctx, db=str(out_db))
        cs.splitting_labeling(shtyp=shtyp, class0=cls, delta_hours=str(delta), db_type="sqlite", save_csv=False)
        con = sqlite3.connect(out_db)
        con.execute(f'DROP TABLE IF EXISTS "SHARPC_all_{shtyp}"')     # helper table only
        con.commit()
        con.execute("VACUUM")
        con.close()
        _rmdir_if_empty(ctx.dir / "_cornerstone_tmp")
    summary = summarize_labels(out_db, table)
    summary.update(info)
    summary["labels_db"] = "labels.sqlite"
    # open, interoperable copy
    import pandas as pd
    con = sqlite3.connect(out_db)
    pd.read_sql_query(f'SELECT * FROM "{table}"', con).to_csv(ctx.dir / "labels.csv", index=False)
    con.close()
    return summary


# ======================================================================================
# 2. HDF5 extraction
# ======================================================================================
def step_dataset(ctx: Ctx) -> dict:
    lab = ctx.parent("labels")
    lc, c = lab["config"], ctx.config
    shtyp, cls = lc["table_type"], lc["label_type"]
    expected = adapters.label_table_name(shtyp, cls, c["window_hours"])   # name built by the package
    shim = adapters.readonly_sqlite_module(ctx.db, labels_db=str(ctx.parent_dir("labels") / "labels.sqlite"),
                                           view_map={expected: lab["summary"]["table"]})
    adapters.patch_cornerstone_sqlite(shim)
    cs = _cornerstone(ctx, output_folder=ctx.dir)
    cs.create_hdf5_from_db(
        label_type=cls, data_version=c["data_version"], table_type=shtyp,
        window_hours=int(c["window_hours"]), buffer_minutes=int(c["buffer_minutes"]),
        sampling_interval=int(c["sampling_interval"]),
        test_mode=bool(c.get("subsample")), test_size=int(c.get("subsample_size") or 0),
        random_seed=int(c.get("random_seed", 42)), db_type="sqlite",
    )
    files = sorted(ctx.dir.glob("*.h5"))
    if not files:
        raise RuntimeError("create_hdf5_from_db did not produce any HDF5 file")
    h5 = files[0]
    s = h5info.summarize(str(h5))
    if s["n_samples"] == 0:
        raise RuntimeError("The extracted dataset is empty: no event has data in the database")
    s["h5_file"] = h5.name
    s["label_key"] = f"{cls}_delta{c['window_hours']}"
    s["label_horizon_hours"] = int(lc["delta_hours"])
    s["n_label_events"] = lab["summary"]["n_rows"] if not c.get("subsample") else int(c["subsample_size"])
    miss = ctx.dir / "missing_data_log.csv"
    if miss.exists():
        import pandas as pd
        md = pd.read_csv(miss)
        s["n_excluded_events"] = int(len(md))
        s["excluded_reasons"] = {str(k): int(v) for k, v in md["reason"].value_counts().items()} if "reason" in md else {}
    return s


# ======================================================================================
# 3. Preprocessing + split
# ======================================================================================
LABEL_SHORT = {"Cplus": "C+", "Mplus": "M+", "Xplus": "X+"}


def _split_files(split_dir: Path) -> dict[str, Path]:
    out = {}
    for p in sorted((split_dir / "splits").glob("*.h5")):
        out[p.stem[len(SPLIT_PREFIX) + 1:]] = p        # test, cv_0_train, cv_0_val, ...
    return out


def split_leakage_report(files: dict[str, Path]) -> dict:
    """HARP-level and temporal separation between test and train/val, and within each fold."""
    test_h = h5info.harps(str(files["test"])) if "test" in files else set()
    trainval_h, fold_overlaps = set(), {}
    k = 0
    while f"cv_{k}_train" in files:
        tr, va = h5info.harps(str(files[f"cv_{k}_train"])), h5info.harps(str(files[f"cv_{k}_val"]))
        fold_overlaps[str(k)] = len(tr & va)
        trainval_h |= tr | va
        k += 1
    test_keys = h5info.sample_keys(str(files["test"])) if "test" in files else set()
    tv_keys = set()
    for name, p in files.items():
        if name != "test":
            tv_keys |= h5info.sample_keys(str(p))
    return {
        "test_vs_trainval_shared_harps": len(test_h & trainval_h),
        "test_vs_trainval_shared_samples": len(test_keys & tv_keys),
        "train_vs_val_shared_harps_per_fold": fold_overlaps,
        "n_folds": k,
    }


def step_split(ctx: Ctx) -> dict:
    ds = ctx.parent("dataset")
    cfg = dict(ctx.config)
    cfg.update({
        "INPUT_H5_FILE": str(ctx.parent_dir("dataset") / ds["summary"]["h5_file"]),
        "OUTPUT_BASE_DIR": str(ctx.dir),
        "EXPERIMENT_NAME": SPLIT_PREFIX,
        "LABEL_TYPE": ds["summary"]["label_key"],
    })
    (ctx.dir / "split_config.json").write_text(json.dumps(cfg, indent=2))
    cs = _cornerstone(ctx)
    try:
        cs.create_dataset_splits(cfg)
    except TypeError as e:
        if "NoneType" in str(e) and cfg.get("SPLITTING_STRATEGY") == "chronological":
            raise RuntimeError("The chronological split produced no test set (no sample in the window/after "
                               "the cutoff date). Change TEST_CUTOFF_DATE or TEST_TIME_WINDOW.") from e
        raise
    _rmdir_if_empty(ctx.dir / "_cornerstone_tmp")
    split_dir = ctx.dir / SPLIT_PREFIX
    files = _split_files(split_dir)
    if "test" not in files:
        raise RuntimeError("No split file was produced")
    per_file = {}
    for name, p in files.items():
        s = h5info.summarize(str(p), max_rows_for_nan=2000)
        per_file[name] = {k: s[k] for k in ("n_samples", "n_active_regions", "n_positive", "positive_rate",
                                            "time_start", "time_end", "class_counts", "class_counts_u")}
    first = h5info.summarize(str(files["test"]), max_rows_for_nan=1)
    kept = per_file["test"]["n_samples"] + per_file.get("cv_0_train", {}).get("n_samples", 0) \
        + per_file.get("cv_0_val", {}).get("n_samples", 0)
    label_type = ds["summary"]["label_key"].split("_")[0]
    return {
        "splits_dir": f"{SPLIT_PREFIX}/splits",
        "data_prefix": SPLIT_PREFIX,
        "files": per_file,
        "feature_names": first["feature_names"],
        "n_features": first["n_features"],
        "timesteps": first["timesteps"],
        "label_key": ds["summary"]["label_key"],
        "task_name": LABEL_SHORT.get(label_type, label_type),
        "n_samples_input": ds["summary"]["n_samples"],
        "n_samples_kept": kept,
        "n_samples_excluded": ds["summary"]["n_samples"] - kept,
        "leakage": split_leakage_report(files),
    }


# ======================================================================================
# 4. Training
# ======================================================================================
def _training_setup(ctx_store: ArtifactStore, training_m: dict, train_cfg_path: Path):
    """Register dataset, apply model overrides, set device and seed. Shared by training/inference."""
    split = ctx_store.get(training_m["parents"]["split"])
    mc = ctx_store.get(training_m["parents"]["model_config"])["config"]
    tc = json.loads(train_cfg_path.read_text())
    adapters.register_dataset(tc["CURRENT_DATASET"], split["summary"]["task_name"], SPLIT_PREFIX,
                              split["summary"]["n_features"],
                              str(ctx_store.path(split["id"]) / split["summary"]["splits_dir"]))
    adapters.apply_model_overrides(mc["model"], mc.get("model_params", {}), mc.get("regularization_params"))
    device = adapters.set_device(training_m["config"].get("device", "auto"))
    adapters.set_seeds(int(mc.get("seed", 42)))
    from cornerstone.ml.utils import config as cfg
    cfg.set_config_path(str(train_cfg_path))
    return split, mc, tc, device


def _path_manager(results_dir: Path, task_name: str, model: str):
    from cornerstone.ml.utils.path_manager import PathManager
    return PathManager(task_name=task_name, model_type=model, dataset_name=SPLIT_PREFIX, base_dir=str(results_dir))


def normalize_training_metrics(model: str, metrics: dict, is_sk: bool) -> list[dict]:
    """Flatten the package metrics file (PT and SK formats differ) into table rows."""
    rows = []
    for w, data in metrics.items():
        if is_sk:
            by_c = {e["C"]: e for e in data}
            mean_val = {C: np.mean([f["val_metrics"]["TSS"] for f in e["folds"]]) for C, e in by_c.items()}
            best_c = max(mean_val, key=mean_val.get)
            best_fold = int(np.argmax([f["val_metrics"]["TSS"] for f in by_c[best_c]["folds"]]))
            for C, e in by_c.items():
                for f in e["folds"]:
                    rows.append({"window": w, "C": C, "fold": f["fold"], "threshold": f.get("threshold"),
                                 **{f"val_{k}": v for k, v in f["val_metrics"].items()},
                                 **{f"test_{k}": v for k, v in f["test_metrics"].items()},
                                 "selected": C == best_c and f["fold"] == best_fold})
        else:
            for f in data["folds"]:
                vm = f.get("val_metrics_best_epoch") or {}
                tm = f.get("test_metrics") or {}
                rows.append({"window": w, "fold": f["fold"], "threshold": f.get("best_threshold"),
                             "best_epoch": f.get("best_epoch"),
                             **{f"val_{k}": v for k, v in vm.items()},
                             **{f"test_{k}": v for k, v in tm.items() if k in ("TSS", "HSS", "F1", "AUC", "MCC")},
                             "selected": f["fold"] == data["best_fold"]})
    return rows


def step_training(ctx: Ctx) -> dict:
    split, mc_art = ctx.parent("split"), ctx.parent("model_config")
    mc = mc_art["config"]
    model = mc["model"]
    windows = ctx.config["windows"]
    window_map = {w: int(ctx.config["window_map"][w]) for w in windows}
    if max(window_map.values()) > split["summary"]["timesteps"]:
        raise RuntimeError(f"Window longer ({max(window_map.values())} steps) than the dataset timesteps "
                           f"({split['summary']['timesteps']})")
    training_cfg = dict(mc["training_config"])
    training_cfg["n_folds"] = int(ctx.config["n_folds"])
    results = ctx.dir / "results"
    train_config = {
        "CURRENT_DATASET": f"CSGUI_{split['id']}",
        "CURRENT_MODELS": [model],
        "OUTPUT_CONFIG": {"results_dir": str(results), "save_models": True,
                          "plot_metrics": ["TSS", "HSS", "F1", "MCC"], "plot_training_curves": True,
                          "plot_metric_curves": True, "plot_boxplot": True, "plot_test_metrics": True},
        "TRAINING_CONFIG": training_cfg,
        "DATA_CONFIG": {"window_map": window_map,
                        "window_descriptions": {w: DEFAULT_WINDOW_DESCRIPTIONS.get(w, w) for w in windows}},
    }
    cfg_path = ctx.dir / "train_config.json"
    cfg_path.write_text(json.dumps(train_config, indent=2))
    _, _, _, device = _training_setup(ctx.store, ctx.m, cfg_path)
    print(f"[cs-gui] training {model} on {device}, windows={window_map}, seed={mc.get('seed', 42)}")
    t0 = time.time()
    _cornerstone(ctx).train_models(train_config_path=str(cfg_path))
    elapsed = time.time() - t0
    _rmdir_if_empty(ctx.dir / "_cornerstone_tmp")

    from cornerstone.ml.utils.training import PT_MODELS, SK_MODELS
    task = split["summary"]["task_name"]
    pm = _path_manager(results, task, model)
    mfiles = glob.glob(os.path.join(pm.task_dir, "metrics", f"*_metrics_{pm.model_type}.json"))
    if not mfiles:
        raise RuntimeError("Training did not produce the expected metrics file")
    metrics = json.loads(Path(mfiles[0]).read_text())
    adapters_used = []
    if model in PT_MODELS:
        best = adapters.make_best_pt_models(pm, windows, metrics)
        adapters_used.append({"name": "best_pt_from_best_fold", "detail": best})
    rows = normalize_training_metrics(model, metrics, is_sk=model in SK_MODELS)
    return {
        "model": model, "family": "sklearn" if model in SK_MODELS else "pytorch", "device": device,
        "windows": window_map, "n_folds": training_cfg["n_folds"],
        "task_dir": os.path.relpath(pm.task_dir, ctx.dir), "metrics_file": os.path.relpath(mfiles[0], ctx.dir),
        "train_config": "train_config.json", "rows": rows, "training_seconds": round(elapsed, 1),
        "adapters": adapters_used,
        "notes": ["test_* = metrics computed by the package on the split test set during training "
                  "(per fold). Use an Inference artifact for the evaluation."],
    }


# ======================================================================================
# 5. Inference
# ======================================================================================
def _write_inference_input(src: Path, dst: Path, features: list[str], harp_filter: list[int],
                           t_start: str | None, t_end: str | None) -> dict:
    import h5py
    with h5py.File(src, "r") as f:
        names = h5info.feature_names(str(src))
        missing = [x for x in features if x not in names]
        if missing:
            raise RuntimeError(f"The inference dataset does not contain the model features: {missing}")
        cols = [names.index(x) for x in features]
        harp = f["HARPNUM"][:]
        ts = f["timestamps"][:]
        it = h5info.issue_times(ts)
        keep = np.ones(len(harp), bool)
        if harp_filter:
            keep &= np.isin(harp, np.asarray(harp_filter))
        if t_start:
            keep &= it >= t_start
        if t_end:
            keep &= it <= t_end
        idx = np.flatnonzero(keep)
        if idx.size == 0:
            raise RuntimeError("No sample matches the inference filters (HARP / time range)")
        yu_key = "Y_u" if "Y_u" in f else next((k for k in f.keys() if k.startswith("label_u_")), None)
        with h5py.File(dst, "w") as g:
            g.create_dataset("X", data=f["X"][idx][:, :, cols])
            g.create_dataset("Y", data=f["Y"][idx])
            if yu_key:
                g.create_dataset("Y_u", data=f[yu_key][idx])
            g.create_dataset("HARPNUM", data=harp[idx])
            g.create_dataset("timestamps", data=ts[idx])
            g.attrs["feature_names"] = ",".join(features)
    return {"n_input": int(len(harp)), "n_selected": int(idx.size)}


def _valid_meta(path: Path, steps: int) -> dict:
    """Metadata aligned with FlareHDF5Dataset (which drops samples empty in the last ``steps``)."""
    import h5py
    with h5py.File(path, "r") as f:
        X = f["X"][:, -steps:, :]
        valid = (~np.isnan(X)).sum(axis=(1, 2)) > 0
        y = np.array([v.decode() for v in f["Y"][:]])
        yu = np.array([v.decode() for v in f["Y_u"][:]]) if "Y_u" in f else np.array([""] * len(y))
        return {"HARPNUM": f["HARPNUM"][:][valid], "t_issue": h5info.issue_times(f["timestamps"][:])[valid],
                "label": y[valid], "label_u": yu[valid]}


def step_inference(ctx: Ctx) -> dict:
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    from cornerstone.ml.utils.calibration import sigmoid
    from cornerstone.ml.utils.datasets import FlareHDF5Dataset
    from cornerstone.ml.utils.eval_io import collect_logits
    from cornerstone.ml.utils.metrics import compute_metrics
    from cornerstone.ml.utils.training import SK_MODELS

    c = ctx.config
    tr = ctx.parent("training")
    tr_dir = ctx.parent_dir("training")
    split, mc, tc, device = _training_setup(ctx.store, tr, tr_dir / "train_config.json")
    model = mc["model"]
    features = split["summary"]["feature_names"]
    pm = _path_manager(tr_dir / "results", split["summary"]["task_name"], model)
    pkg_metrics = json.loads((tr_dir / tr["summary"]["metrics_file"]).read_text())

    src = ctx.parent("source")
    src_dir = ctx.parent_dir("source")
    if src["type"] == "split":
        src_file = src_dir / src["summary"]["splits_dir"] / f"{SPLIT_PREFIX}_{c.get('source_file', 'test')}.h5"
    elif src["type"] == "dataset":
        src_file = src_dir / src["summary"]["h5_file"]
    else:
        raise RuntimeError(f"Unsupported inference source: {src['type']}")
    inp = ctx.dir / "inference_input.h5"
    sel = _write_inference_input(src_file, inp, features, c.get("harp_filter") or [],
                                 c.get("time_start"), c.get("time_end"))

    # separation from the data used for training / model selection (train + val of every fold)
    trainval_keys = set()
    split_dir = ctx.store.path(split["id"]) / split["summary"]["splits_dir"]
    for p in split_dir.glob(f"{SPLIT_PREFIX}_cv_*_*.h5"):
        trainval_keys |= h5info.sample_keys(str(p))
    trainval_harps = {h for h, _ in trainval_keys}

    is_sk = model in SK_MODELS
    windows = c.get("windows") or list(tr["summary"]["windows"])
    all_rows, per_window = [], {}
    for w in windows:
        steps = int(tr["summary"]["windows"][w])
        meta = _valid_meta(inp, steps)
        fold = None if c.get("model_choice", "best") == "best" else int(c["fold"])
        ds = FlareHDF5Dataset(str(inp), timesteps=steps)
        if is_sk:
            import joblib
            path = pm.get_model_path(w, best=True, model_type="SK") if fold is None else \
                pm.get_model_path(w, fold=fold, model_type="SK")
            est = joblib.load(path)
            p = est.predict_proba(ds.X.numpy(), mask=ds.mask.numpy())[:, 1]
            y = ds.Y.numpy().astype(int)
            cv_thr, cv_src = getattr(est, "threshold_", None), "model.threshold_ (optimized on validation)"
        else:
            from cornerstone.ml.model_configs import MODEL_CONFIGS
            from cornerstone.ml.utils.model_manager import ModelManager
            cfgm = MODEL_CONFIGS.get(model, {})
            net = ModelManager(model).create_model(model, **{**cfgm, **({"seq_len": steps} if "seq_len" in cfgm else {})})
            k = pkg_metrics[w]["best_fold"] if fold is None else fold
            path = pm.get_model_path(w, fold=k, model_type="PT")
            net.load_state_dict(torch.load(path, map_location=device))
            z, y, _, _ = collect_logits(net, DataLoader(ds, batch_size=256), device=device)
            p, y = sigmoid(z), y.astype(int)
            cv_thr = pkg_metrics[w]["folds"][k].get("best_threshold")
            cv_src = f"best_threshold of fold {k} (optimized on validation, uncalibrated sigmoid probabilities)"
        if len(p) != len(meta["HARPNUM"]):
            raise RuntimeError("Predictions and sample metadata are misaligned")
        if c.get("threshold_mode", "cv") == "cv":
            if cv_thr is None:
                raise RuntimeError("No CV-optimized threshold available for this model: use a fixed threshold")
            thr, thr_src = float(cv_thr), cv_src
        else:
            thr, thr_src = float(c["threshold"]), "fixed threshold chosen by the user"
        overlap = np.array([(int(h), str(t)) in trainval_keys for h, t in zip(meta["HARPNUM"], meta["t_issue"])])
        harp_overlap = np.isin(meta["HARPNUM"], list(trainval_harps))
        df = pd.DataFrame({"window": w, "HARPNUM": meta["HARPNUM"], "t_issue": meta["t_issue"],
                           "label": meta["label"], "label_u": meta["label_u"], "y_true": y,
                           "score": p, "threshold": thr, "y_pred": (p >= thr).astype(int),
                           "in_trainval_sample": overlap, "in_trainval_harp": harp_overlap,
                           "model_file": os.path.basename(path)})
        all_rows.append(df)
        evalmask = ~(df["in_trainval_harp"].values) if c.get("exclude_overlap", True) else np.ones(len(df), bool)
        ye, pe = y[evalmask], p[evalmask]
        entry = {"threshold": thr, "threshold_source": thr_src, "n_predictions": int(len(df)),
                 "n_evaluated": int(evalmask.sum()), "n_excluded_trainval_harp": int((~evalmask).sum()),
                 "n_trainval_samples": int(overlap.sum()), "model_file": os.path.basename(path)}
        if evalmask.sum() and 0 < ye.sum() < len(ye):
            entry["package_metrics"] = {k: float(v) for k, v in compute_metrics(ye, pe, threshold=thr).items()}
            entry["metrics"] = gm.point_metrics(ye, pe, thr)
            boot = c.get("bootstrap", {"n": 1000, "cluster": True})
            if boot.get("n", 0):
                entry["ci"] = gm.bootstrap_ci(ye, pe, thr, n_boot=int(boot["n"]),
                                              group=meta["HARPNUM"][evalmask] if boot.get("cluster", True) else None)
        else:
            entry["metrics"] = None
            entry["note"] = ("Metrics not computable: evaluable samples of both classes are needed "
                             "(no ground truth or a single class).")
        per_window[w] = entry
    pd.concat(all_rows).to_csv(ctx.dir / "predictions.csv", index=False)
    (ctx.dir / "metrics.json").write_text(json.dumps(per_window, indent=2, default=float))
    inp.unlink(missing_ok=True)     # reconstructible from source artifact + config; predictions keep the keys
    return {"model": model, "family": "sklearn" if is_sk else "pytorch", "device": device,
            "source_file": src_file.name, "selection": sel, "windows": per_window,
            "trainval_samples_known": len(trainval_keys)}


STEPS = {"labels": step_labels, "dataset": step_dataset, "split": step_split,
         "training": step_training, "inference": step_inference}
