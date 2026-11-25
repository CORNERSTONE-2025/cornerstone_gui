"""Nightly test campaign of the CORNERSTONE pipeline (runs inside the GUI Docker image).

    python -m scripts.nightly.campaign --out /nightly --deadline 07:30 [--quick] [--only P5,P7]

Every test is recorded in ``<out>/results.json`` as soon as it ends (a run stopped half-way still
leaves its results), with status pass / warn / fail / skip, duration, measured values and the log
file. ``<out>/summary.md`` is the automatic table of the results; figures go in ``<out>/figures``.
Tests are ordered by priority: before each one the remaining time to the deadline is compared with
its estimated duration, and tests that would not end in time are skipped (status ``skip``).

The data store (``/data/datastore``) is only read: the update test works on a copy, and the last
test checks that the store files are unchanged. Artifacts go in ``<out>/workspace``.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CORNERSTONE_ROOT = os.environ.get("CS_GUI_CORNERSTONE_ROOT", "/opt/cornerstone")
sys.path[:0] = [str(ROOT), CORNERSTONE_ROOT]
os.environ.setdefault("MPLBACKEND", "Agg")

from cs_gui.jobs import JobManager  # noqa: E402
from cs_gui.pipeline import (DEFAULT_TRAINING_CONFIG, Pipeline, default_label_config,  # noqa: E402
                             default_split_config, validate_dataset, validate_labels, validate_split,
                             validate_training)
from cs_gui.settings import Settings  # noqa: E402
from cs_gui.store import ArtifactStore, StoreError  # noqa: E402

DATA_ROOT = os.environ.get("CS_GUI_DATA_ROOT", "/data/datastore")
# former SQLite database (optional): as configured for the GUI, or in db/ of the data folder
LEGACY_DB = next((p for p in (os.environ.get("CS_GUI_LEGACY_DB"), "/data/cornerstone.sqlite", "/data/db/cornerstone.sqlite")
                  if p and Path(p).exists()), "/data/cornerstone.sqlite")
PYTHON = sys.executable

# Reference: InceptionTime, DEF M+ 24 h, clean features, random split, 5 folds, 100 epochs
# (GUI experiment of 2026-11-12, store of 2026-11-12): mean test TSS of the 5 folds per window.
REFERENCE_INCEPTIONTIME_TEST_TSS = {"3h": 0.610, "6h": 0.588, "12h": 0.590, "24h": 0.569}
LEGACY_AGREEMENT = {"NRT": 0.9971, "DEF": 0.9784}      # share of identical labels, M+, Nov 2026
CLEAN_FEATURES = ["LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX", "USFLUX_N1", "MEANGBT", "MEANGBZ_N2",
                  "MEANGBH_N2", "MEANJZD", "TOTUSJZ_N1", "MEANJZH", "TOTUSJH_N1", "ABSNJZH", "MEANPOT",
                  "SHRGT45", "R_VALUE", "CMASK"]
NN_MODELS = ["LSTM", "INCEPTIONTIME", "TRANSFORMER", "MINIROCKET"]


class Skip(Exception):
    """The test cannot run here (missing input): recorded as ``skip``."""


class Campaign:
    def __init__(self, out: Path, deadline: dt.datetime, quick: bool, only: set[str] | None):
        self.out, self.deadline, self.quick, self.only = out, deadline, quick, only
        for d in ("logs", "figures", "workspace"):
            (out / d).mkdir(parents=True, exist_ok=True)
        self.results: list[dict] = []
        self.state: dict = {}                       # artifacts and values shared between tests
        self.started = dt.datetime.now()
        self.settings = Settings(cornerstone_root=CORNERSTONE_ROOT, cornerstone_python=PYTHON,
                                 data_root=DATA_ROOT, workspace=str(out / "workspace"),
                                 demo_workspace=str(out / "workspace_demo"), mode="real", min_free_disk_gb=2.0)
        self.store = ArtifactStore(self.settings.active_workspace)
        self.jobs = JobManager(self.settings, self.store)
        self.pl = Pipeline(self.settings, self.store)

    # ------------------------------------------------------------------ bookkeeping
    def remaining_min(self) -> float:
        return (self.deadline - dt.datetime.now()).total_seconds() / 60

    def save(self) -> None:
        doc = {"started_at": self.started.isoformat(timespec="seconds"),
               "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
               "deadline": self.deadline.isoformat(timespec="minutes"), "quick": self.quick,
               "results": self.results}
        tmp = self.out / "results.json.tmp"
        tmp.write_text(json.dumps(doc, indent=2, default=str))
        tmp.replace(self.out / "results.json")
        write_summary(self.out, doc)

    def run(self, tests) -> None:
        for t in tests:
            tid, phase, title, est = t.tid, t.phase, t.title, t.est(self)
            if self.only and tid not in self.only and phase not in self.only:
                continue
            rec = {"id": tid, "phase": phase, "title": title, "estimated_min": est,
                   "started_at": dt.datetime.now().isoformat(timespec="seconds")}
            log = self.out / "logs" / f"{tid}.log"
            if self.remaining_min() < est:
                rec.update(status="skip", message=f"Not enough time before the deadline "
                           f"({self.remaining_min():.0f} min left, about {est:.0f} min needed)")
            else:
                print(f"[{dt.datetime.now():%H:%M:%S}] {tid} {title} (≈{est:.0f} min)", flush=True)
                t0 = time.time()
                try:
                    status, message, details = t(self, log)
                    rec.update(status=status, message=message, details=details)
                except Skip as e:
                    rec.update(status="skip", message=str(e))
                except Exception as e:  # noqa: BLE001 - every failure is a result
                    rec.update(status="fail", message=f"{type(e).__name__}: {e}",
                               traceback=traceback.format_exc()[-6000:])
                rec["duration_min"] = round((time.time() - t0) / 60, 2)
            rec["log"] = f"logs/{log.name}" if log.exists() else None
            print(f"    -> {rec['status']}: {str(rec.get('message'))[:300]}", flush=True)
            self.results.append(rec)
            self.save()

    # ------------------------------------------------------------------ helpers for the GUI pipeline
    def run_artifact(self, aid: str, timeout_min: float) -> dict:
        """Run one artifact through the GUI job runner; cancel it at the timeout or the deadline."""
        job = self.jobs.launch([aid])
        limit = time.time() + 60 * min(timeout_min, max(self.remaining_min(), 1))
        while self.store.get(aid)["status"] in ("pending", "running"):
            time.sleep(3)
            self.jobs.reconcile()
            if time.time() > limit:
                self.jobs.cancel(job["id"])
                tail = self.store.log_path(aid).read_text(errors="replace").strip().splitlines()[-3:] \
                    if self.store.log_path(aid).exists() else []
                raise TimeoutError(f"{aid} stopped after {timeout_min:.0f} min (timeout or deadline); "
                                   f"last log lines: {' | '.join(tail)}")
        m = self.store.get(aid)
        if m["status"] != "completed":
            raise RuntimeError(f"{aid} {m['status']}: {(m.get('run') or {}).get('error')}")
        return m

    def need(self, key: str) -> dict:
        if key not in self.state:
            raise Skip(f"Requires '{key}', produced by an earlier test that failed or did not run")
        return self.state[key]


def test(tid, phase, title, est, quick_est=None):
    """Register a test; ``est`` (minutes) is used to skip it when the deadline is near."""
    def wrap(fn):
        fn.tid, fn.phase, fn.title = tid, phase, title
        fn.est = lambda c: (quick_est if (c.quick and quick_est is not None) else est)
        TESTS.append(fn)
        return fn
    return wrap


TESTS: list = []


def sh(cmd: list[str], log: Path, timeout_min: float, env: dict | None = None, cwd: str | None = None) -> int:
    with open(log, "a") as fh:
        fh.write(f"$ {' '.join(cmd)}\n")
        fh.flush()
        try:
            return subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, timeout=60 * timeout_min,
                                  env={**os.environ, **(env or {})}, cwd=cwd).returncode
        except subprocess.TimeoutExpired:
            fh.write(f"\n!! timeout after {timeout_min} min\n")
            return -1


def pyenv() -> dict:
    return {"PYTHONPATH": f"{ROOT}:{CORNERSTONE_ROOT}"}


def junit_counts(path: Path) -> dict:
    import xml.etree.ElementTree as ET
    if not path.exists():
        return {}
    r = ET.parse(path).getroot()
    suites = [r] if r.tag == "testsuite" else list(r)
    tot = {k: sum(int(s.get(k, 0)) for s in suites) for k in ("tests", "failures", "errors", "skipped")}
    tot["failed_names"] = [f"{c.get('classname')}::{c.get('name')}" for s in suites for c in s.iter("testcase")
                           if c.find("failure") is not None or c.find("error") is not None]
    tot["time_s"] = round(sum(float(s.get("time", 0)) for s in suites), 1)
    return tot


def figure(name: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt, name


def mean_by_window(rows: list[dict], key: str) -> dict:
    df = pd.DataFrame(rows)
    if key not in df:
        return {}
    return {w: round(float(v), 4) for w, v in df.groupby("window")[key].mean().items()}


# ======================================================================================
# P0 environment
# ======================================================================================
@test("P0.env", "P0 environment", "Versions, hardware, disk, data store present", 1)
def t_env(c: Campaign, log: Path):
    from cs_gui.runner import provenance
    prov = provenance.collect(CORNERSTONE_ROOT)
    mem = {l.split(":")[0]: l.split()[1] for l in Path("/proc/meminfo").read_text().splitlines()[:3]}
    free = shutil.disk_usage(c.out).free / 1e9
    try:
        import torch
        cuda = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        cuda = None
    d = {"python": platform.python_version(), "cpus": os.cpu_count(), "ram_gb": round(int(mem["MemTotal"]) / 1e6, 1),
         "disk_free_gb": round(free, 1), "cuda": cuda, "cornerstone": prov.get("cornerstone"),
         "gui": provenance.git_state(str(ROOT)), "packages": prov.get("packages"),
         "data_store": DATA_ROOT, "store_has_manifest": (Path(DATA_ROOT) / "manifest.json").exists(),
         "legacy_sqlite": Path(LEGACY_DB).exists()}
    c.state["env"] = d
    probs = c.settings.problems()
    if probs:
        return "fail", "; ".join(probs), d
    if free < 20:
        return "warn", f"Only {free:.0f} GB free on the disk of the results (≥ 20 GB recommended)", d
    return "pass", f"{d['cpus']} CPU, {d['ram_gb']} GB RAM, CUDA {cuda}, {free:.0f} GB free", d


# ======================================================================================
# P2 data store integrity
# ======================================================================================
@test("P2.manifest", "P2 data store", "Every store file matches the sha256 of the manifest", 6, 6)
def t_manifest(c: Campaign, log: Path):
    from cornerstone.data import DataStore
    from cornerstone.data.bootstrap import _dataset_dir
    from cornerstone.data.store import sha256
    st = DataStore(DATA_ROOT)
    man = st.load_manifest()["datasets"]
    bad, missing, n, size = [], [], 0, 0
    digests = {}
    for name, d in man.items():
        for f, info in d["files"].items():
            p = st.root / _dataset_dir(name) / f
            if not p.exists():
                missing.append(str(p.relative_to(st.root)))
                continue
            h = sha256(p)
            digests[str(p.relative_to(st.root))] = h
            n, size = n + 1, size + p.stat().st_size
            if info.get("sha256") and h != info["sha256"]:
                bad.append(str(p.relative_to(st.root)))
    c.state["store_digests"] = digests
    extra = sorted({str(p.relative_to(st.root)) for p in (st.root / "raw").rglob("*.parquet")} - set(digests))
    d = {"datasets": sorted(man), "files": n, "size_mb": round(size / 1e6), "corrupted": bad, "missing": missing,
         "files_not_in_manifest": extra[:20]}
    if bad or missing:
        return "fail", f"{len(bad)} corrupted, {len(missing)} missing files", d
    if extra:
        return "warn", f"{len(extra)} Parquet files not listed in the manifest", d
    expected = {"sharp/DEF", "sharp/NRT", "goes/swpc_events", "harp_noaa"}
    if not expected <= set(man):
        return "fail", f"Datasets missing: {sorted(expected - set(man))}", d
    if any(k.startswith("reference/") for k in man):
        return "warn", "The store still lists reference/ datasets (removed by the next update)", d
    return "pass", f"{n} files, {size / 1e6:.0f} MB, all verified", d


@test("P2.sharp", "P2 data store", "SHARP DEF/NRT: rows, keys, order, missing values, freshness", 8, 8)
def t_sharp(c: Campaign, log: Path):
    from cornerstone.data import DataStore
    st = DataStore(DATA_ROOT)
    man = st.load_manifest()["datasets"]
    out, problems, warns = {}, [], []
    plt, _ = figure("records")
    fig, ax = plt.subplots(figsize=(9, 3.5))
    for s in ("DEF", "NRT"):
        df = st.read_sharp(s, columns=["HARPNUM", "T_OBS", "USFLUX", "LAT_MIN", "QUALITY", "NOAA_AR"])
        rows_man = sum(f["rows"] for f in man[f"sharp/{s}"]["files"].values())
        dup = int(df.duplicated(["HARPNUM", "T_OBS"]).sum())
        order_ok = bool((df.groupby("HARPNUM", sort=False)["T_OBS"].apply(lambda x: x.is_monotonic_increasing)).all())
        last = df["T_OBS"].max()
        lag_days = (pd.Timestamp.now(tz="UTC") - last).days
        o = {"rows": len(df), "rows_manifest": rows_man, "harps": int(df["HARPNUM"].nunique()),
             "harp_max": int(df["HARPNUM"].max()), "t_first": str(df["T_OBS"].min()), "t_last": str(last),
             "days_since_last_record": lag_days, "duplicated_keys": dup, "sorted_within_harp": order_ok,
             "null_fraction": {k: round(float(df[k].isna().mean()), 4) for k in ("USFLUX", "LAT_MIN", "NOAA_AR")},
             "empty_patch_records": int((df["LAT_MIN"].isna() & (df["USFLUX"].fillna(0) == 0)).sum())}
        out[s] = o
        if len(df) != rows_man:
            problems.append(f"{s}: {len(df)} rows read, manifest says {rows_man}")
        if dup:
            problems.append(f"{s}: {dup} duplicated (HARPNUM, T_OBS)")
        if not order_ok:
            problems.append(f"{s}: T_OBS not sorted within a HARP")
        if lag_days > (90 if s == "DEF" else 7):
            warns.append(f"{s}: last record {lag_days} days ago (store not updated?)")
        per_m = df.set_index("T_OBS").resample("MS").size()
        ax.plot(per_m.index, per_m.values, label=f"SHARP {s}")
    ax.set_ylabel("records / month")
    ax.legend()
    ax.set_title("SHARP records per month in the store")
    fig.tight_layout()
    fig.savefig(c.out / "figures" / "sharp_records_per_month.png", dpi=110)
    plt.close(fig)
    if problems:
        return "fail", "; ".join(problems), out
    if warns:
        return "warn", "; ".join(warns), out
    return "pass", ", ".join(f"{s} {o['rows']:,} rows / {o['harps']} HARPs to {o['t_last'][:10]}"
                             for s, o in out.items()), out


@test("P2.goes", "P2 data store", "GOES events and HARP-NOAA map: coverage and consistency", 2, 2)
def t_goes(c: Campaign, log: Path):
    from cornerstone.data import DataStore
    st = DataStore(DATA_ROOT)
    g = pd.concat([pd.read_parquet(p) for p in sorted((st.root / "raw/goes").glob("*.parquet"))], ignore_index=True)
    hn = pd.read_parquet(next((st.root / "raw/harp_noaa").glob("*.parquet")))
    t = pd.to_datetime(g["tmax"], utc=True, errors="coerce")
    tb = pd.to_datetime(g["tstart"], utc=True, errors="coerce")
    year = tb.dt.year
    per_year = year.value_counts().sort_index()
    cls = g["goes_class"].astype(str).str[0]
    d = {"events": len(g), "columns": list(g.columns), "years": {int(k): int(v) for k, v in per_year.items()},
         "missing_peak": int(t.isna().sum()), "duplicated_event_id": int(g["event_id"].duplicated().sum()),
         "satellites": g["satellite"].value_counts().to_dict(),
         "class_counts": cls.value_counts().to_dict(), "harp_noaa_pairs": len(hn),
         "last_event": str(tb.max())}
    plt, _ = figure("goes")
    fig, ax = plt.subplots(figsize=(9, 3.2))
    pd.crosstab(year, cls).reindex(columns=[x for x in "ABCMX" if x in set(cls)]).plot.bar(stacked=True, ax=ax, width=0.8)
    ax.set_title("GOES X-ray events per year and class")
    fig.tight_layout()
    fig.savefig(c.out / "figures" / "goes_events_per_year.png", dpi=110)
    plt.close(fig)
    probs = []
    if d["duplicated_event_id"]:
        probs.append(f"{d['duplicated_event_id']} duplicated event_id")
    gaps = [y for y in range(2010, dt.date.today().year + 1) if per_year.get(y, 0) == 0]
    if gaps:
        probs.append(f"years without events: {gaps}")
    if probs:
        return "fail", "; ".join(probs), d
    lag = (pd.Timestamp.now(tz="UTC") - tb.max()).days
    if lag > 14:
        return "warn", f"Last GOES event {lag} days ago", d
    return "pass", f"{len(g):,} events {min(d['years'])}-{max(d['years'])}, {len(hn):,} HARP-NOAA pairs", d


# ======================================================================================
# P9 parameter validation (the GUI must refuse wrong inputs before running anything)
# ======================================================================================
@test("P9.validation", "P9 robustness", "Wrong parameters are refused with a message", 1)
def t_validation(c: Campaign, log: Path):
    s = c.settings
    base = default_label_config("DEF", "M")
    lab_cases = {
        "unknown threshold": dict(threshold="Z"), "zero window": dict(flare_window_hours=0),
        "start after end": dict(start="2020-01-01", end="2019-01-01"), "unknown series": dict(series="XYZ"),
        "bad calibration": dict(flares={**base["flares"], "calibration": "foo"}),
        "max_abs_lon > 90": dict(quality={**base["quality"], "max_abs_lon": 120}),
    }
    fake_labels = {"config": {**base, "step_hours": 24}, "summary": {"n_rows": 100}}
    ds_ok = {"features": "clean", "quality_source": "feature_set", "window_hours": 24, "sampling_minutes": 12,
             "subsample": False, "subsample_size": 0, "random_seed": 42}
    ds_cases = {"sampling 7 min": dict(sampling_minutes=7), "window 0 h": dict(window_hours=0),
                "unknown features": dict(features="foo"), "subsample > rows": dict(subsample=True, subsample_size=10**9)}
    fake_ds = {"summary": {"feature_names": CLEAN_FEATURES, "timesteps": 121, "time_start": "2010-05-01",
                           "time_end": "2026-10-01"}}
    sp_ok = {**default_split_config(s), "FEATURE_SELECTION_MODE": "direct", "DIRECT_FEATURES": CLEAN_FEATURES,
             "TEMPORAL_FILTER_MODE": None, "TEMPORAL_WINDOW": None}
    sp_cases = {"unknown feature": dict(DIRECT_FEATURES=["NOPE"]), "TIMESTEPS > dataset": dict(TIMESTEPS=500),
                "N_FOLDS 1": dict(N_FOLDS=1), "cutoff with random split": dict(SPLITTING_STRATEGY="random", TEST_CUTOFF_DATE="2018-01-01"),
                "multi-class": dict(MULTI_CLASS=True)}
    fake_split = {"summary": {"timesteps": 121, "leakage": {"n_folds": 5, "test_vs_trainval_shared_harps": 0,
                                                           "test_vs_trainval_shared_samples": 0}, "files": {}}}
    fake_mc = {"config": {"model": "XGBOOST"}}
    tr_ok = {"windows": ["24h"], "window_map": {"24h": 121}, "n_folds": 5, "device": "cpu"}
    tr_cases = {"no window": dict(windows=[]), "window longer than split": dict(window_map={"24h": 500}),
                "more folds than split": dict(n_folds=9), "unknown device": dict(device="tpu")}
    checks = {}
    checks["labels: valid config accepted"] = not validate_labels(base, s)
    for k, v in lab_cases.items():
        checks[f"labels: {k} refused"] = bool(validate_labels({**base, **v}, s))
    checks["dataset: valid config accepted"] = not validate_dataset(ds_ok, fake_labels, s)
    for k, v in ds_cases.items():
        checks[f"dataset: {k} refused"] = bool(validate_dataset({**ds_ok, **v}, fake_labels, s))
    checks["split: valid config accepted"] = not validate_split(sp_ok, fake_ds, c.pl.info)
    for k, v in sp_cases.items():
        checks[f"split: {k} refused"] = bool(validate_split({**sp_ok, **v}, fake_ds, c.pl.info))
    checks["training: valid config accepted"] = not validate_training(tr_ok, fake_split, fake_mc, c.pl.info)
    for k, v in tr_cases.items():
        checks[f"training: {k} refused"] = bool(validate_training({**tr_ok, **v}, fake_split, fake_mc, c.pl.info))
    leaky = copy.deepcopy(fake_split)
    leaky["summary"]["leakage"]["test_vs_trainval_shared_harps"] = 3
    checks["training: split with test/train shared HARPs refused"] = bool(validate_training(tr_ok, leaky, fake_mc, c.pl.info))
    empty = Settings(cornerstone_root=CORNERSTONE_ROOT, cornerstone_python=PYTHON, data_root=str(c.out / "empty_store"),
                     workspace=str(c.out / "workspace"), mode="real")
    checks["empty store reported by the settings"] = bool(empty.problems())
    failed = [k for k, ok in checks.items() if not ok]
    return ("fail" if failed else "pass"), (f"{len(failed)} checks failed: {failed}" if failed
                                            else f"{len(checks)} checks"), {"checks": checks}


# ======================================================================================
# P4 labeling (package API, without artifacts)
# ======================================================================================
def _label_invariants(lab: pd.DataFrame, cfg) -> list[str]:
    from cornerstone.labeling.core import _REL, class_letter
    p = []
    if lab.duplicated(["t_X", "HARPNUM"]).any():
        p.append("duplicated (t_X, HARPNUM)")
    if set(lab["label"]) - {"YES", "NO"}:
        p.append(f"labels other than YES/NO: {sorted(set(lab['label']) - {'YES', 'NO'})}")
    yes = lab["label"] == "YES"
    if (yes != (lab["max_flux"].fillna(0) >= cfg.s_p * (1 - _REL))).any():
        p.append("YES not equivalent to max_flux >= S_p")
    if (yes != (lab["n_window"] > 0)).any():
        p.append("YES not equivalent to n_window > 0")
    if "target" in lab and (lab["target"] != yes.astype(int)).any():
        p.append("target != (label == YES)")
    big = lab["max_flux"].fillna(0) >= cfg.s_s * (1 - _REL)
    exp_no = np.array(["NO1", "NO2", "NO4", "NO3"])[2 * (lab["n_before"] > 0).to_numpy() + (lab["n_after"] > 0).to_numpy()]
    exp = np.where(big, class_letter(lab["max_flux"].fillna(0).to_numpy()), exp_no)
    if (lab["strat_class"].to_numpy() != exp).any():
        p.append(f"strat_class inconsistent with max_flux/n_before/n_after in {(lab['strat_class'].to_numpy() != exp).sum()} rows")
    tx = pd.to_datetime(lab["t_X"], utc=True)
    if ((tx < pd.to_datetime(lab["obs_start"], utc=True)) | (tx > pd.to_datetime(lab["obs_end"], utc=True))).any():
        p.append("t_X outside the observation span of the HARP")
    return p


@test("P4.labeling", "P4 labeling", "Labels DEF/NRT × C/M/X: invariants, monotonicity, timing", 25, 6)
def t_labeling(c: Campaign, log: Path):
    from cornerstone.data import DataStore
    from cornerstone.labeling import LabelConfig, label_from_store
    st = DataStore(DATA_ROOT)
    kw = dict(start="2012-01-01", end="2015-12-31") if c.quick else {}
    d, probs, yes_rate = {}, [], {}
    for s in ("DEF", "NRT"):
        yes = {}
        for thr in ("C", "M", "X"):
            cfg = LabelConfig(threshold=thr, **kw)
            t0 = time.time()
            lab, meta = label_from_store(st, s, cfg)
            secs = time.time() - t0
            inv = _label_invariants(lab, cfg)
            yes[thr] = int((lab["label"] == "YES").sum())
            d[f"{s}_{thr}"] = {"rows": len(lab), "harps": int(lab["HARPNUM"].nunique()), "yes": yes[thr],
                               "strat_class": lab["strat_class"].value_counts().to_dict(), "seconds": round(secs, 1),
                               "association": meta.get("flare_association"), "invariant_violations": inv}
            probs += [f"{s} {thr}: {x}" for x in inv]
            tx = pd.to_datetime(lab["t_X"], utc=True)
            yes_rate[f"{s} {thr}"] = (lab["label"] == "YES").groupby(tx.dt.year).mean()
            with open(log, "a") as fh:
                fh.write(f"{s} {thr}: {len(lab):,} rows, {yes[thr]:,} YES, {secs:.1f}s\n")
        if not yes["C"] >= yes["M"] >= yes["X"]:
            probs.append(f"{s}: YES counts not monotonic in the threshold {yes}")
    plt, _ = figure("yes")
    fig, ax = plt.subplots(figsize=(9, 3.5))
    for k, v in yes_rate.items():
        ax.plot(v.index, 100 * v.values, marker="o", ms=3, label=k, ls="-" if "DEF" in k else "--")
    ax.set_ylabel("% YES")
    ax.set_title("Share of YES labels per year (W = 24 h)")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(c.out / "figures" / "labels_yes_rate.png", dpi=110)
    plt.close(fig)
    return ("fail" if probs else "pass"), ("; ".join(probs[:8]) if probs else
                                           ", ".join(f"{k} {v['yes']:,} YES/{v['rows']:,} ({v['seconds']:.0f}s)"
                                                     for k, v in d.items())), d


@test("P4.determinism", "P4 labeling", "Labeling twice with the same parameters gives identical labels", 6, 2)
def t_label_determinism(c: Campaign, log: Path):
    from cornerstone.data import DataStore
    from cornerstone.labeling import LabelConfig, label_from_store
    st = DataStore(DATA_ROOT)
    cfg = LabelConfig(threshold="M", strat_threshold="C", strat_window_hours=48,
                      **(dict(start="2012-01-01", end="2015-12-31") if c.quick else {}))
    a, _ = label_from_store(st, "NRT", cfg)
    b, _ = label_from_store(st, "NRT", cfg)
    pd.testing.assert_frame_equal(a, b)
    inv = _label_invariants(a, cfg)
    d = {"rows": len(a), "strat_class": a["strat_class"].value_counts().to_dict(), "invariant_violations": inv}
    if inv:
        return "fail", "S_s = C, L = 48 h: " + "; ".join(inv), d
    return "pass", f"identical ({len(a):,} rows); S_s = C, L = 48 h invariants hold", d


@test("P4.legacy", "P4 labeling", "Comparison with the static tables of the former database", 10, 10)
def t_legacy(c: Campaign, log: Path):
    if not Path(LEGACY_DB).exists():
        raise Skip(f"{LEGACY_DB} not present on this machine (the former database is optional)")
    from cornerstone.labeling.validation import compare_with_legacy
    d = {}
    for s in ("NRT", "DEF"):
        r = compare_with_legacy(LEGACY_DB, s, "M")
        r.pop("differences")
        d[s] = {k: (v if not isinstance(v, dict) else {str(a): b for a, b in v.items()}) for k, v in r.items()}
    msg = ", ".join(f"{s}: {100 * v['same_label']:.2f}% same label, {100 * v['same_strat_class']:.2f}% same class"
                    for s, v in d.items())
    # known differences with the former pipeline (Nov 2026: NRT 99.71%, DEF 97.84% same label, mostly
    # DEF YES -> NO): a warning only if the agreement gets worse than that
    worse = [s for s, v in d.items() if v["same_label"] < LEGACY_AGREEMENT[s] - 0.002]
    return ("warn" if worse else "pass"), (f"agreement lower than known for {worse}: " if worse else "") + msg, d


# ======================================================================================
# P5 the GUI pipeline on real data (labels -> dataset -> split -> training -> inference)
# ======================================================================================
def _split_cfg(c: Campaign, features: list[str], n_folds: int = 5) -> dict:
    cfg = default_split_config(c.settings)
    cfg.update(FEATURE_SELECTION_MODE="direct", DIRECT_FEATURES=features, TEMPORAL_FILTER_MODE=None,
               TEMPORAL_WINDOW=None, N_FOLDS=n_folds, SAVE_PLOTS=True, RANDOM_STATE=42)
    return cfg


def _chain_to_split(c: Campaign, tag: str, series: str, features: str, log: Path) -> dict:
    lab_cfg = default_label_config(series, "M")
    if c.quick:
        lab_cfg.update(start="2011-01-01", end="2015-12-31")
    t = {}
    t0 = time.time()
    lab = c.run_artifact(c.pl.create_labels(f"nightly {tag} labels", lab_cfg)["id"], 30)
    t["labels_min"] = round((time.time() - t0) / 60, 1)
    t0 = time.time()
    ds = c.run_artifact(c.pl.create_dataset(f"nightly {tag} dataset", lab["id"], {
        "features": features, "quality_source": "feature_set", "window_hours": 24, "sampling_minutes": 12,
        "subsample": False, "subsample_size": 0, "random_seed": 42})["id"], 60)
    t["dataset_min"] = round((time.time() - t0) / 60, 1)
    t0 = time.time()
    sp = c.run_artifact(c.pl.create_split(f"nightly {tag} split", ds["id"],
                                          _split_cfg(c, ds["summary"]["feature_names"]))["id"], 30)
    t["split_min"] = round((time.time() - t0) / 60, 1)
    with open(log, "a") as fh:
        fh.write(json.dumps({"labels": lab["id"], "dataset": ds["id"], "split": sp["id"], **t}) + "\n")
    return {"lab": lab, "ds": ds, "sp": sp, "times": t}


def _train_and_infer(c: Campaign, tag: str, sp: dict, model: str, model_params: dict, tcfg: dict,
                     windows: list[str], n_folds: int, timeout_min: float, reg=None) -> tuple[dict, dict]:
    wm = {"3h": 16, "6h": 31, "12h": 61, "24h": 121}
    if reg is None and model in c.pl.info.get("sk_models", []):
        reg = [1e2, 1e3]                      # sklearn-style models need a regularization grid
    mc = c.pl.create_model_config(f"nightly {tag} {model}", {
        "model": model, "model_params": model_params, "training_config": tcfg,
        "regularization_params": reg, "seed": 42})
    tr = c.run_artifact(c.pl.create_training(f"nightly {tag} {model}", sp["id"], mc["id"], {
        "windows": windows, "window_map": {w: wm[w] for w in windows}, "n_folds": n_folds, "device": "auto"})["id"],
        timeout_min)
    inf = c.run_artifact(c.pl.create_inference(f"nightly {tag} {model} test", tr["id"], sp["id"], {
        "windows": windows, "threshold_mode": "cv", "model_choice": "best", "source_file": "test",
        "exclude_overlap": True, "bootstrap": {"n": 200, "cluster": True}})["id"], 30)
    return tr, inf


def _inference_tss(inf: dict) -> dict:
    out = {}
    for w, e in inf["summary"]["windows"].items():
        m = e.get("metrics") or {}
        out[w] = None if not m else round(float(m.get("TSS", m.get("tss", np.nan))), 4)
    return out


def _short_tcfg(epochs: int, folds: int) -> dict:
    t = copy.deepcopy(DEFAULT_TRAINING_CONFIG)
    t.update(num_epochs=epochs, patience=max(2, epochs), n_folds=folds)
    t["lr_scheduler"]["T_max"] = epochs
    t["early_stopping"]["patience"] = max(2, epochs)
    return t


@test("P5.def_xgb", "P5 pipeline (real data)", "DEF M+ 24 h: labels → dataset → split → XGBoost → inference", 40, 12)
def t_chain_def(c: Campaign, log: Path):
    ch = _chain_to_split(c, "DEF", "DEF", "clean", log)
    c.state["def_chain"] = ch
    lk = ch["sp"]["summary"].get("leakage", {})
    t0 = time.time()
    windows = ["24h"] if c.quick else ["3h", "6h", "12h", "24h"]
    tr, inf = _train_and_infer(c, "DEF", ch["sp"], "XGBOOST", {"n_estimators": 30 if c.quick else 100},
                               copy.deepcopy(DEFAULT_TRAINING_CONFIG), windows, 5, 120, reg=[1e2, 1e3])
    c.state["def_xgb"] = {"tr": tr, "inf": inf}
    tss = _inference_tss(inf)
    d = {"artifacts": {k: ch[k]["id"] for k in ("lab", "ds", "sp")} | {"training": tr["id"], "inference": inf["id"]},
         "times_min": {**ch["times"], "train_infer_min": round((time.time() - t0) / 60, 1)},
         "labels": {k: ch["lab"]["summary"][k] for k in ("n_rows", "n_active_regions", "label_counts")},
         "dataset": {k: ch["ds"]["summary"].get(k) for k in ("n_samples", "n_features", "n_positive", "nan_ratio_overall",
                                                             "n_active_regions")},
         "split": {k: ch["sp"]["summary"].get(k) for k in ("n_samples_kept", "n_samples_excluded")}, "leakage": lk,
         "cv_mean_val_TSS": mean_by_window(tr["summary"]["rows"], "val_TSS"),
         "cv_mean_test_TSS": mean_by_window(tr["summary"]["rows"], "test_TSS"), "inference_test_TSS": tss}
    probs = [a for a in (ch["lab"]["id"], ch["ds"]["id"], ch["sp"]["id"], tr["id"], inf["id"]) if c.store.verify(a)]
    if probs:
        return "fail", f"Artifacts not intact: {probs}", d
    if lk.get("test_vs_trainval_shared_harps") or lk.get("test_vs_trainval_shared_samples"):
        return "fail", f"Leakage between test and train/val: {lk}", d
    pred = pd.read_csv(c.store.path(inf["id"]) / "predictions.csv")
    if pred["in_trainval_harp"].any():
        return "fail", "Test predictions on HARPs seen in training", d
    if any(v is None or v < 0.3 for v in tss.values()):
        return "warn", f"Low or missing test TSS: {tss}", d
    return "pass", f"test TSS {tss}; {ch['ds']['summary']['n_samples']:,} samples", d


@test("P5.nrt_xgb", "P5 pipeline (real data)", "NRT M+ 24 h: same chain on the NRT series", 30, 10)
def t_chain_nrt(c: Campaign, log: Path):
    ch = _chain_to_split(c, "NRT", "NRT", "clean", log)
    tr, inf = _train_and_infer(c, "NRT", ch["sp"], "XGBOOST", {"n_estimators": 30 if c.quick else 100},
                               copy.deepcopy(DEFAULT_TRAINING_CONFIG), ["24h"], 5, 90, reg=[1e2, 1e3])
    tss = _inference_tss(inf)
    d = {"artifacts": {"split": ch["sp"]["id"], "training": tr["id"], "inference": inf["id"]}, "times_min": ch["times"],
         "dataset": {k: ch["ds"]["summary"].get(k) for k in ("n_samples", "n_positive", "nan_ratio_overall")},
         "leakage": ch["sp"]["summary"].get("leakage"), "inference_test_TSS": tss}
    if any(v is None or v < 0.3 for v in tss.values()):
        return "warn", f"Low or missing test TSS: {tss}", d
    return "pass", f"test TSS {tss}; {ch['ds']['summary']['n_samples']:,} samples", d


@test("P5.def_all22", "P5 pipeline (real data)", "DEF with all 22 SHARP keywords (features 'all') + XGBoost", 30, 10)
def t_chain_all(c: Campaign, log: Path):
    ch = _chain_to_split(c, "DEF-all", "DEF", "all", log)
    tr, inf = _train_and_infer(c, "DEF-all", ch["sp"], "XGBOOST", {"n_estimators": 30 if c.quick else 100},
                               copy.deepcopy(DEFAULT_TRAINING_CONFIG), ["24h"], 5, 90, reg=[1e2, 1e3])
    tss = _inference_tss(inf)
    ref = (c.state.get("def_xgb") or {}).get("inf")
    d = {"n_features": ch["ds"]["summary"].get("n_features"), "inference_test_TSS": tss,
         "clean_features_test_TSS": _inference_tss(ref) if ref else None}
    if ch["ds"]["summary"].get("n_features") != 22:
        return "fail", f"Expected 22 features, got {ch['ds']['summary'].get('n_features')}", d
    return "pass", f"22 features, test TSS {tss}", d


# ======================================================================================
# P7 reproducibility, P8 bundles, P10 GUI pages
# ======================================================================================
@test("P7.reproduce", "P7 reproducibility", "Re-running dataset and XGBoost training gives identical results", 25, 6)
def t_reproduce(c: Campaign, log: Path):
    import h5py
    ch, x = c.need("def_chain"), c.need("def_xgb")
    d, probs = {}, []
    ds2 = c.run_artifact(c.pl.reproduce_step(ch["ds"]["id"])["id"], 120)
    f1, f2 = (c.store.path(a) / "dataset.h5" for a in (ch["ds"]["id"], ds2["id"]))
    with h5py.File(f1) as a, h5py.File(f2) as b:
        same = {k: (a[k].shape == b[k].shape and bool(np.array_equal(a[k][()], b[k][()], equal_nan=a[k].dtype.kind == "f")))
                for k in a.keys() if k in b}
    d["dataset_arrays_equal"] = same
    if not all(same.values()):
        probs.append(f"dataset arrays differ: {[k for k, v in same.items() if not v]}")
    tr2 = c.run_artifact(c.pl.reproduce_step(x["tr"]["id"])["id"], 120)
    r1, r2 = pd.DataFrame(x["tr"]["summary"]["rows"]), pd.DataFrame(tr2["summary"]["rows"])
    cols = [k for k in r1.columns if k.startswith(("val_", "test_"))]
    diff = float(np.nanmax(np.abs(r1[cols].to_numpy(float) - r2[cols].to_numpy(float)))) if len(r1) == len(r2) else None
    d["training_max_metric_difference"] = diff
    if diff is None or diff > 1e-9:
        probs.append(f"training metrics differ (max |Δ| = {diff})")
    return ("fail" if probs else "pass"), ("; ".join(probs) if probs else "dataset and XGBoost metrics identical"), d


@test("P8.bundle", "P8 sharing", "Export the DEF chain as a bundle and import it into an empty workspace", 10, 4)
def t_bundle(c: Campaign, log: Path):
    from cs_gui.bundle import export_bundle, import_bundle
    x = c.need("def_xgb")
    path = c.out / "bundle_test.zip"
    t0 = time.time()
    desc = export_bundle(c.store, [x["inf"]["id"]], path, note="nightly test")
    size = path.stat().st_size
    other = ArtifactStore(c.out / "workspace_import")
    res = import_bundle(other, path, mode="real")
    bad = {a: other.verify(a) for a in res["imported"] if other.verify(a)}
    again = import_bundle(other, path, mode="real")
    try:
        import_bundle(ArtifactStore(c.out / "workspace_import_demo"), path, mode="demo")
        mixed = "accepted"
    except StoreError:
        mixed = "refused"
    d = {"artifacts": len(desc.get("artifacts", [])), "size_mb": round(size / 1e6, 1), "imported": len(res["imported"]),
         "second_import_skipped": len(again.get("skipped", [])), "not_intact": bad, "demo_import": mixed,
         "seconds": round(time.time() - t0, 1)}
    path.unlink(missing_ok=True)
    shutil.rmtree(c.out / "workspace_import", ignore_errors=True)
    shutil.rmtree(c.out / "workspace_import_demo", ignore_errors=True)
    if bad or mixed != "refused" or not res["imported"]:
        return "fail", f"not intact {list(bad)}, demo import {mixed}", d
    return "pass", f"{d['imported']} artifacts, {d['size_mb']} MB, re-import skipped {d['second_import_skipped']}", d


@test("P10.gui_pages", "P10 GUI", "Every GUI page renders without exceptions (Streamlit AppTest)", 3)
def t_pages(c: Campaign, log: Path):
    code = r"""
import json, sys
from streamlit.testing.v1 import AppTest
res = {}
for p in sys.argv[2:]:
    try:
        at = AppTest.from_file(f"app/views/{p}.py", default_timeout=180)
        at.run()
        res[p] = {"exceptions": [e.value[:300] for e in at.exception], "errors": [e.body[:200] for e in at.error]}
    except Exception as e:
        res[p] = {"exceptions": [f"{type(e).__name__}: {e}"], "errors": []}
json.dump(res, open(sys.argv[1], "w"), indent=1)
"""
    pages = sorted(p.stem for p in (ROOT / "app" / "views").glob("*.py"))
    out = c.out / "logs" / "gui_pages.json"
    settings_file = c.out / "gui_settings.json"          # not created: real mode, paths from CS_GUI_*
    rc = sh([PYTHON, "-c", code, str(out), *pages], log, 10,
            env={**pyenv(), "CS_GUI_SETTINGS_FILE": str(settings_file), "CS_GUI_WORKSPACE": str(c.out / "workspace")},
            cwd=str(ROOT))
    res = json.loads(out.read_text()) if out.exists() else {}
    bad = {p: r for p, r in res.items() if r["exceptions"]}
    errs = {p: r["errors"] for p, r in res.items() if r["errors"]}
    if rc or bad or len(res) != len(pages):
        return "fail", f"pages with exceptions: {list(bad) or 'runner failed'}", {"pages": res}
    # error boxes are messages of the pages about the data (e.g. Compare: inferences on different
    # splits are not comparable): listed, not a failure
    note = f"; error boxes shown by {list(errs)} (see details)" if errs else ""
    return "pass", f"{len(pages)} pages, no exceptions{note}", {"pages": res}


# ======================================================================================
# P1 automatic test suites
# ======================================================================================
@test("P1.package_tests", "P1 test suites", "Package unit tests (cornerstone_db_package/tests)", 5)
def t_pkg_tests(c: Campaign, log: Path):
    xml = c.out / "logs" / "package_tests.xml"
    sh([PYTHON, "-m", "pytest", f"{CORNERSTONE_ROOT}/tests", "-q", "-p", "no:cacheprovider", f"--junitxml={xml}"],
       log, 30, env=pyenv(), cwd=str(c.out))
    r = junit_counts(xml)
    if not r:
        return "fail", "pytest did not run (see the log)", r
    bad = r["failures"] + r["errors"]
    return ("fail" if bad else "pass"), f"{r['tests']} tests, {bad} failed, {r['skipped']} skipped", r


@test("P1.gui_unit", "P1 test suites", "GUI unit tests", 3)
def t_gui_unit(c: Campaign, log: Path):
    xml = c.out / "logs" / "gui_unit.xml"
    sh([PYTHON, "-m", "pytest", str(ROOT / "tests"), "-m", "not integration", "-q", "-p", "no:cacheprovider",
        f"--junitxml={xml}"], log, 20, env=pyenv(), cwd=str(c.out))
    r = junit_counts(xml)
    if not r:
        return "fail", "pytest did not run (see the log)", r
    bad = r["failures"] + r["errors"]
    return ("fail" if bad else "pass"), f"{r['tests']} tests, {bad} failed, {r['skipped']} skipped", r


@test("P1.gui_integration", "P1 test suites", "GUI integration tests (whole chain on synthetic data)", 15)
def t_gui_integration(c: Campaign, log: Path):
    xml = c.out / "logs" / "gui_integration.xml"
    sh([PYTHON, "-m", "pytest", str(ROOT / "tests"), "-m", "integration", "-q", "-p", "no:cacheprovider",
        f"--junitxml={xml}"], log, 45, env={**pyenv(), "CS_GUI_CORNERSTONE_PYTHON": PYTHON}, cwd=str(c.out))
    r = junit_counts(xml)
    if not r:
        return "fail", "pytest did not run (see the log)", r
    bad = r["failures"] + r["errors"]
    return ("fail" if bad else "pass"), f"{r['tests']} tests, {bad} failed, {r['skipped']} skipped", r


# ======================================================================================
# P6 neural models (short trainings: do they run end to end?)
# ======================================================================================
def _nn_test(model: str):
    @test(f"P6.{model.lower()}", "P6 models (short training)", f"{model}: short training + inference on the DEF split", 45, 15)
    def t(c: Campaign, log: Path):
        ch = c.need("def_chain")
        epochs, folds = (1, 2) if c.quick else (5, 2)
        t0 = time.time()
        tr, inf = _train_and_infer(c, f"DEF-short", ch["sp"], model, {}, _short_tcfg(epochs, folds), ["24h"], folds,
                                   90 if not c.quick else 30)
        tss = _inference_tss(inf)
        d = {"epochs": epochs, "folds": folds, "device": tr["summary"].get("device"),
             "training_min": round(tr["summary"].get("training_seconds", 0) / 60, 1),
             "total_min": round((time.time() - t0) / 60, 1), "cv_mean_val_TSS": mean_by_window(tr["summary"]["rows"], "val_TSS"),
             "inference_test_TSS": tss}
        if any(v is None for v in tss.values()):
            return "warn", f"completed, but test metrics not computable: {tss}", d
        return "pass", f"completed in {d['total_min']} min ({epochs} epochs, {folds} folds), test TSS {tss}", d
    return t


for _m in NN_MODELS:
    _nn_test(_m)


@test("P6.other_models", "P6 models (short training)", "Other models offered by the GUI: 1 epoch, 1 fold, 3 h window", 60, 0)
def t_other_models(c: Campaign, log: Path):
    if c.quick:
        raise Skip("not run in quick mode")
    ch = c.need("def_chain")
    others = [m for m in c.pl.info["trainable_models"] if m not in NN_MODELS + ["XGBOOST"]]
    d = {}
    for m in others:
        if c.remaining_min() < 15:
            d[m] = {"status": "skip", "error": "deadline"}
            continue
        t0 = time.time()
        try:
            tr, inf = _train_and_infer(c, "DEF-smoke", ch["sp"], m, {}, _short_tcfg(1, 2), ["3h"], 1, 30)
            d[m] = {"status": "pass", "minutes": round((time.time() - t0) / 60, 1), "test_TSS": _inference_tss(inf)}
        except Exception as e:  # noqa: BLE001
            d[m] = {"status": "fail", "minutes": round((time.time() - t0) / 60, 1), "error": f"{type(e).__name__}: {e}"[:400]}
        with open(log, "a") as fh:
            fh.write(f"{m}: {d[m]}\n")
    failed = [m for m, v in d.items() if v["status"] == "fail"]
    ok = [m for m, v in d.items() if v["status"] == "pass"]
    return ("warn" if failed else "pass"), f"working: {ok}; failing: {failed}", d


# ======================================================================================
# P3 update from the sources (on a copy of the store)
# ======================================================================================
@test("P3.update", "P3 data update", "Incremental update from JSOC/NOAA on a copy of the store", 60, 10)
def t_update(c: Campaign, log: Path):
    src = Path(DATA_ROOT)
    size = sum(p.stat().st_size for p in src.rglob("*") if p.is_file() and ".cache" not in p.parts)
    if shutil.disk_usage(c.out).free < 3 * size:
        raise Skip(f"Not enough disk for a copy of the store ({size / 1e9:.1f} GB)")
    copy_root = c.out / "store_copy"
    shutil.rmtree(copy_root, ignore_errors=True)
    shutil.copytree(src, copy_root, ignore=shutil.ignore_patterns(".cache", ".snapshot-partial"))
    sources = "harp_noaa,goes" if c.quick else "harp_noaa,goes,sharp_DEF,sharp_NRT"
    code = ("import sys; from cornerstone.data import DataStore; from cornerstone.data.update import update_all;"
            f"update_all(DataStore(sys.argv[1]), sources={sources.split(',')!r})")
    before = json.loads((copy_root / "manifest.json").read_text())["datasets"]
    t0 = time.time()
    rc = sh([PYTHON, "-c", code, str(copy_root)], log, min(120, max(c.remaining_min() - 30, 5)), env=pyenv())
    after = json.loads((copy_root / "manifest.json").read_text())["datasets"]
    rows = lambda m, k: sum(f.get("rows", 0) for f in m.get(k, {}).get("files", {}).values())  # noqa: E731
    d = {"sources": sources, "minutes": round((time.time() - t0) / 60, 1), "exit_code": rc,
         "rows_before": {k: rows(before, k) for k in sorted(before)}, "rows_after": {k: rows(after, k) for k in sorted(after)},
         "log_tail": log.read_text()[-3000:] if log.exists() else ""}
    shutil.rmtree(copy_root, ignore_errors=True)
    shrunk = [k for k in before if rows(after, k) < rows(before, k)]
    if rc:
        return "fail", f"update exited with code {rc} after {d['minutes']} min (see the log)", d
    if shrunk:
        return "warn", f"rows decreased in {shrunk}", d
    return "pass", f"{d['minutes']} min; " + ", ".join(f"{k} +{d['rows_after'][k] - d['rows_before'].get(k, 0):,}"
                                                        for k in d["rows_after"]), d


# ======================================================================================
# P11 full training (as the reference experiment), only when there is time
# ======================================================================================
@test("P11.inceptiontime_full", "P11 full training", "InceptionTime as the reference experiment (4 windows, 5 folds, 100 epochs)", 360, 0)
def t_full(c: Campaign, log: Path):
    if c.quick:
        raise Skip("not run in quick mode")
    ch = c.need("def_chain")
    tr, inf = _train_and_infer(c, "DEF-full", ch["sp"], "INCEPTIONTIME", {}, copy.deepcopy(DEFAULT_TRAINING_CONFIG),
                               ["3h", "6h", "12h", "24h"], 5, max(c.remaining_min() - 20, 30))
    cv = mean_by_window(tr["summary"]["rows"], "test_TSS")
    delta = {w: round(cv[w] - REFERENCE_INCEPTIONTIME_TEST_TSS[w], 3) for w in cv if w in REFERENCE_INCEPTIONTIME_TEST_TSS}
    d = {"training_hours": round(tr["summary"].get("training_seconds", 0) / 3600, 2), "cv_mean_test_TSS": cv,
         "reference": REFERENCE_INCEPTIONTIME_TEST_TSS, "delta": delta, "inference_test_TSS": _inference_tss(inf)}
    if any(abs(v) > 0.05 for v in delta.values()):
        return "warn", f"test TSS differs from the reference by more than 0.05: {delta}", d
    return "pass", f"{d['training_hours']} h, test TSS {cv} (Δ vs reference {delta})", d


# ======================================================================================
# P12 the data store was not modified
# ======================================================================================
@test("P12.store_unchanged", "P12 final checks", "The tests did not modify the data store", 6, 6)
def t_unchanged(c: Campaign, log: Path):
    from cornerstone.data.store import sha256
    before = c.need("store_digests")
    root = Path(DATA_ROOT)
    changed = [k for k, h in before.items() if not (root / k).exists() or sha256(root / k) != h]
    if changed:
        return "fail", f"{len(changed)} store files changed during the tests", {"changed": changed[:30]}
    return "pass", f"{len(before)} files unchanged", {}


ORDER = ["P0.env", "P2.manifest", "P2.sharp", "P2.goes", "P9.validation", "P4.labeling", "P4.determinism",
         "P4.legacy", "P5.def_xgb", "P5.nrt_xgb", "P5.def_all22", "P7.reproduce", "P8.bundle", "P10.gui_pages",
         "P1.package_tests", "P1.gui_unit", "P1.gui_integration", *[f"P6.{m.lower()}" for m in NN_MODELS],
         "P3.update", "P6.other_models", "P11.inceptiontime_full", "P12.store_unchanged"]


# ======================================================================================
# summary
# ======================================================================================
ICON = {"pass": "✅", "warn": "⚠️", "fail": "❌", "skip": "⏭️"}


def write_summary(out: Path, doc: dict) -> None:
    res = doc["results"]
    counts = {s: sum(r["status"] == s for r in res) for s in ICON}
    env = next((r.get("details") for r in res if r["id"] == "P0.env"), None) or {}
    lines = [f"# CORNERSTONE nightly tests — {doc['started_at'][:10]}", "",
             f"Started {doc['started_at']}, last update {doc['updated_at']}, deadline {doc['deadline']}"
             f"{' (quick mode)' if doc['quick'] else ''}.", "",
             " · ".join(f"{ICON[s]} {s} {n}" for s, n in counts.items()), ""]
    if env:
        cs, gui = env.get("cornerstone") or {}, env.get("gui") or {}
        lines += [f"Package `{str(cs.get('commit'))[:7]}`{' (modified)' if cs.get('dirty') else ''}, "
                  f"GUI `{str(gui.get('commit'))[:7]}`{' (modified)' if gui.get('dirty') else ''}; "
                  f"{env.get('cpus')} CPU, {env.get('ram_gb')} GB RAM, CUDA {env.get('cuda')}, "
                  f"{env.get('disk_free_gb')} GB free.", ""]
    lines += ["| | test | min | result |", "|---|---|---|---|"]
    for r in res:
        msg = str(r.get("message", "")).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {ICON[r['status']]} | **{r['id']}** {r['title']} | {r.get('duration_min', '')} | {msg[:400]} |")
    figs = sorted((out / "figures").glob("*.png"))
    if figs:
        lines += ["", "## Figures", ""] + [f"![{f.stem}](figures/{f.name})" for f in figs]
    (out / "summary.md").write_text("\n".join(lines) + "\n")


def parse_deadline(s: str) -> dt.datetime:
    now = dt.datetime.now()
    h, m = map(int, s.split(":"))
    d = now.replace(hour=h, minute=m, second=0, microsecond=0)
    return d if d > now else d + dt.timedelta(days=1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--deadline", default="07:30", help="HH:MM, next occurrence (local time of the container)")
    ap.add_argument("--deadline-epoch", type=float, default=None,
                    help="deadline as Unix time (run.sh passes it: the container clock may be UTC)")
    ap.add_argument("--quick", action="store_true", help="small time range, few epochs: about 1 hour")
    ap.add_argument("--only", default="", help="comma separated test ids or phases")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    deadline = dt.datetime.fromtimestamp(a.deadline_epoch) if a.deadline_epoch else parse_deadline(a.deadline)
    c = Campaign(out, deadline, a.quick, {x for x in a.only.split(",") if x} or None)
    by_id = {t.tid: t for t in TESTS}
    tests = [by_id[i] for i in ORDER]
    if c.only and "P12.store_unchanged" not in c.only:
        tests = [t for t in tests if t.tid != "P12.store_unchanged"]
    c.run(tests)
    print(f"Done: {out / 'summary.md'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
