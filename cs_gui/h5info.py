"""Read-only summaries of the HDF5 files written by CORNERSTONE.

Layout written by ``create_hdf5_from_db`` / ``save_split_to_h5``::

    X           (N, T, F) float32, NaN = missing observation
    Y           (N,) bytes  binary label: YES / NO1..NO4
    Y_u         (N,) bytes  (split files) flare class of the label: C/M/X or NO*
    label_<L>   (N,) bytes  (raw files) same as Y, named after the label type
    HARPNUM     (N,) int
    timestamps  (N, T) bytes  ``YYYY-mm-ddTHH:MM:SS`` (empty when missing)
    attrs: feature_names (comma separated), table_name, database_version (raw files)
"""
from __future__ import annotations

from collections import Counter

import h5py
import numpy as np


def _decode(arr) -> np.ndarray:
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in arr])


def feature_names(path: str) -> list[str]:
    with h5py.File(path, "r") as f:
        fn = f.attrs.get("feature_names", "")
    fn = fn.decode() if isinstance(fn, bytes) else str(fn)
    return [x for x in fn.split(",") if x]


def label_keys(path: str) -> list[str]:
    """Label types available in a raw HDF5 (e.g. ['Cplus_delta24'])."""
    with h5py.File(path, "r") as f:
        return [k[len("label_"):] for k in f.keys() if k.startswith("label_") and not k.startswith("label_u_")]


def issue_times(timestamps: np.ndarray) -> np.ndarray:
    """Last non-empty timestamp of each sample (the forecast issuing time)."""
    ts = timestamps.astype(str) if timestamps.dtype.kind == "S" else timestamps
    out = []
    for row in ts:
        nonempty = [t for t in row if t and t != "nan"]
        out.append(nonempty[-1] if nonempty else "")
    return np.array(out)


def time_bounds(timestamps: np.ndarray) -> tuple[str, str]:
    flat = timestamps.astype(str).ravel() if timestamps.dtype.kind == "S" else timestamps.ravel()
    flat = flat[(flat != "") & (flat != "nan")]
    if flat.size == 0:
        return "", ""
    # ISO strings sort chronologically; numpy>=2 has no min/max loop for unicode arrays
    return str(min(flat.tolist())), str(max(flat.tolist()))


def summarize(path: str, max_rows_for_nan: int = 20000) -> dict:
    """Numbers shown in the GUI for a dataset file (samples, features, time span, classes, NaN)."""
    with h5py.File(path, "r") as f:
        n, t, nf = f["X"].shape
        harp = f["HARPNUM"][:]
        y = _decode(f["Y"][:]) if "Y" in f else np.array([])
        yu_key = "Y_u" if "Y_u" in f else next((k for k in f.keys() if k.startswith("label_u_")), None)
        yu = _decode(f[yu_key][:]) if yu_key else np.array([])
        ts = f["timestamps"][:]
        # NaN statistics on (a bounded number of) samples
        idx = np.arange(n) if n <= max_rows_for_nan else np.linspace(0, n - 1, max_rows_for_nan).astype(int)
        X = f["X"][np.sort(idx)] if n else np.zeros((0, t, nf))
        attrs = {k: (v.decode() if isinstance(v, bytes) else str(v)) for k, v in f.attrs.items() if k != "quality"}
    names = [x for x in attrs.get("feature_names", "").split(",") if x]
    nan_ratio = (np.isnan(X).mean(axis=(0, 1)) if X.size else np.zeros(nf)).tolist()
    empty = int((np.isnan(X).all(axis=(1, 2))).sum()) if X.size else 0
    start, end = time_bounds(ts)
    yes = int((y == "YES").sum())
    return {
        "n_samples": int(n),
        "timesteps": int(t),
        "n_features": int(nf),
        "feature_names": names,
        "time_start": start,
        "time_end": end,
        "n_active_regions": int(len(np.unique(harp))),
        "class_counts": dict(Counter(y.tolist())),
        "class_counts_u": dict(Counter(yu.tolist())),
        "n_positive": yes,
        "positive_rate": (yes / n) if n else 0.0,
        "nan_ratio_per_feature": dict(zip(names or [f"f{i}" for i in range(nf)], nan_ratio)),
        "nan_ratio_overall": float(np.mean(nan_ratio)) if nan_ratio else 0.0,
        "empty_samples_in_sample": empty,
        "nan_stats_on_samples": int(len(idx)),
        "attrs": attrs,
    }


def harps(path: str) -> set[int]:
    with h5py.File(path, "r") as f:
        return set(int(h) for h in f["HARPNUM"][:])


def sample_keys(path: str) -> set[tuple[int, str]]:
    """(HARPNUM, issuing time) pairs: identifies a forecast sample across files."""
    with h5py.File(path, "r") as f:
        h = f["HARPNUM"][:]
        it = issue_times(f["timestamps"][:])
    return set(zip(h.astype(int).tolist(), it.tolist()))
