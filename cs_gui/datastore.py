"""Read-only access to the CORNERSTONE Parquet data store (the GUI never writes to it; updates run
in background jobs through ``cornerstone.data``, see ``cs_gui.datasync``).

Layout (``cornerstone.data.DataStore``)::

    <root>/manifest.json                     datasets, files, sources, update times
    <root>/raw/sharp/<DEF|NRT>/harp_*.parquet SHARP keywords, one row per HARP and T_OBS (UTC)
    <root>/raw/goes/swpc_events_<year>.parquet GOES X-ray events (NOAA SWPC)
    <root>/raw/harp_noaa/*.parquet           HARP <-> NOAA region map (JSOC)

Only pyarrow/pandas are needed: the GUI process does not import the package.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

SERIES = {"DEF": "DEF (definitive SHARP, hmi.sharp_720s)", "NRT": "NRT (near-real-time, hmi.sharp_720s_nrt)"}
# derived features (cornerstone.processing.features): KEY_N1 = KEY / CMASK, KEY_N2 = KEY / MEANGBT
DERIVED = ("USFLUX_N1", "TOTUSJZ_N1", "TOTUSJH_N1", "MEANGBZ_N2", "MEANGBH_N2")
META_COLS = ("T_OBS", "HARPNUM", "QUALITY", "NOAA_AR", "NOAA_NUM", "NOAA_ARS")
DATASET_NAMES = {"sharp/DEF": "SHARP DEF", "sharp/NRT": "SHARP NRT", "goes/swpc_events": "GOES flares (SWPC)",
                 "harp_noaa": "HARP ↔ NOAA map"}
OBSOLETE = ("reference/",)   # tables of the former dump in stores of earlier versions: not shown


def manifest(root: str) -> dict:
    p = Path(root) / "manifest.json"
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return {"datasets": {}}


def _mtime(root: str) -> float:
    try:
        return (Path(root) / "manifest.json").stat().st_mtime
    except OSError:
        return 0.0


def status(root: str) -> pd.DataFrame:
    """One row per dataset of the store: files, rows, size, time coverage, last update."""
    rows = []
    for name, d in sorted(manifest(root).get("datasets", {}).items()):
        if name.startswith(OBSOLETE):
            continue
        files = d.get("files", {})
        until = [f.get("max_t_obs") for f in files.values() if f.get("max_t_obs")]
        src = d.get("source") or {}
        rows.append({
            "dataset": DATASET_NAMES.get(name, name), "key": name, "files": len(files),
            "rows": sum(int(f.get("rows") or 0) for f in files.values()),
            "MB": round(sum(int(f.get("bytes") or 0) for f in files.values()) / 1e6, 1),
            "data until": (max(until)[:16] if until else _years(files)),
            "updated": (d.get("updated_at") or "")[:16], "source": src.get("kind") or "",
        })
    return pd.DataFrame(rows, columns=["dataset", "key", "files", "rows", "MB", "data until", "updated", "source"])


def _years(files: dict) -> str:
    years = sorted(n.rsplit("_", 1)[-1].split(".")[0] for n in files if n.rsplit("_", 1)[-1][:4].isdigit())
    return f"{years[0]}–{years[-1]}" if years else ""


def sharp_files(root: str, series: str) -> list[str]:
    d = Path(root) / "raw" / "sharp" / series
    return sorted(str(p) for p in d.glob("harp_*.parquet")) if d.exists() else []


def has_series(root: str, series: str) -> bool:
    return bool(sharp_files(root, series))


def _dataset(root: str, series: str) -> ds.Dataset:
    files = sharp_files(root, series)
    if not files:
        raise FileNotFoundError(f"No SHARP {series} data in the store {root}")
    return ds.dataset(files, format="parquet")


def sharp_columns(root: str, series: str) -> list[str]:
    """Raw keywords and derived features available for plots."""
    names = [n for n in _dataset(root, series).schema.names if n not in META_COLS]
    return names + [d for d in DERIVED if d.rsplit("_", 1)[0] in names]


def harp_index(root: str, series: str) -> pd.DataFrame:
    """One row per HARP: first/last observation, number of records, NOAA region (cached)."""
    return _harp_index(root, series, _mtime(root)).copy()


@lru_cache(maxsize=8)
def _harp_index(root: str, series: str, _mt: float) -> pd.DataFrame:
    t = _dataset(root, series).to_table(columns=["HARPNUM", "T_OBS", "NOAA_AR"]).to_pandas()
    g = t.groupby("HARPNUM")
    out = pd.DataFrame({"start": _naive_utc(g["T_OBS"].min()), "end": _naive_utc(g["T_OBS"].max()), "n_obs": g.size(),
                        "NOAA_AR": g["NOAA_AR"].max()}).reset_index()
    return out


def harp_series(root: str, series: str, harpnum: int, features: list[str] | None = None) -> pd.DataFrame:
    """Records of one HARP (``time`` + the requested features, derived ones computed on the fly)."""
    avail = _dataset(root, series).schema.names
    feats = features or [c for c in avail if c not in META_COLS]
    raw = []
    for f in feats:
        if f in DERIVED:
            raw += [f.rsplit("_", 1)[0], "CMASK" if f.endswith("_N1") else "MEANGBT"]
        elif f in avail:
            raw.append(f)
    cols = list(dict.fromkeys(["T_OBS", *raw]))
    df = _dataset(root, series).to_table(columns=cols, filter=ds.field("HARPNUM") == int(harpnum)).to_pandas()
    df = df.sort_values("T_OBS")
    for f in feats:
        if f in DERIVED:
            den = df["CMASK" if f.endswith("_N1") else "MEANGBT"].astype(float)
            df[f] = df[f.rsplit("_", 1)[0]].astype(float) / den.where(den != 0)
    df["time"] = _naive_utc(df.pop("T_OBS"))
    return df[["time", *[f for f in feats if f in df.columns]]]


def _naive_utc(t: pd.Series) -> pd.Series:
    """UTC times without time zone (what plotly and the HDF5 timestamps use)."""
    t = pd.to_datetime(t, utc=True)
    return t.dt.tz_convert(None)


def naive_times(df: pd.DataFrame, cols) -> pd.DataFrame:
    return df.assign(**{c: _naive_utc(df[c]) for c in cols if c in df.columns})
