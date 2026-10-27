"""Read-only access to the CORNERSTONE SQLite database (the GUI never writes to it).

Table naming used by the package:
- ``SHARPC_<version>_<DEF|NRT>``: SHARP CEA keywords, one row per HARP and T_OBS (12 min);
- ``NOAA_SHARPC_<DEF|NRT>_AR``: GOES X-ray flare events associated to HARPs (``fflux`` in W/m²);
- ``DF_<DEF|NRT>_<Cplus|Mplus|Xplus>_delta<h>_three_classes``: labels per (t_X, HARP).
CME catalogues are not part of the database.
"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from functools import lru_cache

import pandas as pd

TOBS_FMT = "%Y.%m.%d_%H:%M:%S_TAI"
GOES_CLASS = [("X", 1e-4), ("M", 1e-5), ("C", 1e-6), ("B", 1e-7), ("A", 0.0)]


def connect(db: str) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{db}?mode=ro", uri=True, check_same_thread=False)


def tables(db: str) -> list[str]:
    return list(_tables(db, _mtime(db)))


def _mtime(db: str) -> float:
    import os
    try:
        return os.path.getmtime(db)
    except OSError:
        return 0.0


@lru_cache(maxsize=8)
def _tables(db: str, _mt: float) -> tuple[str, ...]:
    with closing(connect(db)) as c:
        return tuple(r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"))


def columns(db: str, table: str) -> list[str]:
    with closing(connect(db)) as c:
        return [r[1] for r in c.execute(f'PRAGMA table_info("{table}")')]


def label_tables(db: str) -> pd.DataFrame:
    rows = []
    for t in tables(db):
        if t.startswith("DF_") and t.endswith("_three_classes"):
            parts = t.split("_")          # DF, DEF, Cplus, delta24, three, classes
            rows.append({"table": t, "table_type": parts[1], "label_type": parts[2],
                         "delta_hours": int(parts[3].replace("delta", ""))})
    return pd.DataFrame(rows)


def feature_tables(db: str) -> list[str]:
    return [t for t in tables(db) if t.startswith("SHARPC_")]


@lru_cache(maxsize=16)
def harp_index(db: str, table: str, _mt: float = 0.0) -> pd.DataFrame:
    """One row per HARP: first/last observation and number of records (full scan, cached)."""
    with closing(connect(db)) as c:
        df = pd.read_sql_query(f'SELECT HARPNUM, MIN(T_OBS1) AS start, MAX(T_OBS1) AS end, COUNT(*) AS n_obs, '
                               f'MAX(NOAA_AR) AS NOAA_AR FROM "{table}" GROUP BY HARPNUM', c)
    df["start"] = pd.to_datetime(df["start"], errors="coerce")
    df["end"] = pd.to_datetime(df["end"], errors="coerce")
    return df


def harp_series(db: str, table: str, harpnum: int, features: list[str] | None = None) -> pd.DataFrame:
    cols = columns(db, table)
    keep = [f for f in (features or cols) if f in cols and f not in ("T_OBS", "T_OBS1", "HARPNUM")]
    sel = ", ".join(f'"{x}"' for x in ["T_OBS1", *keep])
    with closing(connect(db)) as c:
        df = pd.read_sql_query(f'SELECT {sel} FROM "{table}" WHERE HARPNUM = ? ORDER BY T_OBS', c, params=(int(harpnum),))
    df["time"] = pd.to_datetime(df.pop("T_OBS1"), errors="coerce")
    return df


def harp_flares(db: str, table_type: str, harpnum: int) -> pd.DataFrame:
    t = f"NOAA_SHARPC_{table_type}_AR"
    if t not in tables(db):
        return pd.DataFrame(columns=["tstart", "tmax", "tend", "part1", "fflux"])
    with closing(connect(db)) as c:
        df = pd.read_sql_query(f'SELECT tstart, tmax, tend, part1, fclass, fflux, NOAA_AR FROM "{t}" '
                               f'WHERE HARPNUM = ? ORDER BY tmax', c, params=(int(harpnum),))
    for col in ("tstart", "tmax", "tend"):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def harp_labels(db: str, table: str, harpnum: int) -> pd.DataFrame:
    with closing(connect(db)) as c:
        df = pd.read_sql_query(f'SELECT t_X, label, label_u, n_a FROM "{table}" WHERE HARPn = ? ORDER BY t_X',
                               c, params=(int(harpnum),))
    df["t_X"] = pd.to_datetime(df["t_X"], errors="coerce")
    return df


def labels_from_artifact(labels_db: str, table: str, harpnum: int) -> pd.DataFrame:
    with closing(sqlite3.connect(f"file:{labels_db}?mode=ro", uri=True)) as c:
        df = pd.read_sql_query(f'SELECT t_X, label, label_u, n_a FROM "{table}" WHERE HARPn = ? ORDER BY t_X',
                               c, params=(int(harpnum),))
    df["t_X"] = pd.to_datetime(df["t_X"], errors="coerce")
    return df


def goes_class(flux: float) -> str:
    for letter, lo in GOES_CLASS:
        if flux >= lo:
            return letter
    return "A"
