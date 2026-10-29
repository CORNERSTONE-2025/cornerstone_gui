"""Synthetic database for the DEMO mode and the automated tests.

!!! The values generated here are NOT scientific data. !!!
They only reproduce the *schema* of the CORNERSTONE SQLite database (table and column names,
time formats) so that the real package functions can run end-to-end on a small, fast input.
A latent "activity" signal drives both a few features and the flare rate, so trained models
show some skill: this is a property of the generator, not of the Sun.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

DEMO_NOTE = "SYNTHETIC DATA GENERATED FOR DEMO/TEST - NOT SOLAR OBSERVATIONS"

ALL_FEATURES = ["LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX", "USFLUX", "MEANGAM", "MEANGBT", "MEANGBZ",
                "MEANGBH", "MEANJZD", "TOTUSJZ", "MEANALP", "MEANJZH", "TOTUSJH", "ABSNJZH", "SAVNCPP",
                "MEANPOT", "TOTPOT", "MEANSHR", "SHRGT45", "R_VALUE", "CMASK"]
CLEAN_FEATURES = ["LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX", "USFLUX_N1", "MEANGBT", "MEANGBZ_N2",
                  "MEANGBH_N2", "MEANJZD", "TOTUSJZ_N1", "MEANJZH", "TOTUSJH_N1", "ABSNJZH", "MEANPOT",
                  "SHRGT45", "R_VALUE", "CMASK"]
NOAA_COLS = ["enmb", "yyyymmdd", "event", "tstart", "tmax", "tend", "obs", "q", "type", "loc_frq", "part1",
             "fclass", "fdecim", "fflux", "part2", "regn", "position", "SHARP_NOAA", "HARPNUM", "NOAA_AR"]


def build_demo_db(path: str | Path, n_harps: int = 36, seed: int = 7, year: int = 2012,
                  table_types=("DEF",)) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    rng = np.random.default_rng(seed)
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE DEMO_INFO (note TEXT, seed INTEGER, n_harps INTEGER)")
    con.execute("INSERT INTO DEMO_INFO VALUES (?, ?, ?)", (DEMO_NOTE, seed, n_harps))
    stats = {}
    for tt in table_types:
        sharp_rows, flare_rows = [], []
        for i in range(n_harps):
            harp = 1000 + i
            start = pd.Timestamp(f"{year}-01-05") + pd.Timedelta(days=float(rng.uniform(0, 330)))
            start = start.floor("12min") + pd.Timedelta(seconds=4)
            n = int(rng.integers(5, 12) * 24 * 5)                  # 5-12 days at 12 min
            times = start + pd.to_timedelta(np.arange(n) * 12, unit="min")
            act = np.cumsum(rng.normal(0, 0.08, n))
            act = (act - act.min()) / (np.ptp(act) + 1e-9) * rng.uniform(0.5, 2.0)   # latent activity
            lat = rng.uniform(-30, 30)
            lon0 = rng.uniform(-70, -40)
            lon = lon0 + np.arange(n) * 13.2 / (24 * 5)            # ~13 deg/day rotation
            base = {
                "LAT_MIN": lat - 3, "LAT_MAX": lat + 3, "LON_MIN": lon - 4, "LON_MAX": lon + 4,
                "USFLUX": 1e21 * (1 + 4 * act), "MEANGAM": 30 + 10 * act, "MEANGBT": 80 + 40 * act,
                "MEANGBZ": 90 + 30 * act, "MEANGBH": 50 + 25 * act, "MEANJZD": rng.normal(0, 0.3, n) + act,
                "TOTUSJZ": 5e12 * (1 + 3 * act), "MEANALP": rng.normal(0, 0.02, n), "MEANJZH": 0.002 * (1 + act),
                "TOTUSJH": 200 * (1 + 3 * act), "ABSNJZH": 20 * (1 + 2 * act), "SAVNCPP": 1e12 * (1 + 2 * act),
                "MEANPOT": 3000 * (1 + act), "TOTPOT": 1e22 * (1 + 3 * act), "MEANSHR": 30 + 15 * act,
                "SHRGT45": 10 + 20 * act, "R_VALUE": 2 + 2.5 * act, "CMASK": 500 + 3000 * act,
            }
            feats = {k: np.asarray(v, float) * rng.normal(1, 0.03, n) if np.ndim(v) else np.full(n, v)
                     for k, v in base.items()}
            # missing observations (gaps), as in the real data
            gap = rng.random(n) < 0.03
            for k in feats:
                feats[k] = np.where(gap, np.nan, feats[k])
            df = pd.DataFrame(feats)
            df.insert(0, "HARPNUM", harp)
            df.insert(0, "T_OBS1", times.strftime("%Y-%m-%dT%H:%M:%S"))
            df.insert(0, "T_OBS", times.strftime("%Y.%m.%d_%H:%M:%S_TAI"))
            df["QUALITY"] = "0x00000000"
            df["NOAA_AR"] = 11300 + i
            df["NOAA_NUM"] = 1
            df["NOAA_ARS"] = str(11300 + i)
            sharp_rows.append(df)
            # flares: rate grows with activity
            rate = 0.0004 + 0.006 * act ** 2
            for j in np.flatnonzero(rng.random(n) < rate):
                logf = rng.normal(-5.6 + 0.6 * act[j], 0.5)
                flux = float(10 ** np.clip(logf, -6.9, -3.5))
                tmax = times[j]
                cls = "X" if flux >= 1e-4 else "M" if flux >= 1e-5 else "C" if flux >= 1e-6 else "B"
                mant = flux / {"X": 1e-4, "M": 1e-5, "C": 1e-6, "B": 1e-7}[cls]
                flare_rows.append({
                    "enmb": f"{tmax:%Y%m%d}-{len(flare_rows)}", "yyyymmdd": f"{tmax:%Y%m%d}",
                    "event": str(len(flare_rows)), "tstart": (tmax - pd.Timedelta(minutes=8)).strftime("%Y-%m-%dT%H:%M:%S"),
                    "tmax": tmax.strftime("%Y-%m-%dT%H:%M:%S"),
                    "tend": (tmax + pd.Timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%S"),
                    "obs": "G15", "q": "5", "type": "XRA", "loc_frq": "1-8A", "part1": f"{cls}{mant:.1f}",
                    "fclass": cls, "fdecim": round(mant, 1), "fflux": flux, "part2": "", "regn": str(11300 + i),
                    "position": "", "SHARP_NOAA": f"{harp}-{11300 + i}", "HARPNUM": harp, "NOAA_AR": 11300 + i})
        sharp = pd.concat(sharp_rows, ignore_index=True)
        cols_all = ["T_OBS", "T_OBS1", "HARPNUM", *ALL_FEATURES, "QUALITY", "NOAA_AR", "NOAA_NUM", "NOAA_ARS"]
        sharp[cols_all].to_sql(f"SHARPC_all_{tt}", con, index=False)
        clean = sharp.rename(columns={"USFLUX": "USFLUX_N1", "MEANGBZ": "MEANGBZ_N2", "MEANGBH": "MEANGBH_N2",
                                      "TOTUSJZ": "TOTUSJZ_N1", "TOTUSJH": "TOTUSJH_N1"})
        for c in ("USFLUX_N1", "TOTUSJZ_N1", "TOTUSJH_N1"):
            clean[c] = clean[c] / clean[c].max()
        for c in ("MEANGBZ_N2", "MEANGBH_N2"):
            clean[c] = clean[c] / 100.0
        clean[["T_OBS", "T_OBS1", "HARPNUM", *CLEAN_FEATURES, "QUALITY", "NOAA_AR", "NOAA_NUM", "NOAA_ARS"]] \
            .to_sql(f"SHARPC_clean_{tt}", con, index=False)
        pd.DataFrame(flare_rows, columns=NOAA_COLS).to_sql(f"NOAA_SHARPC_{tt}_AR", con, index=False)
        stats[tt] = {"sharp_rows": int(len(sharp)), "flares": len(flare_rows), "harps": n_harps}
    con.commit()
    con.close()
    return stats


def is_demo_db(path: str | Path) -> bool:
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as c:
            return c.execute("SELECT count(*) FROM sqlite_master WHERE name='DEMO_INFO'").fetchone()[0] == 1
    except sqlite3.Error:
        return False
