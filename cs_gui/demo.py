"""Synthetic data store for the DEMO mode and the automated tests.

!!! The values generated here are NOT scientific data. !!!
They only reproduce the *layout* of the CORNERSTONE Parquet store (SHARP DEF/NRT records, GOES
events, HARP <-> NOAA map), written with the package's own ``cornerstone.data`` API, so that the
real labeling, dataset and training code can run end-to-end on a small, fast input. A latent
"activity" signal drives both a few features and the flare rate, so trained models show some
skill: this is a property of the generator, not of the Sun.

    python -m cs_gui.demo --root DIR [--harps N] [--seed S]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEMO_NOTE = "SYNTHETIC DATA GENERATED FOR DEMO/TEST - NOT SOLAR OBSERVATIONS"
DEMO_MARKER = "DEMO_INFO.json"


def build_demo_store(root: str | Path, n_harps: int = 36, seed: int = 7, year: int = 2014) -> dict:
    """Create (replacing it) a synthetic store in ``root``; returns counts."""
    import pyarrow as pa

    from cornerstone.data import DataStore, goes, harp_noaa, sharp
    from cornerstone.data.store import block_start

    root = Path(root)
    if root.exists():
        shutil.rmtree(root)
    store = DataStore(root)
    rng = np.random.default_rng(seed)
    rows, events, pairs = [], [], []
    for i in range(n_harps):
        harp, noaa = 1000 + i, 11900 + i
        start = pd.Timestamp(f"{year}-01-05", tz="UTC") + pd.Timedelta(days=float(rng.uniform(0, 330)))
        start = start.floor("12min") - pd.Timedelta(seconds=31)          # JSOC T_OBS (TAI) on the UTC scale
        n = int(rng.integers(5, 12) * 24 * 5)                           # 5-12 days at 12 min
        times = start + pd.to_timedelta(np.arange(n) * 12, unit="min")
        act = np.cumsum(rng.normal(0, 0.08, n))
        act = (act - act.min()) / (np.ptp(act) + 1e-9) * rng.uniform(0.5, 2.0)   # latent activity
        lat = rng.uniform(-30, 30)
        lon = rng.uniform(-70, -40) + np.arange(n) * 13.2 / (24 * 5)      # ~13 deg/day rotation
        base = {
            "LAT_MIN": lat - 3, "LAT_MAX": lat + 3, "LON_MIN": lon - 4, "LON_MAX": lon + 4,
            "USFLUX": 1e21 * (1 + 4 * act), "MEANGAM": 30 + 10 * act, "MEANGBT": 80 + 40 * act,
            "MEANGBZ": 90 + 30 * act, "MEANGBH": 50 + 25 * act, "MEANJZD": rng.normal(0, 0.3, n) + act,
            "TOTUSJZ": 5e12 * (1 + 3 * act), "MEANALP": rng.normal(0, 0.02, n), "MEANJZH": 0.002 * (1 + act),
            "TOTUSJH": 200 * (1 + 3 * act), "ABSNJZH": 20 * (1 + 2 * act), "SAVNCPP": 1e12 * (1 + 2 * act),
            "MEANPOT": 3000 * (1 + act), "TOTPOT": 1e22 * (1 + 3 * act), "MEANSHR": 30 + 15 * act,
            "SHRGT45": 10 + 20 * act, "R_VALUE": 2 + 2.5 * act, "CMASK": 6000 + 3000 * act,
        }
        feats = {k: (np.asarray(v, float) * rng.normal(1, 0.03, n) if np.ndim(v) else np.full(n, float(v)))
                 for k, v in base.items()}
        keep = rng.random(n) >= 0.03                                     # missing records, as in the real data
        df = pd.DataFrame({k: v[keep] for k, v in feats.items()})
        df.insert(0, "T_OBS", times[keep])
        df.insert(1, "HARPNUM", harp)
        df["QUALITY"] = 0
        df["NOAA_AR"], df["NOAA_NUM"], df["NOAA_ARS"] = noaa, 1, str(noaa)
        rows.append(df)
        pairs.append((harp, noaa))
        rate = 0.0004 + 0.006 * act ** 2                                  # flare rate grows with activity
        for j in np.flatnonzero(rng.random(n) < rate):
            flux = float(10 ** np.clip(rng.normal(-5.6 + 0.6 * act[j], 0.5), -6.9, -3.5))
            t = times[j].floor("min")
            k = len(events)
            events.append({"event_id": f"{t:%Y%m%d}-{k}-1", "event": f"{t:%Y%m%d}-{k}", "tstart": t - pd.Timedelta(minutes=8),
                           "tmax": t, "tend": t + pd.Timedelta(minutes=10), "satellite": "G16", "quality": "5",
                           "goes_class": None, "flux_reported": flux, "integrated_flux": np.nan,
                           "region_reported": noaa % 10000, "NOAA_AR": noaa, "location": None,
                           "lat": np.float32(lat), "lon": np.float32(lon[j])})
    rec = pd.concat(rows, ignore_index=True)
    rec = rec.astype({"QUALITY": "UInt32", "NOAA_AR": "Int32", "NOAA_NUM": "Int16", "NOAA_ARS": "string",
                      **{c: "float32" for c in sharp.FLOAT_KEYS}})
    from cornerstone.processing import flux_to_class
    ev = pd.DataFrame(events)
    ev["goes_class"] = [flux_to_class(f) for f in ev["flux_reported"]]
    ev = goes._typed(ev[goes.SCHEMA.names])
    for series in ("DEF", "NRT"):
        r = rec if series == "DEF" else rec.assign(QUALITY=rec["QUALITY"] | 0x400)  # NRT: cosmic-ray bit set
        files = {}
        for b, g in r.groupby(r["HARPNUM"].map(block_start)):
            files[f"harp_{b:06d}.parquet"] = store.write_sharp_block(series, int(b), g.sort_values(["HARPNUM", "T_OBS"]))
        store.record_dataset(f"sharp/{series}", files, source={"kind": "demo", "note": DEMO_NOTE})
    p = goes.year_path(store, year)
    store.record_dataset("goes/swpc_events", {p.name: store.write_table(
        p, pa.Table.from_pandas(ev, schema=goes.SCHEMA, preserve_index=False))}, source={"kind": "demo"})
    hp = harp_noaa.path(store)
    store.record_dataset("harp_noaa", {hp.name: store.write_table(hp, pa.Table.from_pandas(
        pd.DataFrame(pairs, columns=["HARPNUM", "NOAA_AR"]).astype("int32"), schema=harp_noaa.SCHEMA,
        preserve_index=False))}, source={"kind": "demo"})
    stats = {"harps": n_harps, "sharp_records": int(len(rec)), "flares": len(ev), "seed": seed}
    (root / DEMO_MARKER).write_text(json.dumps({"note": DEMO_NOTE, **stats}, indent=2))
    return stats


def is_demo_store(root: str | Path) -> bool:
    return (Path(root) / DEMO_MARKER).exists()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="cs_gui.demo")
    ap.add_argument("--root", required=True)
    ap.add_argument("--harps", type=int, default=36)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)
    print(json.dumps(build_demo_store(a.root, a.harps, a.seed)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
