"""Shared fixtures. Unit tests use only synthetic data in temporary folders.

Integration tests (``-m integration``) run the real CORNERSTONE package through the runner, on
the synthetic demo database; they are skipped if the CORNERSTONE checkout/virtualenv is missing.
"""
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cs_gui.settings import DEFAULT_CORNERSTONE_ROOT, Settings  # noqa: E402
from cs_gui.store import ArtifactStore  # noqa: E402

CORNERSTONE_ROOT = os.environ.get("CS_GUI_CORNERSTONE_ROOT", DEFAULT_CORNERSTONE_ROOT)


def cornerstone_available() -> bool:
    s = Settings(cornerstone_root=CORNERSTONE_ROOT)
    return (Path(CORNERSTONE_ROOT) / "cornerstone" / "__init__.py").exists() and Path(s.python).exists()


@pytest.fixture
def store(tmp_path):
    return ArtifactStore(tmp_path / "ws")


@pytest.fixture
def settings(tmp_path):
    return Settings(cornerstone_root=CORNERSTONE_ROOT, workspace=str(tmp_path / "ws"),
                    demo_workspace=str(tmp_path / "ws"), mode="demo", min_free_disk_gb=0.1)


def make_h5(path: Path, n=40, t=121, features=("A", "B", "C"), harps=None, pos_every=4, start="2012-01-01"):
    """Small HDF5 file with the CORNERSTONE layout (synthetic, for unit tests only)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, t, len(features))).astype("float32")
    X[0, :, :] = np.nan                                   # one fully empty sample
    harps = np.array(harps if harps is not None else [100 + i // 4 for i in range(n)])
    Y = np.array([b"YES" if i % pos_every == 0 else b"NO1" for i in range(n)])
    base = np.datetime64(start)
    ts = np.array([[str(base + np.timedelta64(i, "D") + np.timedelta64(12 * k, "m")).replace("T", " ")[:19]
                    for k in range(t)] for i in range(n)], dtype="S19")
    with h5py.File(path, "w") as f:
        f.create_dataset("X", data=X)
        f.create_dataset("Y", data=Y)
        f.create_dataset("Y_u", data=Y)
        f.create_dataset("HARPNUM", data=harps)
        f.create_dataset("timestamps", data=ts)
        f.attrs["feature_names"] = ",".join(features)
    return path


def run_and_wait(jobs, store, aid, timeout=1800):
    jobs.launch([aid])
    t0 = time.time()
    while store.get(aid)["status"] in ("pending", "running"):
        time.sleep(1.5)
        jobs.reconcile()
        if time.time() - t0 > timeout:
            raise TimeoutError(aid)
    return store.get(aid)


def dump(m) -> str:
    return json.dumps(m.get("run"), indent=1)
