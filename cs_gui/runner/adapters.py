"""Adapters between the GUI pipeline and the CORNERSTONE package.

Every adapter works *around* the package (monkeypatching module attributes inside the runner
process, preparing inputs, post-processing outputs) and never edits the package sources.
Each one is documented in docs/ARCHITECTURE.md ("Adattatori").
"""
from __future__ import annotations

import os
import random
import shutil
import sqlite3
import types
from pathlib import Path

LABELS_DB_ALIAS = "cs_gui_labels"


def label_table_name(table_type: str, label_type: str, delta_hours: int | str) -> str:
    """Name used by ``Cornerstone.splitting_labeling`` / ``create_hdf5_from_db``."""
    return f"DF_{table_type}_{label_type}_delta{delta_hours}_three_classes"


def readonly_sqlite_module(main_db: str, labels_db: str | None = None, view_map: dict | None = None):
    """Drop-in replacement for the ``sqlite3`` module used by ``cornerstone.cornerstone``.

    - the main database is always opened **read-only** (``mode=ro``): the package cannot
      modify the shared DB from the GUI;
    - if ``labels_db`` is given it is attached read-only and, for every ``view -> table`` in
      ``view_map``, a TEMP view shadows the table of the same name in the main DB (SQLite
      resolves unqualified names in the temp schema first). This lets
      ``create_hdf5_from_db`` read the labels produced by a GUI labeling artifact.
    """
    main_abs = os.path.abspath(main_db)

    def connect(path, *args, **kwargs):
        if os.path.abspath(str(path)) == main_abs:
            conn = sqlite3.connect(f"file:{main_abs}?mode=ro", uri=True)
        else:
            conn = sqlite3.connect(path, *args, **kwargs)
        if labels_db:
            conn.execute(f"ATTACH DATABASE ? AS {LABELS_DB_ALIAS}", (f"file:{os.path.abspath(labels_db)}?mode=ro",))
            for view, table in (view_map or {}).items():
                conn.execute(f'CREATE TEMP VIEW IF NOT EXISTS "{view}" AS SELECT * FROM {LABELS_DB_ALIAS}."{table}"')
        return conn

    return types.SimpleNamespace(connect=connect, Row=sqlite3.Row, Error=sqlite3.Error)


def patch_cornerstone_sqlite(shim) -> None:
    import cornerstone.cornerstone as cc
    cc.sqlite3 = shim


def build_labeling_db(main_db: str, labels_db: str, shtyp: str) -> dict:
    """Prepare a small SQLite DB on which ``splitting_labeling`` can run without touching the main DB.

    ``splitting_labeling`` only needs, from ``SHARPC_all_<shtyp>``, the first and last T_OBS of
    every HARP (``SELECT HARPNUM, MIN(T_OBS), MAX(T_OBS) ... GROUP BY HARPNUM``): two rows per
    HARP reproduce exactly the same result. The flare events are copied as ``NOAA_SHARPC_<shtyp>_AR``,
    the name the package reads, from the table chosen by ``cs_gui.db.flare_table``.
    """
    from ..db import flare_table
    src_table = flare_table(main_db, shtyp)
    if src_table is None:
        raise RuntimeError(f"No flare event table for {shtyp} HARPs in the database (NOAA_SHARPC_{shtyp}_AR)")
    src = sqlite3.connect(f"file:{os.path.abspath(main_db)}?mode=ro", uri=True)
    dst = sqlite3.connect(labels_db)
    try:
        noaa = f"NOAA_SHARPC_{shtyp}_AR"
        sharp = f"SHARPC_all_{shtyp}"
        cols = src.execute(f'PRAGMA table_info("{src_table}")').fetchall()
        dst.execute(f'CREATE TABLE "{noaa}" (' + ", ".join(f'"{c[1]}" {c[2]}'.rstrip() for c in cols) + ")")
        rows = src.execute(f'SELECT * FROM "{src_table}"').fetchall()
        if rows:
            dst.executemany(f'INSERT INTO "{noaa}" VALUES ({",".join("?" * len(rows[0]))})', rows)
        bounds = src.execute(f'SELECT HARPNUM, MIN(T_OBS), MAX(T_OBS) FROM "{sharp}" GROUP BY HARPNUM').fetchall()
        dst.execute(f'CREATE TABLE "{sharp}" (HARPNUM INTEGER, T_OBS TEXT)')
        dst.executemany(f'INSERT INTO "{sharp}" VALUES (?, ?)',
                        [(h, t) for h, a, b in bounds for t in (a, b)])
        dst.commit()
        return {"n_flare_events": len(rows), "n_harps": len(bounds)}
    finally:
        src.close()
        dst.close()


def snapshot_label_table(main_db: str, labels_db: str, table: str) -> int:
    """Copy an existing label table (or view) of the main DB into the artifact DB (read-only on main)."""
    src = sqlite3.connect(f"file:{os.path.abspath(main_db)}?mode=ro", uri=True)
    dst = sqlite3.connect(labels_db)
    try:
        obj = src.execute("SELECT type, sql FROM sqlite_master WHERE type IN ('table', 'view') AND name=?",
                          (table,)).fetchone()
        if obj is None:
            raise RuntimeError(f"Label table '{table}' not found in the database")
        if obj[0] == "table":
            dst.execute(obj[1])
        else:
            # a view refers to tables that are not copied: materialize it as a table with its columns
            cols = src.execute(f'PRAGMA table_info("{table}")').fetchall()
            dst.execute(f'CREATE TABLE "{table}" (' + ", ".join(f'"{c[1]}" {c[2]}'.rstrip() for c in cols) + ")")
        rows = src.execute(f'SELECT * FROM "{table}"').fetchall()
        if rows:
            dst.executemany(f'INSERT INTO "{table}" VALUES ({",".join("?" * len(rows[0]))})', rows)
        dst.commit()
        return len(rows)
    finally:
        src.close()
        dst.close()


def set_seeds(seed: int) -> None:
    """The package does not seed training; the runner seeds python, numpy and torch."""
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    os.environ["PYTHONHASHSEED"] = str(seed)


def apply_model_overrides(model: str, params: dict, reg_grid: list | None) -> None:
    """Hyper-parameters: ``train_models`` reads ``MODEL_CONFIGS`` directly (JSON
    ``MODEL_CONFIG_OVERRIDES`` are overwritten by it in ``ModelManager.create_model``), so the
    dict is updated in place inside the runner process."""
    from cornerstone.ml.model_configs import MODEL_CONFIGS
    from cornerstone.ml.regularization_configs import REGULARIZATION_PARAMS
    MODEL_CONFIGS.setdefault(model, {}).update(params or {})
    if reg_grid:
        REGULARIZATION_PARAMS.setdefault(model, {})["regularization_params"] = list(reg_grid)


def register_dataset(key: str, name: str, data_prefix: str, input_dim: int, base_path: str) -> None:
    """``train_models`` resolves datasets through ``AVAILABLE_DATASETS``: register the split."""
    from cornerstone.ml.utils.dataset_config import AVAILABLE_DATASETS
    AVAILABLE_DATASETS[key] = {"name": name, "description": "registered by cornerstone-gui",
                               "data_prefix": data_prefix, "input_dim": int(input_dim), "base_path": base_path}


def cupy_cpu_fallback() -> bool:
    """ROCKET/MINIROCKET compute on CPU (sktime + sklearn; their cupy code is commented out) but
    still call ``cupy._default_memory_pool.free_all_blocks()``, which raises without an NVIDIA GPU.
    Without a usable CUDA device the pool is replaced by a no-op. Returns True if patched."""
    try:
        import cupy
    except ImportError:
        return False
    try:
        cupy.cuda.runtime.getDeviceCount()
        return False
    except Exception:  # no NVIDIA driver/device
        cupy._default_memory_pool = types.SimpleNamespace(free_all_blocks=lambda: None)
        return True


def set_device(device: str) -> str:
    from cornerstone.ml.utils import config as cfg
    import torch
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available on this machine")
    cfg.DEVICE_CONFIG["device"] = device
    return device


def make_best_pt_models(path_manager, windows: list[str], metrics: dict) -> dict:
    """PyTorch models: training saves only ``*_fold{k}.pt`` while ``run_test`` loads ``*_best.pt``.
    Copy the fold selected by the package (``best_fold`` in the metrics file)."""
    done = {}
    for w in windows:
        if w not in metrics:
            continue
        k = int(metrics[w]["best_fold"])
        src = Path(path_manager.get_model_path(w, fold=k, model_type="PT"))
        dst = Path(path_manager.get_model_path(w, best=True, model_type="PT"))
        if src.exists():
            shutil.copyfile(src, dst)
            done[w] = {"fold": k, "file": dst.name}
    return done
