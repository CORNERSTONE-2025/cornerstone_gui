"""Adapters between the GUI pipeline and the CORNERSTONE package.

Every adapter works *around* the package (monkeypatching module attributes inside the runner
process, preparing inputs, post-processing outputs) and never edits the package sources.
Each one is documented in docs/ARCHITECTURE.md ("Adattatori").
"""
from __future__ import annotations

import os
import random
import shutil
import types
from pathlib import Path


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
