# Package analysis and GUI architecture

## 1. Analysis of the CORNERSTONE repository (`cornerstone_db_package`)

Analysis made on commit `897a662` (branch `main` of CORNERSTONE-2025). The GUI does not modify the package.

### Public API (`cornerstone.Cornerstone`)

| Method | What it does | Input → output |
|---|---|---|
| `python -m cornerstone.data update / bootstrap-*` (`cornerstone.data`) | Download/update of the Parquet store from JSOC and NOAA SWPC, or creation from the snapshot / former dump | network → `<data_root>/raw/**.parquet` + `manifest.json` |
| `cornerstone.labeling.labels_and_flares(store, series, LabelConfig, QualityFilter, FlareConfig)` | Parametric labels per (t_X, HARP): prediction label YES/NO and stratification class (C/M/X… or NO1–NO4) | store → labels, flare → HARP association, metadata |
| `cornerstone.dataset.build_dataset(store, series, labels, path, features, quality, window_hours, sampling_minutes, …)` | SHARP time series in the `window_hours` before every t_X | store + labels → HDF5 (`label_<name>`, `label_u_<name>`) + `*.missing.csv` |
| `create_dataset_splits(config)` | Feature selection, missing-data/time/CMASK filters, random or chronological split with k-fold grouped by HARP | HDF5 → `splits/*_cv_k_{train,val}.h5`, `*_test.h5`, plots, config |
| `train_models(train_config_path)` | Cross-validation for every window (3h/6h/12h/24h) | split + JSON → per-fold models, JSON metrics, plots |
| `test_model(test_config_path)` | Evaluation of the `_best` model on the test set | models → metrics (ClearML log only) |

### Data (Parquet store, `cornerstone.data`)

- `raw/sharp/{DEF,NRT}`: every SHARP record as published by JSOC (28 keywords, 12 min cadence, UTC).
  Which records are used (former `SHARPC_all` / `SHARPC_clean` selections, or custom) is a parameter
  of labeling and dataset (`QualityFilter`); the `_N1/_N2` features are computed on the fly.
- `raw/goes`: GOES X-ray events from the NOAA SWPC edited reports (2010→), flux as reported; the
  calibration (science scale by default) and the flare → HARP association are labeling parameters
  (DEF: JSOC HARP ↔ NOAA map; NRT: `NOAA_ARS` keyword).
- **CME: no catalogue.** The package only handles flares; the GUI states it and does not simulate CME data.

### Models (`cornerstone.ml`)

- PyTorch (`PT_MODELS`): LSTM, TRANSFORMER, INCEPTIONTIME, ONEFITSALL and the Time-Series-Library models
  (Autoformer, DLinear, FEDformer, LightTS, Nonstationary_Transformer, Pyraformer, Reformer, TimesNet, Informer).
- sklearn-like (`SK_MODELS`): XGBOOST, ROCKET, MINIROCKET. ROCKET/MINIROCKET import cupy but compute on CPU (sktime + sklearn;
  their GPU code is commented out); only `free_all_blocks()` is still called.
- Hyperparameters: `cornerstone/ml/model_configs.py`; SK regularization grid: `regularization_configs.py`.
- PT training: bce/focal/asymmetric_focal/curriculum loss, Cosine/Plateau scheduler, early stopping on
  tss/val_loss/auc/f1/mcc/hss, threshold optimization on validation, optional Platt/temperature calibration.
- SK training: C grid × folds, C chosen by mean validation TSS, threshold optimized on validation (`model.threshold_`).
- Metrics (`metrics.compute_metrics`): F1, AUC, MCC, TSS (= TPR − FPR), HSS, confusion matrix.

### Gaps found and how the GUI handles them

| Gap | Handling |
|---|---|
| The JSON `MODEL_CONFIG_OVERRIDES` are overwritten by `MODEL_CONFIGS` in `ModelManager.create_model` | The runner updates `MODEL_CONFIGS` in its own process |
| `train_models` requires the dataset to be registered in `AVAILABLE_DATASETS` | The runner registers the artifact split |
| No seed set during training | The runner sets the python/numpy/torch seeds (recorded in the config) |
| PyTorch models: the `*_best.pt` required by `run_test` is never saved | Copy of the `best_fold` chosen by the package (recorded in the manifest) |
| `run_test`: fixed 0.5 threshold for PT, corrupted return value (reassigns `test_metrics` in the loop), no per-sample predictions | GUI inference built from the package building blocks (`FlareHDF5Dataset`, `ModelManager`, `collect_logits`, `compute_metrics`), explicit threshold, saved predictions |
| `calibration.kind` key in `train_config.json` ignored (the code reads `type`) | The GUI writes `type` and points it out |
| ROCKET/MINIROCKET call `cupy._default_memory_pool.free_all_blocks()`, which fails without an NVIDIA GPU | `cupy_cpu_fallback` replaces the pool with a no-op when no CUDA device is found |
| ONEFITSALL builds its layers on the first forward pass, so a fresh model cannot load its weights (`run_test` fails too) | GUI inference runs one batch through the model before `load_state_dict` |
| No CME data | Stated in the GUI; nothing simulated |

## 2. Architecture

```
┌──────────── Streamlit (GUI environment: streamlit, plotly, pandas, h5py, sklearn) ─────────┐
│ app/views/*.py  ── cs_gui.ui (components) ── cs_gui.pipeline (validation, variants,        │
│                                               reproduction) ── cs_gui.store (artifacts)     │
│                                               cs_gui.jobs (background processes)            │
│                                    cs_gui.datastore (read-only Parquet), metrics, h5info │
└──────────────────────────────────┬─────────────────────────────────────────────────────────┘
                                   │ subprocess (own process group, log to file)
┌──────────────────────────────────▼──────── CORNERSTONE environment (torch, xgboost, …) ────┐
│ python -m cs_gui.runner  ── steps.py (one step per artifact type)                          │
│ python -m cornerstone.data ── data download/update (cs_gui.datasync, Data page)            │
│                          ── adapters.py (integration without modifying the package)        │
│                          ── cornerstone.Cornerstone / cornerstone.ml.*                     │
└────────────────────────────────────────────────────────────────────────────────────────────┘
```

Choices:
- **Streamlit**: pure Python, compatible with the scientific stack, dynamic forms and interactive Plotly
  plots; long operations do not block the interface because they run in subprocesses.
- **Two environments (local install)**: the GUI stays light; the runner uses the exact environment of the
  package (same torch/cupy/xgboost versions used in research). In the **Docker image** a single environment
  contains both (see [DOCKER.md](DOCKER.md)).
- **State on disk** (JSON manifest per artifact + job records): it survives GUI restarts; a job that died
  without updating its state is marked `failed` at the next access (`JobManager.reconcile`).
- **Logs**: the runner and the download write the package stdout/stderr unchanged to a file (`dup2`,
  unbuffered). `cs_gui.termlog` reads it incrementally and interprets it like a terminal (`\r` redraws
  the current line, as tqdm does; `\n` adds a line; ANSI cursor/erase sequences), so a progress bar is
  one updated line. `ui.log_box` shows it in a `st.components.v2` component (no iframe): the DOM
  survives the 3 s refreshes, so the reader's scroll position and selection are kept, the view follows
  the end only while scrolled to the bottom, and *Copy all / Select all / Download* act on the whole
  log (the last 50,000 lines are shown; the file on disk is always complete).
- **ClearML offline**: the runner forces `CLEARML_OFFLINE_MODE=1` so that runs from the GUI are never published.

## 3. Adapters (`cs_gui/runner/adapters.py`)

Labeling and dataset call the package data modules directly (no adapter): they read the store and
write only into the artifact folder.

1. `apply_model_overrides`, `register_dataset`, `set_device`, `set_seeds`: in-process configuration.
2. `make_best_pt_models`: copies `*_fold{best_fold}.pt` → `*_best.pt`.
3. `cupy_cpu_fallback`: no-op cupy memory pool when no CUDA device is available (ROCKET/MINIROCKET on CPU).

## 4. Data leakage: rules applied

- Split: only the package strategies are used, both grouped by HARP; after the split the GUI checks
  that test and train/val share neither HARPs nor samples (`summary.leakage`) and **blocks training** if they do.
- The package preprocessing is per sample (selection/filters, no normalization learned from the data),
  so no transformation is fitted on test data.
- Inference: every prediction is flagged if its HARP appears in the model train/val; by default these
  rows are **excluded from the metrics**. Using train/val of the same split as "unseen events" is rejected
  by the validation.
- Random split: test and training are not separated in time. The GUI points it out and suggests the
  chronological split (cutoff or window) for an operational evaluation.

## 5. Remaining limitations

- PyTorch models use a GPU only if the runner interpreter has a matching torch build (CUDA for NVIDIA,
  ROCm for AMD; ROCm devices appear as `cuda` in torch). Otherwise select `cpu`.
- Bit-for-bit reproducibility is not guaranteed: non-deterministic PyTorch operations, multiple threads,
  different dependency versions. The GUI records the package commit (with a "local changes" flag),
  versions and seeds, and warns if the package changed since the original run.
- The "test_*" metrics shown in training are those the package computes per fold; the reference
  evaluation is the one of the inference artifact.
- Predictions of PyTorch models use **uncalibrated** sigmoid probabilities (like `run_test`); the CV threshold
  used is the fold `best_threshold`, optimized on the same uncalibrated probabilities.
- The Data page is the only place that writes the data store: `python -m cornerstone.data` runs as its own
  background process (`cs_gui.datasync`) and needs the network (JSOC, NOAA SWPC, Google Drive for the snapshot).
