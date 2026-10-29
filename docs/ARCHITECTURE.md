# Package analysis and GUI architecture

## 1. Analysis of the CORNERSTONE repository (`cornerstone_db_package`)

Analysis made on commit `897a662` (branch `main` of CORNERSTONE-2025). The GUI does not modify the package.

### Public API (`cornerstone.Cornerstone`)

| Method | What it does | Input → output |
|---|---|---|
| `db_update()` | Syncs the local DB with remote dumps on Google Drive | network → local SQLite |
| `splitting_labeling(shtyp, class0, delta_hours, db_type, save_csv)` | Labels per (t_X, HARP) counting GOES flares before/after t_X | tables `SHARPC_all_*`, `NOAA_SHARPC_*_AR` → table `DF_<shtyp>_<class0>_delta<h>_three_classes` (**appended** to the DB) |
| `create_hdf5_from_db(label_type, data_version, table_type, window_hours, buffer_minutes, sampling_interval, test_mode, test_size, random_seed, db_type)` | SHARP time series in the `window_hours` before every t_X | labels + `SHARPC_<version>_<type>` → HDF5 + `missing_data_log.csv` |
| `create_dataset_splits(config)` | Feature selection, missing-data/time/CMASK filters, random or chronological split with k-fold grouped by HARP | HDF5 → `splits/*_cv_k_{train,val}.h5`, `*_test.h5`, plots, config |
| `train_models(train_config_path)` | Cross-validation for every window (3h/6h/12h/24h) | split + JSON → per-fold models, JSON metrics, plots |
| `test_model(test_config_path)` | Evaluation of the `_best` model on the test set | models → metrics (ClearML log only) |

### Data (SQLite)

- `SHARPC_clean_{DEF,NRT}` (17 normalized features `_N1/_N2`) and `SHARPC_all_{DEF,NRT}` (22 features): 12 min cadence.
- `NOAA_SHARPC_{DEF,NRT}_AR`: GOES XRA flares associated with the HARPs (`fflux` in W/m², `part1` = class).
- `DF_*_three_classes`: labels `YES`/`NO1..NO4`, `label_u` = flare class (C/M/X).
- **CME: no table.** The package only handles flares; the GUI states it and does not simulate CME data.

### Models (`cornerstone.ml`)

- PyTorch (`PT_MODELS`): LSTM, TRANSFORMER, INCEPTIONTIME, ONEFITSALL and the Time-Series-Library models
  (Autoformer, DLinear, FEDformer, LightTS, Nonstationary_Transformer, Pyraformer, Reformer, TimesNet, Informer).
- sklearn-like (`SK_MODELS`): XGBOOST, ROCKET, MINIROCKET (the last two use **cupy → GPU required**).
- Hyperparameters: `cornerstone/ml/model_configs.py`; SK regularization grid: `regularization_configs.py`.
- PT training: bce/focal/asymmetric_focal/curriculum loss, Cosine/Plateau scheduler, early stopping on
  tss/val_loss/auc/f1/mcc/hss, threshold optimization on validation, optional Platt/temperature calibration.
- SK training: C grid × folds, C chosen by mean validation TSS, threshold optimized on validation (`model.threshold_`).
- Metrics (`metrics.compute_metrics`): F1, AUC, MCC, TSS (= TPR − FPR), HSS, confusion matrix.

### Gaps found and how the GUI handles them

| Gap | Handling |
|---|---|
| `splitting_labeling` **appends** to the shared DB (re-running duplicates the rows) and writes a CSV in the cwd | Labeling on a **private DB of the artifact** (adapter, §3) |
| `create_hdf5_from_db` reads labels and features from the same DB | Main DB opened **read-only** + artifact labels mounted as a temporary view |
| The JSON `MODEL_CONFIG_OVERRIDES` are overwritten by `MODEL_CONFIGS` in `ModelManager.create_model` | The runner updates `MODEL_CONFIGS` in its own process |
| `train_models` requires the dataset to be registered in `AVAILABLE_DATASETS` | The runner registers the artifact split |
| No seed set during training | The runner sets the python/numpy/torch seeds (recorded in the config) |
| PyTorch models: the `*_best.pt` required by `run_test` is never saved | Copy of the `best_fold` chosen by the package (recorded in the manifest) |
| `run_test`: fixed 0.5 threshold for PT, corrupted return value (reassigns `test_metrics` in the loop), no per-sample predictions | GUI inference built from the package building blocks (`FlareHDF5Dataset`, `ModelManager`, `collect_logits`, `compute_metrics`), explicit threshold, saved predictions |
| `calibration.kind` key in `train_config.json` ignored (the code reads `type`) | The GUI writes `type` and points it out |
| Fixed labeling date range (1996–2025) | Documented; cannot be changed without changing the package |
| No CME data | Stated in the GUI; nothing simulated |

## 2. Architecture

```
┌──────────── Streamlit (GUI environment: streamlit, plotly, pandas, h5py, sklearn) ─────────┐
│ app/pages/*.py  ── cs_gui.ui (components) ── cs_gui.pipeline (validation, variants,        │
│                                               reproduction) ── cs_gui.store (artifacts)     │
│                                               cs_gui.jobs (background processes)            │
│                                               cs_gui.db (read-only SQLite), metrics, h5info │
└──────────────────────────────────┬─────────────────────────────────────────────────────────┘
                                   │ subprocess (own process group, log to file)
┌──────────────────────────────────▼──────── CORNERSTONE environment (torch, xgboost, …) ────┐
│ python -m cs_gui.runner  ── steps.py (one step per artifact type)                          │
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
- **ClearML offline**: the runner forces `CLEARML_OFFLINE_MODE=1` so that runs from the GUI are never published.

## 3. Adapters (`cs_gui/runner/adapters.py`)

1. `readonly_sqlite_module`: replaces `cornerstone.cornerstone.sqlite3` in the runner process. The main DB
   is opened with `mode=ro` (writes are impossible); the artifact labels DB is `ATTACH`ed and a
   `TEMP VIEW` with the name expected by the package takes precedence over the table of the same name in the main DB.
2. `build_labeling_db`: private DB with a copy of `NOAA_SHARPC_<t>_AR` and, for `SHARPC_all_<t>`, only the first
   and last `T_OBS` of every HARP. The query used by `splitting_labeling`
   (`SELECT HARPNUM, MIN(T_OBS), MAX(T_OBS) … GROUP BY HARPNUM`) gives the same result (automated test).
3. `apply_model_overrides`, `register_dataset`, `set_device`, `set_seeds`: in-process configuration.
4. `make_best_pt_models`: copies `*_fold{best_fold}.pt` → `*_best.pt`.

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

- ROCKET/MINIROCKET require a CUDA GPU (cupy): on machines without a GPU the GUI blocks them before training.
- Bit-for-bit reproducibility is not guaranteed: non-deterministic PyTorch operations, multiple threads,
  different dependency versions. The GUI records the package commit (with a "local changes" flag),
  versions and seeds, and warns if the package changed since the original run.
- Generated labeling always covers 1996-01-01 → 2025-04-30 (fixed in the package) and can take
  tens of minutes.
- The "test_*" metrics shown in training are those the package computes per fold; the reference
  evaluation is the one of the inference artifact.
- Predictions of PyTorch models use **uncalibrated** sigmoid probabilities (like `run_test`); the CV threshold
  used is the fold `best_threshold`, optimized on the same uncalibrated probabilities.
- `db_update()` (sync from Google Drive) is not exposed: it modifies the shared DB and needs the network.
