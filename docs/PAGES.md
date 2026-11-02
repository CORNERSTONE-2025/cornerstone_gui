# Pages and parameters

Every option listed maps to a package feature (source in brackets).

## Home and status
Workflow diagram, artifact counts by type and status, active jobs, environment status (package
commit, CUDA, DB, disk space), real database download/update (`Cornerstone.db_update`, live log,
cancel, switch to Real mode), path and mode configuration, demo DB generation.

## Labeling (`Cornerstone.splitting_labeling`)
| Parameter | Values | Notes |
|---|---|---|
| Source | snapshot of an existing table / generate | generation uses a private DB (the shared DB is not touched) |
| `shtyp` | DEF, NRT | definitive or near-real-time SHARP data |
| `class0` | Cplus, Mplus, Xplus | thresholds 1e-6, 1e-5, 1e-4 W/m² |
| `delta_hours` | 1–240 (default 24) | width of the before/after/after-after windows and grid step |

Results: rows, ARs, fraction of YES, `label`/`label_u` distribution, positives per year, `labels.csv`.

## Dataset
**1 · Extraction** (`create_hdf5_from_db`): `data_version` (clean = 17 features, all = 22),
`window_hours` (history before t_X), `buffer_minutes`, `sampling_interval` (minutes),
random subsample (`test_mode`/`test_size`, seed). Results: samples, features, time range,
ARs, classes, % NaN per feature, events excluded for missing data.

**2 · Filters and split** (`create_dataset_splits`):
- Features: `FEATURE_SELECTION_MODE` direct (multiselect) or criterion (`FeatureSelector` groups; the GUI
  flags groups not compatible with the dataset).
- Missing data: `MISSINGNESS_FILTER_MODE` (multi_window_empty, multi_window, single_window, none), `MAX_NAN_RATIO`,
  `TIME_WINDOWS`, `TIMESTEPS`.
- Quality: `CMASK_FILTER`, `CMASK_THRESHOLD`, `CMASK_STRATEGY` (any, majority, average, minimum),
  `CMASK_MAJORITY_THRESHOLD`.
- Time: `TEMPORAL_FILTER_MODE` (select, filter, exclude) + `TEMPORAL_WINDOW`; `PROBLEMATIC_PERIODS`.
- Split: `SPLITTING_STRATEGY` random (StratifiedGroupKFold by HARP) or chronological (`TEST_CUTOFF_DATE`,
  `TEST_TIME_WINDOW` or last `TEST_SIZE` fraction), `N_FOLDS`, `RANDOM_STATE`, `FORCE_SEPARATION`, `SAVE_PLOTS`.
- Class balance: the package does not resample; it is handled by the loss (PyTorch) or `scale_pos_weight` (XGBoost).

Results: samples per set, positives, ARs, time ranges, filter exclusions, test ↔ train/val
separation check, package plots.

## Model configuration (`MODEL_CONFIGS`, `REGULARIZATION_PARAMS`)
Menu of the trainable models; form generated from the default parameters of the chosen model (types enforced,
dataset-dependent parameters shown as "auto"). PyTorch training: batch size, learning rate,
weight decay, epochs, patience, scheduler, loss (+ alpha/gamma), early stopping, calibration. All models:
threshold optimization (metric TSS/F1/MCC/precision/recall/precision_at_recall/f1_at_precision), seed.
SK models: regularization grid. Duplication of an existing configuration, diff, JSON export.

## Training and validation (`train_models`)
Choice of split and configuration; table of the split time ranges and anti-leakage check;
windows (3h=16, 6h=31, 12h=61, 24h=121 steps, limited to the `TIMESTEPS`), number of folds, device.
Live monitor (status, time, log, cancellation). Results: metrics per fold/C, selected fold,
CV variability boxplot, package plots, provenance.

## Inference
Trained model; data: test set of the model split, another split, extracted dataset; windows; fold
selected by the package or a specific fold; CV or fixed threshold; HARP and time range filters; exclusion of
samples of ARs seen in training; bootstrap (per sample or per AR). Feature compatibility check.

## Active Regions
Choice of the HARP (optionally only those with predictions), SHARP features, GOES flares (log scale with
C/M/X levels), labels (artifact or DB table), predictions and thresholds of several experiments, membership of
the samples in the train/val/test of a split. Synchronized time axes, zoom, pan, tooltips.

## Global metrics
Multiselect of inferences; filters by window, event type, time range, HARP, threshold (saved or
custom), inclusion of samples seen in training. Table with TSS, HSS, Precision, Recall,
Specificity, F1, Accuracy, ROC-AUC, PR-AUC, FPR, FNR, confusion matrix, threshold criterion;
ROC/PR curves, TSS vs threshold, bootstrap CI; warning when rows are not comparable; CSV/JSON export.

## Algorithm comparison
Comparison table (algorithm, changed hyperparameters, features, split, ranges, source, threshold,
metrics with CI, times). Validity check per window: a model is declared "highest TSS"
only if the evaluated samples are the same; overlapping confidence intervals are flagged.
TSS chart with error bars, overlaid ROC/PR curves.

## Experiments and pipeline
Filterable catalogue (type, status, text) with sizes and dependencies, dependency graph, details
(config, results, provenance, files with sha256, history), log, integrity check of the chain, warnings on
changed commit or local changes, reproduction of the step or the chain, resume from the last valid artifact,
JSON variants, chain export, comparison between versions, protected deletion, job list and logs.
