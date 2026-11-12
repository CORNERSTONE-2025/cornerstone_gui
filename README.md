# CORNERSTONE GUI

Graphical interface for the [CORNERSTONE DB package](https://github.com/CORNERSTONE-2025/cornerstone_db_package)
(solar flare forecasting from the SHARP/HMI parameters of Active Regions). It runs and manages the
whole pipeline without writing code:

**Data (SHARP + GOES) → labeling → HDF5 dataset → filters and split → model configuration → CV training → inference
on unseen events → metrics, comparison, Active Region history**

Every step uses the real functions of the package and produces a **persistent, versioned artifact**
(configuration, files, log, checksums, code versions), so experiments can be resumed, reproduced and compared.

> The GUI never modifies the package code, and only reads the data store. The only exception is the
> **Data** page, which downloads and updates the store through the package (`python -m cornerstone.data`).

## Quick start with Docker (recommended)

```bash
git clone https://github.com/CORNERSTONE-2025/cornerstone_gui.git
cd cornerstone_gui
docker compose up --build -d          # then open http://localhost:8501
```

Without data the GUI starts in demo mode on synthetic data. To use real data, open the **Data** page and
download it (from the sources, from the published snapshot, or by importing a `cornerstone.sqlite` of the
former dump), then switch to Real mode in *Home → GUI configuration*. All details: [docs/DOCKER.md](docs/DOCKER.md).

## Local installation (without Docker)

Requirements:

- Linux, Python ≥ 3.10, [uv](https://docs.astral.sh/uv/).
- A checkout of the CORNERSTONE package with a working environment (its `.venv` with torch,
  xgboost, h5py, clearml…; [docker/requirements-cornerstone.txt](docker/requirements-cornerstone.txt) lists
  what it needs). The data are downloaded from the Data page.
- No GPU is required: every model also runs on CPU. A GPU (NVIDIA CUDA, or AMD with a ROCm build of
  torch) speeds up the PyTorch models.

```bash
git clone https://github.com/CORNERSTONE-2025/cornerstone_gui.git
cd cornerstone_gui
uv sync                       # creates .venv with streamlit, plotly, pandas, h5py, scikit-learn, pytest
cp config/gui_settings.example.json gui_settings.json   # optional: edit the paths
uv run streamlit run app/streamlit_app.py               # then open http://localhost:8501
```

Settings (file `gui_settings.json`, `CS_GUI_*` environment variables, or the Home page):

| key | default | meaning |
|---|---|---|
| `cornerstone_root` | the importable `cornerstone` package, else `../cornerstone_db_package` | package checkout |
| `cornerstone_python` | `<root>/.venv/bin/python`, else the GUI interpreter | interpreter used to run the steps |
| `data_root` | `data_root` of `cornerstone/configs/data_config.json` (or `CORNERSTONE_DATA_ROOT`) | Parquet data store |
| `legacy_db_path` | — | `cornerstone.sqlite` of the former dump, offered for import in the Data page |
| `workspace` | `./workspace` | artifacts of real mode |
| `mode` | `real` | `real` or `demo` (synthetic data in `./workspace_demo`) |
| `min_free_disk_gb` | 5 | minimum free space to start a job |

Environment variables: `CS_GUI_CORNERSTONE_ROOT`, `CS_GUI_CORNERSTONE_PYTHON`, `CS_GUI_DATA_ROOT`, `CS_GUI_LEGACY_DB`,
`CS_GUI_WORKSPACE`, `CS_GUI_DEMO_WORKSPACE`, `CS_GUI_MODE`, `CS_GUI_SETTINGS_FILE`. They take precedence
over the settings file.

On a remote server, reach the GUI through an SSH tunnel (`ssh -L 8501:localhost:8501 user@server`)
rather than exposing the port: the GUI has no login.

## Full workflow in the GUI

0. **Data** — download and update SHARP DEF/NRT (JSOC), GOES flares (NOAA SWPC) and the HARP ↔ NOAA map.
1. **Labeling** — parametric (`cornerstone.labeling`): SHARP series, prediction threshold, flare window,
   grid step, stratification threshold and window, time range, record quality filter, GOES calibration
   and flare → HARP association rules. The former static tables are one preset.
2. **Dataset → HDF5 extraction** — history window, cadence, feature set (clean/all), records used,
   optional subsample for quick tests. The split stratifies on the stratification class.
3. **Dataset → Filters and split** — features, missing data, time and quality filters, random or
   chronological split by Active Region, number of folds.
4. **Model configuration** — algorithm, hyperparameters, loss, early stopping, threshold, seed.
5. **Training and validation** — time windows, folds, device; the job runs in the background with a live log.
6. **Inference** — on the split test set (or other unseen data), CV or fixed threshold, bootstrap CI.
7. **Global metrics**, **Algorithm comparison**, **Active Regions** for the analysis.
8. **Experiments and pipeline** for catalogue, dependencies, integrity, variants, reproduction, export.

Every form shows the configuration summary and the validation errors **before** running.

## Full pipeline without the GUI

```bash
uv run python examples/run_pipeline.py                  # real mode, subsample of 600 events
uv run python examples/run_pipeline.py --mode demo      # synthetic data (test only)
uv run python examples/run_pipeline.py --model model_lstm.json
```

The configurations are in `examples/configs/` (labels, dataset, random and chronological split, XGBoost, LSTM,
training, inference). The artifacts created show up in the GUI right away.
With Docker: `docker compose exec gui python examples/run_pipeline.py --mode demo`.

## Saving and reproducing an experiment

- Every run is saved automatically in `workspace/artifacts/<id>/` (see [docs/ARTIFACTS.md](docs/ARTIFACTS.md)).
- **Variant**: *Experiments* page → *Actions* → edit the JSON → *Create variant* (the original stays intact).
  For models: *Model configuration* → "Start from an existing configuration".
- **Reproduce this step**: same input and configuration, new artifact.
- **Reproduce the whole chain**: new copies of all upstream steps, wired to each other.
- **Resume**: runs only the steps of the chain not completed yet (restart after errors or interruptions).
- **Export**: JSON with manifest, configurations and provenance of the whole chain.
- **Verify integrity**: sha256 of the files and status of the dependencies.

## Demo mode

Home page → *Demo* mode → *Generate demo data* (done automatically by the Docker image on a first start
without data). The data are **synthetic** (same layout as the real store, generated values) and are only meant
for trying out the GUI and for the tests: every page shows a banner and demo artifacts live in a separate
workspace. Demo results have no scientific value.

## Tests

```bash
uv run pytest -m "not integration"     # unit: store, validation, metrics, jobs, adapters (~3 s)
uv run pytest -m integration           # real package on the demo data, whole chain (~7 min)
```

## Documentation

- [docs/DOCKER.md](docs/DOCKER.md): Docker image, data, volumes, options.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): package analysis, architecture, adapters, data leakage, limitations.
- [docs/ARTIFACTS.md](docs/ARTIFACTS.md): format of experiments, artifacts, predictions.
- [docs/PAGES.md](docs/PAGES.md): pages and parameters.

## Main limitations

- No CME catalogue: CME sections are not available.
- The Docker image is CPU-only.
- Bit-for-bit reproducibility is not guaranteed for PyTorch models; commit, versions and seeds are recorded.
- The first download of SHARP from JSOC takes hours (then updates take minutes); the published snapshot
  relies on Google Drive (through `gdown`), which can refuse large automated downloads for a while.
