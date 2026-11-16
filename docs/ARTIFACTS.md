# Experiment and artifact format

## Workspace

```
workspace/                    # real mode (workspace_demo/ for demo mode)
├── package_info.json         # package introspection (models, parameters, feature groups, CUDA)
├── jobs/<job_id>.json|.log   # launched processes: pid, artifacts, command, runner output
└── artifacts/<artifact_id>/
    ├── manifest.json
    ├── logs/run.log           # stdout/stderr of the step (package output included)
    └── … produced files
```

## Artifact types

| Type | Id prefix | Parents | Main files |
|---|---|---|---|
| `labels` | `lbl-` | — | `labels.parquet` (t_X, HARPNUM, label, strat_class, …), `flares.parquet` (flare → HARP association), `labels_meta.json` (all parameters, data provenance), `labels.csv` (up to 300k rows) |
| `dataset` | `ds-` | `labels` | `dataset.h5` (`X`, `Y`, `label_<name>`, `label_u_<name>` = stratification class, `HARPNUM`, `timestamps`, `QUALITY`, attribute `metadata`), `dataset.missing.csv` |
| `split` | `spl-` | `dataset` | `split_config.json`, `split/splits/split_{test,cv_k_train,cv_k_val}.h5`, `split/plots/*.png`, `split/config/` |
| `model_config` | `mdl-` | — | manifest only (configuration) |
| `training` | `trn-` | `split`, `model_config` | `train_config.json`, `results/split/<task>/{models,metrics,plots,calibration,configs}/` |
| `inference` | `inf-` | `training`, `source` (split or dataset) | `predictions.csv`, `metrics.json` |

## `manifest.json`

```json
{
  "id": "trn-20261027-123435-df54",
  "type": "training",
  "name": "XGBOOST example",
  "description": "…",
  "created_at": "2026-10-27T12:34:35+00:00",
  "updated_at": "…",
  "status": "completed",              // draft | pending | running | completed | failed | cancelled
  "mode": "real",                     // real | demo
  "parents": {"split": "spl-…", "model_config": "mdl-…"},
  "derived_from": null,               // id of the artifact this is a variant / reproduction of
  "config": {…},                      // step parameters (frozen once started)
  "outputs": {"files": {"results/…/model.pkl": {"size": 1234, "sha256": "…"}}},
  "summary": {…},                     // numbers shown in the GUI (depend on the type)
  "provenance": {
    "python": "3.10.18", "packages": {"torch": "2.9.1", …},
    "cornerstone": {"commit": "…", "dirty": false, "branch": "main", "remote": "…"},
    "cornerstone_gui": {"commit": "…", "dirty": false}, "host": "…"
  },
  "run": {"job_id": "…", "pid": 123, "started_at": "…", "finished_at": "…", "duration_s": 20.0,
          "error": null, "traceback": null, "exit_code": 0},
  "history": [{"at": "…", "event": "status", "status": "running"}, …]
}
```

## `predictions.csv` (inference)

| column | meaning |
|---|---|
| `window` | time window of the model (3h, 6h, 12h, 24h) |
| `HARPNUM`, `t_issue` | Active Region and issue time (last non-empty timestamp of the series = t_X) |
| `label`, `label_u` | prediction label (YES/NO; NO1..NO4 in datasets of earlier versions) and stratification class (C/M/X… or NO1..NO4) |
| `y_true` | 1 if `label == YES` |
| `score` | predicted probability (sigmoid for PyTorch, `predict_proba` for sklearn) |
| `threshold`, `y_pred` | threshold used and decision |
| `in_trainval_sample` / `in_trainval_harp` | the sample / its HARP appear in the model train/val |
| `model_file` | model file used |

`metrics.json` contains per window: threshold and criterion, counts, `package_metrics`
(`cornerstone.ml.utils.metrics.compute_metrics`), `metrics` (extended set, `cs_gui.metrics`) and `ci`
(bootstrap with the method stated).

## Immutability rules

- `config` and `parents` can only change in the `draft`/`pending` states.
- A `completed` artifact never goes back to another state: to run it again, create a **variant** or a
  **reproduction** (new id, `derived_from` → original).
- Deletion requires explicit confirmation and is refused if dependents exist, unless a cascade is confirmed.
- Integrity is checked by comparing file size and sha256 with the manifest and the status of the parents.

## Moving artifacts between installations (bundles)

*Experiments and pipeline* → *Share (export / import)*, or from the command line:

```bash
python -m cs_gui.bundle export --workspace workspace --out chain.zip trn-20261112-101500-ab12
python -m cs_gui.bundle import --workspace workspace chain.zip
```

A bundle is a zip with `bundle.json` (format, version, list of artifacts with dependencies first,
note) and `artifacts/<id>/…`, the complete folder of every artifact. Rules:

- export takes **completed** artifacts and adds **all their ancestors**, so the bundle is self-contained;
- ids are kept: the imported artifacts keep their dependencies and appear in every page with their results;
  they can be the input of new steps (inference, new splits from an imported dataset, …);
- import checks every file against the sha256 of its manifest before anything enters the workspace, and
  never overwrites: an artifact already present with the same outputs is skipped, the same id with
  different outputs refuses the whole import; entries outside `artifacts/<listed id>/` are refused;
- demo and real artifacts never mix; the manifest records `imported` (date, bundle) and a history event;
- paths inside the artifacts are relative to the artifact folder; the only absolute paths are in the
  configuration files written for the package (`split_config.json`, `train_config.json`), which are a
  record of the run: inference writes a copy of the training configuration pointing to the current folder
  (`train_config_used.json`).

The bundle carries results, not the data store: labeling and dataset extraction on the receiving side
need their own data (Data page), while inference on an imported split or dataset does not use it.
