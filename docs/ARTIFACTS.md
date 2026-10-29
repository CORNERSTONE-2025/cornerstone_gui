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
| `labels` | `lbl-` | — | `labels.sqlite` (table `DF_…_three_classes`), `labels.csv` |
| `dataset` | `ds-` | `labels` | `sharp_dataset_db_<table>_<class>_w<h>h_s<m>min.h5`, `missing_data_log.csv` |
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
| `label`, `label_u` | package label (YES/NO1..NO4) and flare class |
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
