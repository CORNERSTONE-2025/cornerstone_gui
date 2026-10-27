# Formato di esperimenti e artefatti

## Workspace

```
workspace/                    # modalità reale (workspace_demo/ per la modalità demo)
├── package_info.json         # introspezione del pacchetto (modelli, parametri, gruppi di feature, CUDA)
├── jobs/<job_id>.json|.log   # processi lanciati: pid, artefatti, comando, output del runner
└── artifacts/<artifact_id>/
    ├── manifest.json
    ├── logs/run.log           # stdout/stderr del passo (output del pacchetto incluso)
    └── … file prodotti
```

## Tipi di artefatto

| Tipo | Prefisso id | Genitori | File principali |
|---|---|---|---|
| `labels` | `lbl-` | — | `labels.sqlite` (tabella `DF_…_three_classes`), `labels.csv` |
| `dataset` | `ds-` | `labels` | `sharp_dataset_db_<table>_<class>_w<h>h_s<m>min.h5`, `missing_data_log.csv` |
| `split` | `spl-` | `dataset` | `split_config.json`, `split/splits/split_{test,cv_k_train,cv_k_val}.h5`, `split/plots/*.png`, `split/config/` |
| `model_config` | `mdl-` | — | solo manifest (configurazione) |
| `training` | `trn-` | `split`, `model_config` | `train_config.json`, `results/split/<task>/{models,metrics,plots,calibration,configs}/` |
| `inference` | `inf-` | `training`, `source` (split o dataset) | `predictions.csv`, `metrics.json` |

## `manifest.json`

```json
{
  "id": "trn-20261027-123435-df54",
  "type": "training",
  "name": "Esempio XGBOOST",
  "description": "…",
  "created_at": "2026-10-27T12:34:35+00:00",
  "updated_at": "…",
  "status": "completed",              // draft | pending | running | completed | failed | cancelled
  "mode": "real",                     // real | demo
  "parents": {"split": "spl-…", "model_config": "mdl-…"},
  "derived_from": null,               // id dell'artefatto di cui è variante / riproduzione
  "config": {…},                      // parametri del passo (congelati dopo l'avvio)
  "outputs": {"files": {"results/…/model.pkl": {"size": 1234, "sha256": "…"}}},
  "summary": {…},                     // numeri mostrati in GUI (dipendono dal tipo)
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

## `predictions.csv` (inferenza)

| colonna | significato |
|---|---|
| `window` | finestra temporale del modello (3h, 6h, 12h, 24h) |
| `HARPNUM`, `t_issue` | Active Region e istante di emissione (ultimo timestamp non vuoto della serie = t_X) |
| `label`, `label_u` | label del pacchetto (YES/NO1..NO4) e classe del flare |
| `y_true` | 1 se `label == YES` |
| `score` | probabilità predetta (sigmoid per PyTorch, `predict_proba` per sklearn) |
| `threshold`, `y_pred` | soglia usata e decisione |
| `in_trainval_sample` / `in_trainval_harp` | il campione / la sua HARP compaiono in train/val del modello |
| `model_file` | file del modello usato |

`metrics.json` contiene per finestra: soglia e criterio, conteggi, `package_metrics`
(`cornerstone.ml.utils.metrics.compute_metrics`), `metrics` (set esteso, `cs_gui.metrics`) e `ci`
(bootstrap con metodo dichiarato).

## Regole di immutabilità

- `config` e `parents` si possono cambiare solo negli stati `draft`/`pending`.
- Un artefatto `completed` non torna mai in un altro stato: per rieseguire si crea una **variante** o una
  **riproduzione** (nuovo id, `derived_from` → originale).
- La cancellazione richiede conferma esplicita ed è rifiutata se esistono dipendenti, salvo cascata confermata.
- L'integrità si verifica confrontando dimensione e sha256 dei file con il manifest e lo stato dei genitori.
