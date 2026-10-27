# Analisi del pacchetto e architettura della GUI

## 1. Analisi del repository CORNERSTONE (`cornerstone_db_package`)

Analisi fatta sul commit `897a662` (branch `main` di CORNERSTONE-2025). La GUI non modifica il pacchetto.

### API pubblica (`cornerstone.Cornerstone`)

| Metodo | Cosa fa | Input → output |
|---|---|---|
| `db_update()` | Sincronizza il DB locale con dump remoti su Google Drive | rete → SQLite locale |
| `splitting_labeling(shtyp, class0, delta_hours, db_type, save_csv)` | Label per (t_X, HARP) contando i flare GOES prima/dopo t_X | tabelle `SHARPC_all_*`, `NOAA_SHARPC_*_AR` → tabella `DF_<shtyp>_<class0>_delta<h>_three_classes` (**append** nel DB) |
| `create_hdf5_from_db(label_type, data_version, table_type, window_hours, buffer_minutes, sampling_interval, test_mode, test_size, random_seed, db_type)` | Serie temporali SHARP nelle `window_hours` precedenti ogni t_X | label + `SHARPC_<version>_<type>` → HDF5 + `missing_data_log.csv` |
| `create_dataset_splits(config)` | Selezione feature, filtri sui mancanti/temporali/CMASK, split random o cronologico con k-fold raggruppato per HARP | HDF5 → `splits/*_cv_k_{train,val}.h5`, `*_test.h5`, grafici, config |
| `train_models(train_config_path)` | Cross-validation per ogni finestra (3h/6h/12h/24h) | split + JSON → modelli per fold, metriche JSON, grafici |
| `test_model(test_config_path)` | Valutazione del modello `_best` sul test set | modelli → metriche (solo log ClearML) |

### Dati (SQLite)

- `SHARPC_clean_{DEF,NRT}` (17 feature normalizzate `_N1/_N2`) e `SHARPC_all_{DEF,NRT}` (22 feature): cadenza 12 min.
- `NOAA_SHARPC_{DEF,NRT}_AR`: flare GOES XRA associati alle HARP (`fflux` in W/m², `part1` = classe).
- `DF_*_three_classes`: label `YES`/`NO1..NO4`, `label_u` = classe del flare (C/M/X).
- **CME: nessuna tabella.** Il pacchetto tratta solo flare; la GUI lo dichiara e non simula dati CME.

### Modelli (`cornerstone.ml`)

- PyTorch (`PT_MODELS`): LSTM, TRANSFORMER, INCEPTIONTIME, ONEFITSALL e i modelli della Time-Series-Library
  (Autoformer, DLinear, FEDformer, LightTS, Nonstationary_Transformer, Pyraformer, Reformer, TimesNet, Informer).
- sklearn-like (`SK_MODELS`): XGBOOST, ROCKET, MINIROCKET (questi ultimi due usano **cupy → GPU obbligatoria**).
- Iperparametri: `cornerstone/ml/model_configs.py`; griglia di regolarizzazione SK: `regularization_configs.py`.
- Training PT: loss bce/focal/asymmetric_focal/curriculum, scheduler Cosine/Plateau, early stopping su
  tss/val_loss/auc/f1/mcc/hss, ottimizzazione della soglia su validation, calibrazione Platt/temperature opzionale.
- Training SK: griglia C × fold, C scelto per TSS medio di validation, soglia ottimizzata su validation (`model.threshold_`).
- Metriche (`metrics.compute_metrics`): F1, AUC, MCC, TSS (= TPR − FPR), HSS, matrice di confusione.

### Lacune individuate e come la GUI le gestisce

| Lacuna | Gestione |
|---|---|
| `splitting_labeling` scrive in **append** nel DB condiviso (ri-eseguire duplica le righe) e un CSV nella cwd | Labeling su un **DB privato dell'artefatto** (adattatore, §3) |
| `create_hdf5_from_db` legge label e feature dallo stesso DB | DB principale aperto **read-only** + label dell'artefatto montate come vista temporanea |
| Gli override `MODEL_CONFIG_OVERRIDES` del JSON vengono sovrascritti da `MODEL_CONFIGS` in `ModelManager.create_model` | Il runner aggiorna `MODEL_CONFIGS` nel proprio processo |
| `train_models` richiede che il dataset sia registrato in `AVAILABLE_DATASETS` | Il runner registra lo split dell'artefatto |
| Nessun seed impostato nel training | Il runner imposta seed python/numpy/torch (registrato nella config) |
| Per i modelli PyTorch non viene salvato `*_best.pt` richiesto da `run_test` | Copia del fold `best_fold` scelto dal pacchetto (registrata nel manifest) |
| `run_test`: soglia fissa 0.5 per PT, valore di ritorno corrotto (riassegna `test_metrics` nel ciclo), nessuna predizione per campione | Inferenza della GUI costruita con i mattoni del pacchetto (`FlareHDF5Dataset`, `ModelManager`, `collect_logits`, `compute_metrics`), soglia esplicita, predizioni salvate |
| Chiave `calibration.kind` in `train_config.json` ignorata (il codice legge `type`) | La GUI scrive `type` e lo segnala |
| Range di date del labeling fisso (1996–2025) | Documentato; non modificabile senza cambiare il pacchetto |
| CME assenti | Dichiarato in GUI; nessuna simulazione |

## 2. Architettura

```
┌────────────── Streamlit (venv GUI: streamlit, plotly, pandas, h5py, sklearn) ──────────────┐
│ app/pages/*.py  ── cs_gui.ui (componenti) ── cs_gui.pipeline (validazione, varianti,       │
│                                               riproduzione) ── cs_gui.store (artefatti)     │
│                                               cs_gui.jobs (processi in background)          │
│                                               cs_gui.db (SQLite read-only), metrics, h5info │
└──────────────────────────────────┬─────────────────────────────────────────────────────────┘
                                   │ subprocess (process group dedicato, log su file)
┌──────────────────────────────────▼──────── venv CORNERSTONE (torch, xgboost, cupy, …) ─────┐
│ python -m cs_gui.runner  ── steps.py (un passo per tipo di artefatto)                       │
│                          ── adapters.py (integrazione senza modificare il pacchetto)        │
│                          ── cornerstone.Cornerstone / cornerstone.ml.*                      │
└────────────────────────────────────────────────────────────────────────────────────────────┘
```

Scelte:
- **Streamlit**: Python puro, compatibile con lo stack scientifico, form dinamici e grafici Plotly
  interattivi; le operazioni lunghe non bloccano l'interfaccia perché girano in sottoprocessi.
- **Due virtualenv**: la GUI resta leggera; il runner usa l'ambiente esatto del pacchetto (stesse versioni
  di torch/cupy/xgboost usate in ricerca).
- **Stato su disco** (manifest JSON per artefatto + record dei job): sopravvive al riavvio della GUI; un job
  morto senza aggiornare lo stato viene marcato `failed` al successivo accesso (`JobManager.reconcile`).
- **ClearML offline**: il runner forza `CLEARML_OFFLINE_MODE=1` per non pubblicare esecuzioni dalla GUI.

## 3. Adattatori (`cs_gui/runner/adapters.py`)

1. `readonly_sqlite_module`: sostituisce `cornerstone.cornerstone.sqlite3` nel processo runner. Il DB principale
   si apre con `mode=ro` (scritture impossibili); il DB delle label dell'artefatto viene `ATTACH`-ato e una
   `TEMP VIEW` col nome atteso dal pacchetto ha la precedenza sulla tabella omonima del DB principale.
2. `build_labeling_db`: DB privato con `NOAA_SHARPC_<t>_AR` copiata e, per `SHARPC_all_<t>`, solo il primo e
   l'ultimo `T_OBS` di ogni HARP. La query usata da `splitting_labeling`
   (`SELECT HARPNUM, MIN(T_OBS), MAX(T_OBS) … GROUP BY HARPNUM`) dà lo stesso risultato (test automatico).
3. `apply_model_overrides`, `register_dataset`, `set_device`, `set_seeds`: configurazione in-process.
4. `make_best_pt_models`: copia `*_fold{best_fold}.pt` → `*_best.pt`.

## 4. Data leakage: regole applicate

- Split: si usano solo le strategie del pacchetto, entrambe raggruppate per HARP; la GUI verifica dopo lo split
  che test e train/val non condividano HARP né campioni (`summary.leakage`) e **blocca il training** se succede.
- Il preprocessing del pacchetto è per campione (selezione/filtri, nessuna normalizzazione appresa sui dati),
  quindi non c'è fit di trasformazioni su dati di test.
- Inferenza: ogni predizione è marcata se la sua HARP compare in train/val del modello; per default queste
  righe sono **escluse dalle metriche**. Usare train/val dello stesso split come "eventi non visti" è vietato
  dalla validazione.
- Split casuale: test e training non sono separati nel tempo. La GUI lo segnala e suggerisce lo split cronologico
  (cutoff o finestra) per una valutazione operativa.

## 5. Limitazioni residue

- ROCKET/MINIROCKET richiedono GPU CUDA (cupy): su macchine senza GPU la GUI li blocca prima del training.
- Riproducibilità bit-a-bit non garantita: operazioni non deterministiche di PyTorch, thread multipli,
  versioni diverse delle dipendenze. La GUI registra commit del pacchetto (con flag "modifiche locali"),
  versioni e seed, e avvisa se il pacchetto è cambiato rispetto all'esecuzione originale.
- Il labeling generato copre sempre 1996-01-01 → 2025-04-30 (fissato nel pacchetto) e può richiedere
  decine di minuti.
- Le metriche "test_*" mostrate nel training sono quelle che il pacchetto calcola per fold; la valutazione di
  riferimento è quella dell'artefatto di inferenza.
- Le predizioni dei modelli PyTorch usano probabilità sigmoid **non calibrate** (come `run_test`); la soglia CV
  usata è la `best_threshold` del fold, ottimizzata sulle stesse probabilità non calibrate.
- `db_update()` (sincronizzazione da Google Drive) non è esposto: modifica il DB condiviso e richiede rete.
