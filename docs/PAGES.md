# Pagine e parametri

Tutte le opzioni elencate corrispondono a funzionalità del pacchetto (fonte indicata tra parentesi).

## Home e stato
Schema del flusso, conteggio degli artefatti per tipo e stato, job attivi, stato dell'ambiente (commit del
pacchetto, CUDA, DB, spazio disco), configurazione dei percorsi e della modalità, generazione del DB demo.

## Labeling (`Cornerstone.splitting_labeling`)
| Parametro | Valori | Note |
|---|---|---|
| Sorgente | snapshot di una tabella esistente / genera | la generazione usa un DB privato (il DB condiviso non viene toccato) |
| `shtyp` | DEF, NRT | dati SHARP definitivi o near-real-time |
| `class0` | Cplus, Mplus, Xplus | soglie 1e-6, 1e-5, 1e-4 W/m² |
| `delta_hours` | 1–240 (default 24) | ampiezza delle finestre prima/dopo/dopo-dopo e passo della griglia |

Risultati: righe, AR, frazione di YES, distribuzione `label`/`label_u`, positivi per anno, `labels.csv`.

## Dataset
**1 · Estrazione** (`create_hdf5_from_db`): `data_version` (clean = 17 feature, all = 22),
`window_hours` (storia prima di t_X), `buffer_minutes`, `sampling_interval` (minuti),
sottocampione casuale (`test_mode`/`test_size`, seed). Risultati: campioni, feature, intervallo temporale,
AR, classi, % NaN per feature, eventi esclusi per dati assenti.

**2 · Filtri e split** (`create_dataset_splits`):
- Feature: `FEATURE_SELECTION_MODE` direct (multiselezione) o criterion (gruppi di `FeatureSelector`; la GUI
  segnala i gruppi non compatibili con il dataset).
- Mancanti: `MISSINGNESS_FILTER_MODE` (multi_window_empty, multi_window, single_window, none), `MAX_NAN_RATIO`,
  `TIME_WINDOWS`, `TIMESTEPS`.
- Qualità: `CMASK_FILTER`, `CMASK_THRESHOLD`, `CMASK_STRATEGY` (any, majority, average, minimum),
  `CMASK_MAJORITY_THRESHOLD`.
- Tempo: `TEMPORAL_FILTER_MODE` (select, filter, exclude) + `TEMPORAL_WINDOW`; `PROBLEMATIC_PERIODS`.
- Split: `SPLITTING_STRATEGY` random (StratifiedGroupKFold per HARP) o chronological (`TEST_CUTOFF_DATE`,
  `TEST_TIME_WINDOW` o ultima frazione `TEST_SIZE`), `N_FOLDS`, `RANDOM_STATE`, `FORCE_SEPARATION`, `SAVE_PLOTS`.
- Bilanciamento: il pacchetto non ricampiona; si gestisce con la loss (PyTorch) o `scale_pos_weight` (XGBoost).

Risultati: campioni per insieme, positivi, AR, intervalli temporali, esclusioni dei filtri, controllo di
separazione test ↔ train/val, grafici del pacchetto.

## Configurazione modelli (`MODEL_CONFIGS`, `REGULARIZATION_PARAMS`)
Menu con i modelli addestrabili; form generato dai parametri di default del modello scelto (tipi rispettati,
parametri che dipendono dal dataset mostrati come "auto"). Training PyTorch: batch size, learning rate,
weight decay, epoche, patience, scheduler, loss (+ alpha/gamma), early stopping, calibrazione. Tutti i modelli:
ottimizzazione della soglia (metrica TSS/F1/MCC/precision/recall/precision_at_recall/f1_at_precision), seed.
Modelli SK: griglia di regolarizzazione. Duplicazione di una configurazione esistente, diff, esportazione JSON.

## Training e validazione (`train_models`)
Selezione di split e configurazione; tabella degli intervalli temporali e del controllo anti-leakage dello
split; finestre (3h=16, 6h=31, 12h=61, 24h=121 step, limitate ai `TIMESTEPS`), numero di fold, dispositivo.
Monitor in tempo reale (stato, tempo, log, annullamento). Risultati: metriche per fold/C, fold selezionato,
boxplot della variabilità CV, grafici del pacchetto, provenienza.

## Inferenza
Modello addestrato; dati: test dello split del modello, altro split, dataset estratto; finestre; fold
selezionato dal pacchetto o fold specifico; soglia CV o fissa; filtri HARP e intervallo; esclusione dei campioni
di AR viste in training; bootstrap (per campione o per AR). Verifica di compatibilità delle feature.

## Active Regions
Selezione della HARP (anche solo quelle con predizioni), feature SHARP, flare GOES (scala log con livelli
C/M/X), label (artefatto o tabella DB), predizioni e soglie di più esperimenti, appartenenza dei campioni a
train/val/test di uno split. Assi temporali sincronizzati, zoom, pan, tooltip.

## Metriche globali
Multiselezione di inferenze; filtri per finestra, tipologia di evento, intervallo, HARP, soglia (salvata o
personalizzata), inclusione dei campioni visti in training. Tabella con TSS, HSS, Precision, Recall,
Specificity, F1, Accuracy, ROC-AUC, PR-AUC, FPR, FNR, matrice di confusione, criterio della soglia;
curve ROC/PR, TSS vs soglia, CI bootstrap; avviso quando le righe non sono confrontabili; export CSV/JSON.

## Confronto algoritmi
Tabella comparativa (algoritmo, iperparametri modificati, feature, split, intervalli, sorgente, soglia,
metriche con CI, tempi). Controllo di validità per finestra: un modello viene indicato come "TSS più alto"
solo se i campioni valutati coincidono; viene segnalata la sovrapposizione degli intervalli di confidenza.
Grafico TSS con barre d'errore, curve ROC/PR sovrapposte.

## Esperimenti e pipeline
Catalogo filtrabile (tipo, stato, testo) con dimensioni e dipendenze, grafo delle dipendenze, dettaglio
(config, risultati, provenienza, file con sha256, storia), log, verifica di integrità della catena, avvisi su
commit cambiato o modifiche locali, riproduzione del passo o della catena, ripresa dall'ultimo artefatto valido,
varianti da JSON, export della catena, confronto tra versioni, cancellazione protetta, elenco e log dei job.
