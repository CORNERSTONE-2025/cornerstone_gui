# CORNERSTONE GUI

Interfaccia grafica per il pacchetto [CORNERSTONE DB package](https://github.com/CORNERSTONE-2025/cornerstone_db_package)
(previsione dei flare solari da parametri SHARP/HMI delle Active Region). Permette di eseguire e gestire
l'intera pipeline senza scrivere codice:

**DB SQLite → labeling → dataset HDF5 → filtri e split → configurazione modello → training CV → inferenza
su eventi non visti → metriche, confronto, storia delle Active Region**

Ogni passo usa le funzioni reali del pacchetto e produce un **artefatto persistente e versionato**
(configurazione, file, log, checksum, versioni del codice), così gli esperimenti si possono riprendere,
riprodurre e confrontare.

> La GUI non modifica il codice del pacchetto né il database condiviso (aperto in sola lettura).

## Requisiti

- Linux, Python ≥ 3.10, [uv](https://docs.astral.sh/uv/).
- Un checkout del pacchetto CORNERSTONE con il suo virtualenv funzionante
  (`<cornerstone>/.venv`, con torch, xgboost, h5py, clearml…) e il database SQLite locale.
- GPU CUDA solo per ROCKET/MINIROCKET (cupy). Gli altri modelli girano anche su CPU.

## Installazione

```bash
git clone <questo repository> cornerstone_gui
cd cornerstone_gui
uv sync                       # crea .venv con streamlit, plotly, pandas, h5py, scikit-learn, pytest
cp config/gui_settings.example.json gui_settings.json   # opzionale: modifica i percorsi
```

Impostazioni (file `gui_settings.json`, variabili `CS_GUI_*` o pagina Home):

| chiave | default | significato |
|---|---|---|
| `cornerstone_root` | `/bigdisk/daniele/cornerstone_module` | checkout del pacchetto |
| `cornerstone_python` | `<root>/.venv/bin/python` | interprete usato per eseguire i passi |
| `db_path` | `local_db_path` di `cornerstone/configs/db_config.json` | DB SQLite reale |
| `workspace` | `./workspace` | artefatti della modalità reale |
| `mode` | `real` | `real` oppure `demo` (DB sintetico in `./workspace_demo`) |
| `min_free_disk_gb` | 5 | spazio minimo per avviare un job |

## Avvio

```bash
uv run streamlit run app/streamlit_app.py
# poi apri http://localhost:8501  (su server remoto: --server.address 0.0.0.0 oppure un tunnel ssh)
```

## Flusso completo dalla GUI

1. **Labeling** — snapshot di una tabella di label già presente nel DB (veloce) o generazione con
   `splitting_labeling` su un DB privato.
2. **Dataset → Estrazione HDF5** — finestra storica, cadenza, versione delle feature (clean/all),
   eventuale sottocampione per prove rapide.
3. **Dataset → Filtri e split** — feature, dati mancanti, filtri temporali e di qualità, split casuale o
   cronologico per Active Region, numero di fold.
4. **Configurazione modelli** — algoritmo, iperparametri, loss, early stopping, soglia, seed.
5. **Training e validazione** — finestre temporali, fold, dispositivo; il job gira in background con log live.
6. **Inferenza** — sul test set dello split (o altri dati non visti), soglia CV o fissa, CI bootstrap.
7. **Metriche globali**, **Confronto algoritmi**, **Active Regions** per l'analisi.
8. **Esperimenti e pipeline** per catalogo, dipendenze, integrità, varianti, riproduzione, export.

Ogni form mostra il riepilogo della configurazione e gli errori di validazione **prima** dell'esecuzione.

## Pipeline completa senza GUI

```bash
uv run python examples/run_pipeline.py                  # modalità reale, sottocampione di 600 eventi
uv run python examples/run_pipeline.py --mode demo      # DB sintetico (solo prova)
uv run python examples/run_pipeline.py --model model_lstm.json
```

Le configurazioni sono in `examples/configs/` (label, dataset, split random e cronologico, XGBoost, LSTM,
training, inferenza). Gli artefatti creati compaiono subito nella GUI.

## Salvare e riprodurre un esperimento

- Ogni esecuzione è salvata automaticamente in `workspace/artifacts/<id>/` (vedi [docs/ARTIFACTS.md](docs/ARTIFACTS.md)).
- **Variante**: pagina *Esperimenti* → *Azioni* → modifica il JSON → *Crea variante* (l'originale resta intatto).
  Per i modelli: *Configurazione modelli* → "Parti da una configurazione esistente".
- **Riproduci questo passo**: stesso input e stessa configurazione, nuovo artefatto.
- **Riproduci l'intera catena**: nuove copie di tutti i passi a monte, collegate tra loro.
- **Riprendi**: esegue solo i passi non ancora completati della catena (ripartenza dopo errori o interruzioni).
- **Esporta**: JSON con manifest, configurazioni e provenienza di tutta la catena.
- **Verifica integrità**: sha256 dei file e stato delle dipendenze.

## Modalità demo

Pagina Home → modalità *Demo* → *Genera DB demo*. Il database è **sintetico** (stesso schema del DB reale,
valori generati) e serve solo per provare la GUI e per i test: ogni pagina mostra un banner e gli artefatti demo
stanno in un workspace separato. I risultati demo non hanno alcun valore scientifico.

## Test

```bash
uv run pytest -m "not integration"     # unitari: archivio, validazione, metriche, job, adattatori (~3 s)
uv run pytest -m integration           # pacchetto reale sul DB demo + tutte le pagine (AppTest) (~5 min)
```

## Documentazione

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): analisi del pacchetto, architettura, adattatori, data leakage, limitazioni.
- [docs/ARTIFACTS.md](docs/ARTIFACTS.md): formato di esperimenti, artefatti, predizioni.
- [docs/PAGES.md](docs/PAGES.md): pagine e parametri.

## Limitazioni principali

- Nessun dato CME nel database: sezioni CME non disponibili.
- ROCKET/MINIROCKET richiedono GPU.
- Riproducibilità bit-a-bit non garantita per i modelli PyTorch; commit, versioni e seed sono registrati.
- Il labeling generato copre l'intervallo fisso del pacchetto (1996–2025) ed è lento.
- `db_update()` (download dei dump da Google Drive) non è esposto nella GUI.
