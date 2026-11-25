# CORNERSTONE — report della campagna notturna 2026-11-25 (quick)

## 1. Esito in breve

- **Esito: parzialmente riuscito.** Su 25 test: ✅ 19 pass · ⚠️ 1 warn · ❌ 2 fail · ⏭️ 3 skip.
- La catena principale (labeling → dataset → split → XGBoost → inference) funziona su DEF e NRT, ed è riproducibile; tutte le suite (27 + 44 + 7 test) passano.
- **Problema più importante:** `P6.minirocket` fallisce perché nell'immagine Docker manca `numba` (MiniRocket non è utilizzabile dalla GUI così com'è).
- Il secondo fallimento (`P3.update`) è un timeout di rete verso la sorgente GOES; la mappa HARP-NOAA invece si aggiorna.
- Il data store NRT è vecchio di 13 giorni (warn).
- La campagna è terminata alle 17:05:29 (inizio 16:20:50, ~45 min), ben prima della deadline 2026-11-26T07:30; `campaign exit code 0`.

## 2. Tabella dei risultati

| | test | durata (min) | valori chiave |
|---|---|---|---|
| ✅ | P0.env | 0.02 | 12 CPU, 24.6 GB RAM, CUDA False, 55.7 GB liberi |
| ✅ | P2.manifest | 0.01 | 49 file, 652 MB, sha256 tutti verificati |
| ⚠️ | P2.sharp | 0.06 | DEF 5,628,153 righe (ultimo record 49 gg fa); NRT 4,746,570 righe (ultimo record 13 gg fa) |
| ✅ | P2.goes | 0.00 | 33,536 eventi 2010-2026, 3,609 coppie HARP-NOAA |
| ✅ | P9.validation | 0.14 | 25/25 controlli |
| ✅ | P4.labeling | 0.27 | YES: DEF C/M/X 2,101/454/44 su 16,014; NRT C/M/X 1,614/340/32 su 13,474; 2–3 s ciascuno |
| ✅ | P4.determinism | 0.10 | etichette identiche (13,474 righe) |
| ⏭️ | P4.legacy | 0.00 | `/data/cornerstone.sqlite` assente |
| ✅ | P5.def_xgb | 1.25 | test TSS 24h **0.631**; 8,584 campioni |
| ✅ | P5.nrt_xgb | 11.07 | test TSS 24h **0.4131**; 4,281 campioni |
| ✅ | P5.def_all22 | 6.81 | 22 feature, test TSS 24h **0.7042** |
| ✅ | P7.reproduce | 0.61 | dataset e metriche identici (diff. max 0.0) |
| ✅ | P8.bundle | 0.03 | 6 artefatti, 384.5 MB; re-import: 6 saltati |
| ✅ | P10.gui_pages | 0.09 | 11 pagine, 0 eccezioni; box di errore in `compare` |
| ✅ | P1.package_tests | 0.08 | 27 test, 0 falliti |
| ✅ | P1.gui_unit | 0.06 | 44 test, 0 falliti |
| ✅ | P1.gui_integration | 2.07 | 7 test, 0 falliti (122.7 s) |
| ✅ | P6.lstm | 2.15 | test TSS 24h 0.4494 (1 epoca, 2 fold) |
| ✅ | P6.inceptiontime | 0.65 | test TSS 24h 0.4329 (1 epoca, 2 fold) |
| ✅ | P6.transformer | 1.10 | test TSS 24h 0.561 (1 epoca, 2 fold) |
| ❌ | P6.minirocket | 0.15 | `ModuleNotFoundError`: manca `numba` |
| ❌ | P3.update | 17.92 | exit code 1; GOES: connessione scaduta |
| ⏭️ | P6.other_models | 0.00 | non eseguito in modalità quick |
| ⏭️ | P11.inceptiontime_full | 0.00 | non eseguito in modalità quick |
| ✅ | P12.store_unchanged | 0.01 | 49 file invariati |

## 3. Cosa non funziona

### ❌ P6.minirocket — manca `numba` nell'immagine

Il training `trn-20261125-164725-f294` fallisce subito (0.15 min). Traceback (`results.json`, il test non ha un file di log):

```
File "/app/scripts/nightly/campaign.py", line 769, in t
  tr, inf = _train_and_infer(c, f"DEF-short", ch["sp"], model, {}, _short_tcfg(epochs, folds), ["24h"], folds,
File "/app/scripts/nightly/campaign.py", line 550, in _train_and_infer
File "/app/scripts/nightly/campaign.py", line 137, in run_artifact
RuntimeError: trn-20261125-164725-f294 failed: ModuleNotFoundError: MiniRocket requires package 'numba'
to be present in the python environment, but 'numba' was not found. [...] please run: `pip install numba`
```

**Causa probabile:** `numba` è una dipendenza soft di `sktime` (0.39.0), necessaria per MiniRocket, ma non compare in `docker/requirements-cornerstone.txt` (righe 5–27) né in `docker/requirements-gui.txt`. Il commento in `docker/requirements-cornerstone.txt:25-26` dice inoltre che ROCKET/MINIROCKET "still need a CUDA GPU", in contrasto con `docs/ARCHITECTURE.md:50` e `cs_gui/runner/adapters.py:50` (fallback CPU): va riallineato. Correzione ragionevole: aggiungere `numba` (con versione fissata compatibile con numpy 2.2.6) ai requisiti e ricostruire l'immagine. Probabilmente anche ROCKET ha lo stesso problema, ma stanotte non è stato testato (`P6.other_models` è saltato).

**Riprodurre:** `NIGHTLY_ONLY=P6.minirocket scripts/nightly/run.sh --quick` (il test ricostruisce la catena DEF con `c.need("def_chain")`).

### ❌ P3.update — timeout di rete verso GOES

L'aggiornamento incrementale su una copia dello store (sorgenti `harp_noaa,goes` in modalità quick, `campaign.py:822`) termina con exit code 1 dopo 17.9 min. Da `logs/P3.update.log`:

```
HARP-NOAA map: 2,647 HARPs, 3,405 NOAA regions, 3,621 pairs
!! goes failed: URLError: <urlopen error [Errno 110] Connection timed out>
Update finished in 1074s with errors in ['goes']
  File "/opt/cornerstone/cornerstone/data/update.py", line 41, in update_all
RuntimeError: Update failed for: goes: URLError: <urlopen error [Errno 110] Connection timed out>
```

**Causa probabile:** non è un bug del codice, ma la sorgente GOES/NOAA non era raggiungibile dal container (errno 110, timeout TCP). `harp_noaa` invece è stato aggiornato (3,609 → 3,621 coppie), quindi la rete funzionava almeno in parte. L'errore viene sollevato da `cornerstone/data/update.py:41` (pacchetto nell'immagine, non presente in locale: la riga non è verificabile). Da notare che quasi tutti i 17.9 min (1074 s) sono attesa: conviene controllare se `update.py` imposta un timeout breve e un numero limitato di tentativi per GOES. Non si può dire dai file se il problema è temporaneo o dipende dalla rete della macchina.

**Riprodurre:** `NIGHTLY_ONLY=P3.update scripts/nightly/run.sh --quick`, oppure nel container:
`/opt/venv/bin/python -c "import sys; from cornerstone.data import DataStore; from cornerstone.data.update import update_all; update_all(DataStore(sys.argv[1]), sources=['goes'])" <copia_dello_store>`

## 4. Criticità e avvertimenti

- ⚠️ **P2.sharp — store non aggiornato:** l'ultimo record NRT è del 2026-11-11 22:11 (13 giorni fa), l'ultimo evento GOES del 2026-11-10 20:23. L'ultimo record DEF è del 2026-10-06 (49 giorni fa): non si può stabilire dai file se questo ritardo sia normale per la serie definitiva. Dato che anche `P3.update` fallisce su GOES, va lanciato un aggiornamento reale dello store.
- **DEF: valori mancanti e patch vuote:** `LAT_MIN` è nullo nel 4.77% dei record DEF (0% in NRT) e ci sono 268,247 `empty_patch_records` in DEF (0 in NRT). Il test passa, ma la differenza tra le due serie merita un controllo.
- **GOES:** 22 eventi senza picco (`missing_peak`), 1 evento con satellite `////` e 5 con `SDO`. Probabilmente si tratta di valori sporchi della sorgente.
- **NaN nei dataset:** `nan_ratio_overall` è 0.213 (DEF) e 0.318 (NRT).
- **TSS lontani dagli altri:** NRT XGBoost 0.4131 contro DEF 0.631 (−0.22, ma su meno della metà dei campioni: 4,281 contro 8,584). `P5.def_all22` dà 0.7042 contro 0.631 con le 17 feature "clean" (+0.073). L'aumento è plausibile, ma andrebbe confermato su un training completo.
- **Durate:** lo split NRT richiede 5.4 min e quello DEF-all22 5.6 min, contro 0.5 min per lo split DEF di `P5.def_xgb`. Per questo `P5.nrt_xgb` dura 11.07 min (stima quick: 10) e anche il suo training dura ~5 min (artefatti 16:28 → 16:33). `P3.update` dura 17.92 min contro i 10 stimati.
- **P10.gui_pages:** nessuna eccezione, ma la pagina `compare` mostra un box di errore: "Window **24h**: comparison NOT valid — the evaluated samples differ; different inference sources; different splits…". Sembra il messaggio di validazione atteso quando si confrontano run con split diversi, ma va verificato che sia questo il caso.
- **P8.bundle:** `demo_import: "refused"` (registrato nei dettagli, il test passa).
- **Spazio disco:** 55.7 GB liberi, quindi nessuna criticità (lo store occupa 652 MB, il bundle 384.5 MB).

## 5. Confronto con le notti precedenti

Nella cartella `/home/daniele/cornerstone_nightly` non ci sono altre esecuzioni oltre a `2026-11-25_quick`, quindi il confronto non è possibile.

## 6. Test saltati

| test | motivo |
|---|---|
| P4.legacy | database precedente `/data/cornerstone.sqlite` non presente (è opzionale) |
| P6.other_models | non eseguito in modalità quick |
| P11.inceptiontime_full | non eseguito in modalità quick |

Nessun test è stato saltato per la deadline.

## 7. Dati e prestazioni

**Data store** (`/data/datastore`, con manifest): 49 file, 652 MB, sha256 verificati, invariato a fine campagna (`P12.store_unchanged`).

| dataset | righe | HARP | periodo |
|---|---|---|---|
| sharp/DEF | 5,628,153 | 8,301 | 2010-04-30 → 2026-10-06 |
| sharp/NRT | 4,746,570 | 10,749 | 2012-09-14 → 2026-11-11 |
| goes/swpc_events | 33,536 | — | 2010 → 2026-11-10 (C 20,647 · M 2,760 · X 155) |
| harp_noaa | 3,609 coppie | — | — |

**Tempi della pipeline (min):**

| catena | labels | dataset | split | train+inference | totale test |
|---|---|---|---|---|---|
| DEF M+ 24h (P5.def_xgb) | 0.1 | 0.1 | 0.5 | 0.6 | 1.25 |
| NRT M+ 24h (P5.nrt_xgb) | 0.1 | 0.1 | 5.4 | n.d. (~5.5 dagli orari degli artefatti) | 11.07 |
| DEF all 22 (P5.def_all22) | 0.1 | 0.2 | 5.6 | n.d. | 6.81 |

**Metriche dei modelli** (finestra 24h, split DEF salvo NRT; i modelli P6 usano 1 epoca e 2 fold, quindi sono solo smoke test):

| modello | CV val TSS | test TSS (inference) | training (min) |
|---|---|---|---|
| XGBoost DEF (17 feat.) | 0.6629 | 0.631 (CV test medio 0.6081) | — |
| XGBoost NRT | n.d. | 0.4131 | — |
| XGBoost DEF all 22 | n.d. | 0.7042 | — |
| LSTM | 0.4823 | 0.4494 | 1.9 |
| InceptionTime | 0.4884 | 0.4329 | 0.4 |
| Transformer | 0.5705 | 0.561 | 0.9 |
| MiniRocket | — | ❌ fallito | — |

Non ci sono leakage: nessun HARP o campione condiviso tra test e train/val, né tra train e val nei 5 fold (DEF e NRT).

![SHARP records per month](figures/sharp_records_per_month.png)
![GOES events per year](figures/goes_events_per_year.png)
![Labels YES rate](figures/labels_yes_rate.png)

## 8. Ambiente

- **Pacchetto** `cornerstone_db_package`: commit `984804b` (`984804b6efabe66252da5416c6da0e0d760f4c36`, main, non dirty).
- **GUI** `cornerstone_gui`: commit `8c2362d` (`8c2362d551f9e9dc84bfce6fbae2d2e61849c761`, main, non dirty).
- **Hardware:** 12 CPU, 24.6 GB RAM, nessuna CUDA (training su CPU), 55.7 GB di disco libero.
- **Software:** Python 3.10.22, numpy 2.2.6, pandas 2.3.3, torch 2.9.1+cpu, xgboost-cpu 3.1.2, sktime 0.39.0, scikit-learn 1.7.2, h5py 3.15.1, clearml 2.0.2, cupy-cuda12x 13.6.0. `numba` non c'è.
- **Build dell'immagine** (`build.json`): ok in 4 s, terminata alle 16:20:48 UTC, immagine `cornerstone-gui:latest`. Il layer del pacchetto (`ADD ...cornerstone_db_package.git#main`) era in cache, quindi il commit del pacchetto non è cambiato rispetto alla build precedente. Avviso non bloccante: "buildx isn't installed".
