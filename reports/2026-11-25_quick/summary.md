# CORNERSTONE nightly tests — 2026-11-25

Started 2026-11-25T16:20:50, last update 2026-11-25T17:05:29, deadline 2026-11-26T07:30 (quick mode).

✅ pass 19 · ⚠️ warn 1 · ❌ fail 2 · ⏭️ skip 3

Package `984804b`, GUI `8c2362d`; 12 CPU, 24.6 GB RAM, CUDA False, 55.7 GB free.

| | test | min | result |
|---|---|---|---|
| ✅ | **P0.env** Versions, hardware, disk, data store present | 0.02 | 12 CPU, 24.6 GB RAM, CUDA False, 56 GB free |
| ✅ | **P2.manifest** Every store file matches the sha256 of the manifest | 0.01 | 49 files, 652 MB, all verified |
| ⚠️ | **P2.sharp** SHARP DEF/NRT: rows, keys, order, missing values, freshness | 0.06 | NRT: last record 13 days ago (store not updated?) |
| ✅ | **P2.goes** GOES events and HARP-NOAA map: coverage and consistency | 0.0 | 33,536 events 2010-2026, 3,609 HARP-NOAA pairs |
| ✅ | **P9.validation** Wrong parameters are refused with a message | 0.14 | 25 checks |
| ✅ | **P4.labeling** Labels DEF/NRT × C/M/X: invariants, monotonicity, timing | 0.27 | DEF_C 2,101 YES/16,014 (2s), DEF_M 454 YES/16,014 (2s), DEF_X 44 YES/16,014 (2s), NRT_C 1,614 YES/13,474 (3s), NRT_M 340 YES/13,474 (3s), NRT_X 32 YES/13,474 (3s) |
| ✅ | **P4.determinism** Labeling twice with the same parameters gives identical labels | 0.1 | identical (13,474 rows); S_s = C, L = 48 h invariants hold |
| ⏭️ | **P4.legacy** Comparison with the static tables of the former database | 0.0 | /data/cornerstone.sqlite not present on this machine (the former database is optional) |
| ✅ | **P5.def_xgb** DEF M+ 24 h: labels → dataset → split → XGBoost → inference | 1.25 | test TSS {'24h': 0.631}; 8,584 samples |
| ✅ | **P5.nrt_xgb** NRT M+ 24 h: same chain on the NRT series | 11.07 | test TSS {'24h': 0.4131}; 4,281 samples |
| ✅ | **P5.def_all22** DEF with all 22 SHARP keywords (features 'all') + XGBoost | 6.81 | 22 features, test TSS {'24h': 0.7042} |
| ✅ | **P7.reproduce** Re-running dataset and XGBoost training gives identical results | 0.61 | dataset and XGBoost metrics identical |
| ✅ | **P8.bundle** Export the DEF chain as a bundle and import it into an empty workspace | 0.03 | 6 artifacts, 384.5 MB, re-import skipped 6 |
| ✅ | **P10.gui_pages** Every GUI page renders without exceptions (Streamlit AppTest) | 0.09 | 11 pages, no exceptions; error boxes shown by ['compare'] (see details) |
| ✅ | **P1.package_tests** Package unit tests (cornerstone_db_package/tests) | 0.08 | 27 tests, 0 failed, 0 skipped |
| ✅ | **P1.gui_unit** GUI unit tests | 0.06 | 44 tests, 0 failed, 0 skipped |
| ✅ | **P1.gui_integration** GUI integration tests (whole chain on synthetic data) | 2.07 | 7 tests, 0 failed, 0 skipped |
| ✅ | **P6.lstm** LSTM: short training + inference on the DEF split | 2.15 | completed in 2.2 min (1 epochs, 2 folds), test TSS {'24h': 0.4494} |
| ✅ | **P6.inceptiontime** INCEPTIONTIME: short training + inference on the DEF split | 0.65 | completed in 0.7 min (1 epochs, 2 folds), test TSS {'24h': 0.4329} |
| ✅ | **P6.transformer** TRANSFORMER: short training + inference on the DEF split | 1.1 | completed in 1.1 min (1 epochs, 2 folds), test TSS {'24h': 0.561} |
| ❌ | **P6.minirocket** MINIROCKET: short training + inference on the DEF split | 0.15 | RuntimeError: trn-20261125-164725-f294 failed: ModuleNotFoundError: MiniRocket requires package 'numba' to be present in the python environment, but 'numba' was not found. 'numba' is a dependency of MiniRocket and required to construct it. To install the requirement 'numba', please run: `pip install numba`  |
| ❌ | **P3.update** Incremental update from JSOC/NOAA on a copy of the store | 17.92 | update exited with code 1 after 17.9 min (see the log) |
| ⏭️ | **P6.other_models** Other models offered by the GUI: 1 epoch, 1 fold, 3 h window | 0.0 | not run in quick mode |
| ⏭️ | **P11.inceptiontime_full** InceptionTime as the reference experiment (4 windows, 5 folds, 100 epochs) | 0.0 | not run in quick mode |
| ✅ | **P12.store_unchanged** The tests did not modify the data store | 0.01 | 49 files unchanged |

## Figures

![goes_events_per_year](figures/goes_events_per_year.png)
![labels_yes_rate](figures/labels_yes_rate.png)
![sharp_records_per_month](figures/sharp_records_per_month.png)
