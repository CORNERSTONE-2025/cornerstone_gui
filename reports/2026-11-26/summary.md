# CORNERSTONE nightly tests — 2026-11-26

Started 2026-11-26T20:00:11, last update 2026-11-26T22:51:18, deadline 2026-11-27T06:30.

✅ pass 20 · ⚠️ warn 3 · ❌ fail 1 · ⏭️ skip 1

Package `984804b`, GUI `871c1a2`; 12 CPU, 24.6 GB RAM, CUDA False, 41.8 GB free.

| | test | min | result |
|---|---|---|---|
| ✅ | **P0.env** Versions, hardware, disk, data store present | 0.02 | 12 CPU, 24.6 GB RAM, CUDA False, 42 GB free |
| ✅ | **P2.manifest** Every store file matches the sha256 of the manifest | 0.01 | 49 files, 652 MB, all verified |
| ⚠️ | **P2.sharp** SHARP DEF/NRT: rows, keys, order, missing values, freshness | 0.05 | NRT: last record 14 days ago (store not updated?) |
| ⚠️ | **P2.goes** GOES events and HARP-NOAA map: coverage and consistency | 0.0 | Last GOES event 15 days ago |
| ✅ | **P9.validation** Wrong parameters are refused with a message | 0.13 | 25 checks |
| ✅ | **P4.labeling** Labels DEF/NRT × C/M/X: invariants, monotonicity, timing | 0.5 | DEF_C 6,689 YES/48,393 (5s), DEF_M 1,541 YES/48,393 (4s), DEF_X 145 YES/48,393 (5s), NRT_C 5,716 YES/43,275 (6s), NRT_M 1,323 YES/43,275 (5s), NRT_X 121 YES/43,275 (5s) |
| ✅ | **P4.determinism** Labeling twice with the same parameters gives identical labels | 0.18 | identical (43,275 rows); S_s = C, L = 48 h invariants hold |
| ⏭️ | **P4.legacy** Comparison with the static tables of the former database | 0.0 | /data/cornerstone.sqlite not present on this machine (the former database is optional) |
| ✅ | **P5.def_xgb** DEF M+ 24 h: labels → dataset → split → XGBoost → inference | 2.81 | test TSS {'3h': 0.5942, '6h': 0.5964, '12h': 0.5811, '24h': 0.5793}; 21,325 samples |
| ✅ | **P5.nrt_xgb** NRT M+ 24 h: same chain on the NRT series | 1.95 | test TSS {'24h': 0.6721}; 17,457 samples |
| ✅ | **P5.def_all22** DEF with all 22 SHARP keywords (features 'all') + XGBoost | 4.31 | 22 features, test TSS {'24h': 0.7962} |
| ✅ | **P7.reproduce** Re-running dataset and XGBoost training gives identical results | 2.02 | dataset and XGBoost metrics identical |
| ✅ | **P8.bundle** Export the DEF chain as a bundle and import it into an empty workspace | 0.07 | 6 artifacts, 962.7 MB, re-import skipped 6 |
| ✅ | **P10.gui_pages** Every GUI page renders without exceptions (Streamlit AppTest) | 0.1 | 11 pages, no exceptions; error boxes shown by ['compare'] (see details) |
| ✅ | **P1.package_tests** Package unit tests (cornerstone_db_package/tests) | 0.3 | 27 tests, 0 failed, 0 skipped |
| ✅ | **P1.gui_unit** GUI unit tests | 0.09 | 236 tests, 0 failed, 0 skipped |
| ✅ | **P1.gui_integration** GUI integration tests (whole chain on synthetic data) | 2.05 | 7 tests, 0 failed, 0 skipped |
| ✅ | **P6.lstm** LSTM: short training + inference on the DEF split | 18.09 | completed in 18.1 min (5 epochs, 2 folds), test TSS {'24h': 0.5002} |
| ✅ | **P6.inceptiontime** INCEPTIONTIME: short training + inference on the DEF split | 2.71 | completed in 2.7 min (5 epochs, 2 folds), test TSS {'24h': 0.5063} |
| ✅ | **P6.transformer** TRANSFORMER: short training + inference on the DEF split | 7.82 | completed in 7.8 min (5 epochs, 2 folds), test TSS {'24h': 0.5213} |
| ✅ | **P6.minirocket** MINIROCKET: short training + inference on the DEF split | 2.81 | completed in 2.8 min (5 epochs, 2 folds), test TSS {'24h': 0.2718} |
| ❌ | **P3.update** Incremental update from JSOC/NOAA on a copy of the store | 10.4 | update exited with code 1 after 10.4 min (see the log) |
| ✅ | **P6.other_models** Other models offered by the GUI: 1 epoch, 1 fold, 3 h window | 14.74 | working: ['ONEFITSALL', 'Autoformer', 'DLinear', 'FEDformer', 'LightTS', 'Nonstationary_Transformer', 'Pyraformer', 'Reformer', 'TimesNet', 'Informer', 'ROCKET']; failing: [] |
| ⚠️ | **P11.inceptiontime_full** InceptionTime as the reference experiment (4 windows, 5 folds, 100 epochs) | 99.95 | test TSS differs from the reference by more than 0.05: {'12h': -0.067, '24h': -0.071, '3h': -0.079, '6h': -0.072} |
| ✅ | **P12.store_unchanged** The tests did not modify the data store | 0.01 | 49 files unchanged |

## Figures

![goes_events_per_year](figures/goes_events_per_year.png)
![labels_yes_rate](figures/labels_yes_rate.png)
![sharp_records_per_month](figures/sharp_records_per_month.png)
