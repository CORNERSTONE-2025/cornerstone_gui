import datetime as dt

import pandas as pd
import plotly.express as px
import streamlit as st

from cs_gui import ui
from cs_gui.pipeline import (CMASK_STRATEGIES, DATA_VERSIONS, MISSINGNESS_MODES, TEMPORAL_FILTER_MODES,
                             default_split_config, validate_dataset, validate_split)

c = ui.page_header("Creazione dei dataset",
                   "1) Estrazione delle serie temporali SHARP per ogni evento etichettato (HDF5). "
                   "2) Selezione feature, filtri e suddivisione train/validation/test.")

t_ext, t_split, t_list = st.tabs(["1 · Estrazione HDF5", "2 · Filtri e split", "Dataset prodotti"])

# ======================================================================= extraction
with t_ext:
    lab_id = ui.artifact_picker("Label di partenza", "labels", key="ds_labels")
    if lab_id:
        lab = c.store.get(lab_id)
        st.caption(f"{lab['summary']['n_rows']:,} eventi · {lab['config']['table_type']} · "
                   f"{lab['config']['label_type']} · Δ={lab['config']['delta_hours']}h")
        g1, g2 = st.columns(2)
        with g1:
            st.markdown("**Feature HMI/SHARP**")
            dv = st.radio("Versione della tabella feature (data_version)", list(DATA_VERSIONS),
                          format_func=DATA_VERSIONS.get)
            st.caption("La selezione delle singole feature avviene nel passo 2 (filtri e split).")
        with g2:
            st.markdown("**Campionamento temporale**")
            wh = st.number_input("Finestra storica prima di t_X (window_hours)", 1, 240, 24)
            bm = st.number_input("Margine attorno a t_X in minuti (buffer_minutes)", 0, 120, 6)
            si = st.number_input("Passo di campionamento in minuti (sampling_interval)", 1, 120, 12,
                                 help="T_OBS viene arrotondato a questo passo (la cadenza SHARP è 12 min).")
            st.caption(f"Timesteps attesi per campione ≈ {int(wh * 60 / si) + 1}")
        if int(wh) != int(lab["config"]["delta_hours"]):
            st.info(f"Nota: il pacchetto nomina la label nel file HDF5 con window_hours "
                    f"(`{lab['config']['label_type']}_delta{wh}`), mentre l'orizzonte di previsione delle label "
                    f"resta Δ={lab['config']['delta_hours']}h. La GUI registra entrambi i valori.")
        sub = st.checkbox("Sottocampione casuale di eventi (solo per prove rapide — test_mode del pacchetto)")
        ssize = st.number_input("Numero di eventi", 1, int(lab["summary"]["n_rows"]), min(500, int(lab["summary"]["n_rows"])),
                                disabled=not sub)
        seed = st.number_input("Seed del sottocampionamento", 0, 2**31 - 1, 42, disabled=not sub)
        name = st.text_input("Nome", f"Dataset {lab['config']['label_type']} {dv} w{wh}h" + (f" (sub {ssize})" if sub else ""))
        desc = st.text_area("Descrizione", "", key="ds_desc", height=70)
        cfg = {"data_version": dv, "window_hours": int(wh), "buffer_minutes": int(bm), "sampling_interval": int(si),
               "subsample": sub, "subsample_size": int(ssize) if sub else 0, "random_seed": int(seed)}
        ui.config_summary("estrazione", cfg, {"label": lab_id,
                                              "tabella feature": f"SHARPC_{dv}_{lab['config']['table_type']}"})
        if ui.show_errors(validate_dataset(cfg, lab, c.settings)) and st.button("Crea ed esegui estrazione", type="primary"):
            ui.create_and_launch(lambda: c.pipeline.create_dataset(name, lab_id, cfg, desc))
    ui.job_monitor(["dataset"])

# ======================================================================= split
with t_split:
    ds_id = ui.artifact_picker("Dataset HDF5 di partenza", "dataset", key="sp_ds")
    info = ui.package_info()
    if ds_id and info:
        ds = c.store.get(ds_id)
        s = ds["summary"]
        st.caption(f"{s['n_samples']:,} campioni · {s['n_features']} feature · {s['timesteps']} timesteps · "
                   f"{s['time_start'][:10]} → {s['time_end'][:10]} · {s['n_active_regions']} AR")
        base = default_split_config(c.settings)
        lo = dt.date.fromisoformat(s["time_start"][:10])
        hi = dt.date.fromisoformat(s["time_end"][:10])
        cfg = dict(base)

        with st.expander("Feature", expanded=True):
            mode = st.radio("Modalità di selezione (FEATURE_SELECTION_MODE)", ["direct", "criterion"], horizontal=True,
                            format_func=lambda x: {"direct": "Elenco diretto", "criterion": "Gruppo fisico predefinito"}[x])
            cfg["FEATURE_SELECTION_MODE"] = mode
            if mode == "direct":
                default = [f for f in base.get("DIRECT_FEATURES", []) if f in s["feature_names"]] or s["feature_names"]
                cfg["DIRECT_FEATURES"] = st.multiselect("Feature da usare", s["feature_names"], default)
            else:
                groups = info["feature_groups"]
                crit = st.selectbox("Gruppo (FEATURE_CRITERION)", list(groups))
                cfg["FEATURE_CRITERION"] = crit
                ok = [f for f in groups[crit] if f in s["feature_names"]]
                st.caption(f"Feature del gruppo: {groups[crit]} — presenti nel dataset: {ok}")
        with st.expander("Dati mancanti e qualità", expanded=True):
            mm = st.selectbox("Filtro sui valori mancanti (MISSINGNESS_FILTER_MODE)", MISSINGNESS_MODES,
                              index=MISSINGNESS_MODES.index(base.get("MISSINGNESS_FILTER_MODE", "multi_window_empty")),
                              help="multi_window_empty: scarta i campioni vuoti in una qualunque delle finestre; "
                                   "multi_window / single_window: scarta se la frazione di NaN supera MAX_NAN_RATIO.")
            cfg["MISSINGNESS_FILTER_MODE"] = mm
            ts = st.number_input("TIMESTEPS (lunghezza delle serie salvate)", 1, int(s["timesteps"]),
                                 min(int(base.get("TIMESTEPS", 121)), int(s["timesteps"])))
            cfg["TIMESTEPS"] = int(ts)
            if mm in ("single_window", "multi_window"):
                cfg["MAX_NAN_RATIO"] = st.slider("MAX_NAN_RATIO", 0.0, 1.0, float(base.get("MAX_NAN_RATIO") or 0.95), 0.01)
            if mm in ("multi_window", "multi_window_empty"):
                opts = sorted(set([w for w in (base.get("TIME_WINDOWS") or []) if w <= ts] + [1, 16, 31, 61, int(ts)]))
                cfg["TIME_WINDOWS"] = st.multiselect("TIME_WINDOWS (timesteps finali controllati)",
                                                     [o for o in opts if o <= ts],
                                                     [w for w in (base.get("TIME_WINDOWS") or []) if w <= ts])
            cfg["CMASK_FILTER"] = st.checkbox("Filtro qualità CMASK (marcato 'non completamente testato' nel pacchetto)",
                                              bool(base.get("CMASK_FILTER")))
            if cfg["CMASK_FILTER"]:
                a, b, d = st.columns(3)
                cfg["CMASK_THRESHOLD"] = a.number_input("CMASK_THRESHOLD", 0.0, 1e7, float(base.get("CMASK_THRESHOLD", 5000)))
                cfg["CMASK_STRATEGY"] = b.selectbox("CMASK_STRATEGY", CMASK_STRATEGIES,
                                                    CMASK_STRATEGIES.index(base.get("CMASK_STRATEGY", "average")))
                cfg["CMASK_MAJORITY_THRESHOLD"] = d.slider("CMASK_MAJORITY_THRESHOLD", 0.0, 1.0,
                                                           float(base.get("CMASK_MAJORITY_THRESHOLD", 0.5)))
        with st.expander("Intervalli temporali", expanded=True):
            tfm = st.selectbox("Filtro temporale (TEMPORAL_FILTER_MODE)", TEMPORAL_FILTER_MODES,
                               format_func=lambda x: {None: "nessuno", "select": "select: tieni solo l'intervallo",
                                                      "filter": "filter: tieni solo l'intervallo (variante)",
                                                      "exclude": "exclude: escludi l'intervallo"}[x])
            cfg["TEMPORAL_FILTER_MODE"] = tfm
            if tfm:
                tw = st.date_input("Intervallo (TEMPORAL_WINDOW)", (lo, hi), min_value=lo, max_value=hi, key="tw")
                cfg["TEMPORAL_WINDOW"] = [str(tw[0]), str(tw[1])] if len(tw) == 2 else None
            else:
                cfg["TEMPORAL_WINDOW"] = None
            cfg["MULTI_PERIOD_FILTER"] = st.checkbox("Escludi periodi problematici (MULTI_PERIOD_FILTER)",
                                                     bool(base.get("MULTI_PERIOD_FILTER")))
            if cfg["MULTI_PERIOD_FILTER"]:
                pp = st.data_editor(pd.DataFrame(base.get("PROBLEMATIC_PERIODS") or [], columns=["inizio", "fine"]),
                                    num_rows="dynamic", key="pp",
                                    column_config={"inizio": st.column_config.TextColumn(help="YYYY-MM-DD"),
                                                   "fine": st.column_config.TextColumn(help="YYYY-MM-DD")})
                cfg["PROBLEMATIC_PERIODS"] = [[str(a), str(b)] for a, b in pp.dropna().values.tolist()]
            else:
                cfg["PROBLEMATIC_PERIODS"] = []
        with st.expander("Suddivisione train / validation / test", expanded=True):
            strat = st.radio("Strategia (SPLITTING_STRATEGY)", ["random", "chronological"], horizontal=True,
                             format_func=lambda x: {"random": "Casuale, raggruppata per Active Region",
                                                    "chronological": "Cronologica (test nel futuro)"}[x])
            cfg["SPLITTING_STRATEGY"] = strat
            cfg["TEST_CUTOFF_DATE"], cfg["TEST_TIME_WINDOW"] = None, None
            if strat == "random":
                tsz = st.slider("TEST_SIZE", 0.05, 0.5, float(base.get("TEST_SIZE", 0.2)), 0.05)
                cfg["TEST_SIZE"] = tsz
                st.caption(f"Il pacchetto usa StratifiedGroupKFold con {int(1 / tsz)} gruppi: frazione di test "
                           f"effettiva ≈ {1 / int(1 / tsz):.3f}. Nessuna AR è condivisa tra test e train/val.")
            else:
                how = st.radio("Definizione del test", ["default", "cutoff", "window"], horizontal=True,
                               format_func=lambda x: {"default": "ultimo TEST_SIZE del periodo",
                                                      "cutoff": "dopo una data (TEST_CUTOFF_DATE)",
                                                      "window": "in una finestra (TEST_TIME_WINDOW)"}[x])
                cfg["TEST_SIZE"] = st.slider("TEST_SIZE", 0.05, 0.5, float(base.get("TEST_SIZE", 0.2)), 0.05)
                if how == "cutoff":
                    cfg["TEST_CUTOFF_DATE"] = str(st.date_input("TEST_CUTOFF_DATE", lo + (hi - lo) * 0.8, lo, hi))
                    st.caption("Le AR che finiscono prima del cutoff vanno in train/val, quelle che iniziano dopo in test; "
                               "le AR a cavallo vengono scartate.")
                elif how == "window":
                    w = st.date_input("TEST_TIME_WINDOW", (lo + (hi - lo) * 0.4, lo + (hi - lo) * 0.6), lo, hi, key="ttw")
                    cfg["TEST_TIME_WINDOW"] = [str(w[0]), str(w[1])] if len(w) == 2 else None
            a, b = st.columns(2)
            cfg["N_FOLDS"] = int(a.number_input("N_FOLDS (fold di cross-validation)", 2, 10, int(base.get("N_FOLDS", 5))))
            cfg["RANDOM_STATE"] = int(b.number_input("RANDOM_STATE (seed)", 0, 2**31 - 1, int(base.get("RANDOM_STATE", 42))))
            cfg["FORCE_SEPARATION"] = st.checkbox("FORCE_SEPARATION (scarta i campioni sovrapposti tra train/val e test)",
                                                  bool(base.get("FORCE_SEPARATION", True)))
            cfg["SAVE_PLOTS"] = st.checkbox("Salva i grafici delle distribuzioni (SAVE_PLOTS)", bool(base.get("SAVE_PLOTS", True)))
            st.caption("Bilanciamento delle classi: il pacchetto non ricampiona i dati; lo sbilanciamento si gestisce "
                       "nel training (loss focal/pesata per i modelli PyTorch, scale_pos_weight per XGBoost).")
        name = st.text_input("Nome", f"Split {strat} da {ds['name']}", key="sp_name")
        desc = st.text_area("Descrizione", "", key="sp_desc", height=70)
        ui.config_summary("filtri e split", cfg, {"dataset": ds_id, "LABEL_TYPE": s["label_key"]})
        if ui.show_errors(validate_split(cfg, ds, info)) and st.button("Crea ed esegui split", type="primary"):
            ui.create_and_launch(lambda: c.pipeline.create_split(name, ds_id, cfg, desc))
    ui.job_monitor(["split"])

# ======================================================================= list
with t_list:
    kind = st.radio("Tipo", ["dataset", "split"], horizontal=True,
                    format_func=lambda x: {"dataset": "Dataset HDF5", "split": "Split"}[x])
    if kind == "dataset":
        sel = ui.artifact_list("dataset", lambda m: {
            "campioni": m["summary"].get("n_samples"), "feature": m["summary"].get("n_features"),
            "AR": m["summary"].get("n_active_regions"), "% YES": round(100 * m["summary"].get("positive_rate", 0), 2),
            "sottocampione": m["config"].get("subsample")}, key="dsl")
    else:
        sel = ui.artifact_list("split", lambda m: {
            "strategia": m["config"].get("SPLITTING_STRATEGY"), "fold": m["config"].get("N_FOLDS"),
            "campioni tenuti": m["summary"].get("n_samples_kept"), "feature": m["summary"].get("n_features")}, key="spl")
    if sel:
        m = c.store.get(sel)
        ui.artifact_status_box(m)
        s = m["summary"]
        if m["status"] == "completed" and m["type"] == "dataset":
            k = st.columns(5)
            k[0].metric("Campioni", f"{s['n_samples']:,}")
            k[1].metric("Feature", s["n_features"])
            k[2].metric("Active Regions", s["n_active_regions"])
            k[3].metric("Positivi", f"{100 * s['positive_rate']:.2f}%")
            k[4].metric("Eventi esclusi (dati assenti)", s.get("n_excluded_events", 0))
            st.markdown(f"Intervallo temporale: **{s['time_start']} → {s['time_end']}** · timesteps {s['timesteps']}"
                        f" · orizzonte label Δ={s.get('label_horizon_hours')}h")
            c1, c2 = st.columns(2)
            cc = pd.DataFrame({"classe": list(s["class_counts"]), "n": list(s["class_counts"].values())})
            c1.plotly_chart(px.bar(cc, x="classe", y="n", title="Distribuzione delle classi"), width="stretch")
            nr = pd.DataFrame({"feature": list(s["nan_ratio_per_feature"]),
                               "% NaN": [100 * v for v in s["nan_ratio_per_feature"].values()]})
            c2.plotly_chart(px.bar(nr, x="feature", y="% NaN", title=f"Valori mancanti (su {s['nan_stats_on_samples']} campioni)"),
                            width="stretch")
            if s.get("excluded_reasons"):
                st.markdown("**Esclusioni in estrazione:**")
                st.json(s["excluded_reasons"])
        if m["status"] == "completed" and m["type"] == "split":
            k = st.columns(4)
            k[0].metric("Campioni in ingresso", f"{s['n_samples_input']:,}")
            k[1].metric("Campioni dopo i filtri", f"{s['n_samples_kept']:,}")
            k[2].metric("Esclusi dai filtri", f"{s['n_samples_excluded']:,}")
            k[3].metric("Feature", s["n_features"])
            rows = [{"split": n, **{kk: v[kk] for kk in ("n_samples", "n_positive", "n_active_regions", "time_start", "time_end")},
                     "% YES": round(100 * v["positive_rate"], 2)} for n, v in s["files"].items()]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            lk = s["leakage"]
            ok = lk["test_vs_trainval_shared_harps"] == 0 and lk["test_vs_trainval_shared_samples"] == 0
            (st.success if ok else st.error)(
                f"Separazione test ↔ train/val: AR condivise = {lk['test_vs_trainval_shared_harps']}, "
                f"campioni condivisi = {lk['test_vs_trainval_shared_samples']}. "
                f"AR condivise train↔val per fold: {lk['train_vs_val_shared_harps_per_fold']}")
            st.caption("Feature: " + ", ".join(s["feature_names"]))
            plots = sorted((c.store.path(sel) / "split" / "plots").glob("*.png"))
            for p in plots[:6]:
                st.image(str(p), caption=p.name)
        with st.expander("Configurazione"):
            st.json(m["config"])
        with st.expander("Log"):
            ui.log_viewer(sel)
