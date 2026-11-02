import datetime as dt

import pandas as pd
import plotly.express as px
import streamlit as st

from cs_gui import ui
from cs_gui.pipeline import (CMASK_STRATEGIES, DATA_VERSIONS, MISSINGNESS_MODES, TEMPORAL_FILTER_MODES,
                             default_split_config, validate_dataset, validate_split)

c = ui.page_header("Dataset creation",
                   "1) Extraction of the SHARP time series for every labeled event (HDF5). "
                   "2) Feature selection, filters and train/validation/test split.")

t_ext, t_split, t_list = st.tabs(["1 · HDF5 extraction", "2 · Filters and split", "Produced datasets"])

# ======================================================================= extraction
with t_ext:
    lab_id = ui.artifact_picker("Input labels", "labels", key="ds_labels")
    if lab_id:
        lab = c.store.get(lab_id)
        st.caption(f"{lab['summary']['n_rows']:,} events · {lab['config']['table_type']} · "
                   f"{lab['config']['label_type']} · Δ={lab['config']['delta_hours']}h")
        g1, g2 = st.columns(2)
        with g1:
            st.markdown("**HMI/SHARP features**")
            dv = st.radio("Feature table version (data_version)", list(DATA_VERSIONS),
                          format_func=DATA_VERSIONS.get)
            st.caption("Individual features are selected in step 2 (filters and split).")
        with g2:
            st.markdown("**Time sampling**")
            wh = st.number_input("History window before t_X (window_hours)", 1, 240, 24)
            bm = st.number_input("Margin around t_X in minutes (buffer_minutes)", 0, 120, 6)
            si = st.number_input("Sampling step in minutes (sampling_interval)", 1, 120, 12,
                                 help="T_OBS is rounded to this step (the SHARP cadence is 12 min).")
            st.caption(f"Expected timesteps per sample ≈ {int(wh * 60 / si) + 1}")
        if int(wh) != int(lab["config"]["delta_hours"]):
            st.info(f"Note: the package names the label in the HDF5 file after window_hours "
                    f"(`{lab['config']['label_type']}_delta{wh}`), while the forecast horizon of the labels "
                    f"stays Δ={lab['config']['delta_hours']}h. The GUI records both values.")
        sub = st.checkbox("Random subsample of events (for quick tests only — package test_mode)")
        ssize = st.number_input("Number of events", 1, int(lab["summary"]["n_rows"]), min(500, int(lab["summary"]["n_rows"])),
                                disabled=not sub)
        seed = st.number_input("Subsampling seed", 0, 2**31 - 1, 42, disabled=not sub)
        name = st.text_input("Name", f"Dataset {lab['config']['label_type']} {dv} w{wh}h" + (f" (sub {ssize})" if sub else ""))
        desc = st.text_area("Description", "", key="ds_desc", height=70)
        cfg = {"data_version": dv, "window_hours": int(wh), "buffer_minutes": int(bm), "sampling_interval": int(si),
               "subsample": sub, "subsample_size": int(ssize) if sub else 0, "random_seed": int(seed)}
        ui.config_summary("extraction", cfg, {"label": lab_id,
                                              "feature table": f"SHARPC_{dv}_{lab['config']['table_type']}"})
        if ui.show_errors(validate_dataset(cfg, lab, c.settings)) and st.button("Create and run extraction", type="primary"):
            ui.create_and_launch(lambda: c.pipeline.create_dataset(name, lab_id, cfg, desc))
    ui.job_monitor(["dataset"])

# ======================================================================= split
with t_split:
    ds_id = ui.artifact_picker("Input HDF5 dataset", "dataset", key="sp_ds")
    info = ui.package_info()
    if ds_id and info:
        ds = c.store.get(ds_id)
        s = ds["summary"]
        st.caption(f"{s['n_samples']:,} samples · {s['n_features']} features · {s['timesteps']} timesteps · "
                   f"{s['time_start'][:10]} → {s['time_end'][:10]} · {s['n_active_regions']} AR")
        base = default_split_config(c.settings)
        lo = dt.date.fromisoformat(s["time_start"][:10])
        hi = dt.date.fromisoformat(s["time_end"][:10])
        cfg = dict(base)

        with st.expander("Features", expanded=True):
            mode = st.radio("Selection mode (FEATURE_SELECTION_MODE)", ["direct", "criterion"], horizontal=True,
                            format_func=lambda x: {"direct": "Direct list", "criterion": "Predefined physical group"}[x])
            cfg["FEATURE_SELECTION_MODE"] = mode
            if mode == "direct":
                default = [f for f in base.get("DIRECT_FEATURES", []) if f in s["feature_names"]] or s["feature_names"]
                cfg["DIRECT_FEATURES"] = st.multiselect("Features to use", s["feature_names"], default)
            else:
                groups = info["feature_groups"]
                crit = st.selectbox("Group (FEATURE_CRITERION)", list(groups))
                cfg["FEATURE_CRITERION"] = crit
                ok = [f for f in groups[crit] if f in s["feature_names"]]
                st.caption(f"Group features: {groups[crit]} — present in the dataset: {ok}")
        with st.expander("Missing data and quality", expanded=True):
            mm = st.selectbox("Missing values filter (MISSINGNESS_FILTER_MODE)", MISSINGNESS_MODES,
                              index=MISSINGNESS_MODES.index(base.get("MISSINGNESS_FILTER_MODE", "multi_window_empty")),
                              help="multi_window_empty: drops samples empty in any of the windows; "
                                   "multi_window / single_window: drops if the NaN fraction exceeds MAX_NAN_RATIO.")
            cfg["MISSINGNESS_FILTER_MODE"] = mm
            ts = st.number_input("TIMESTEPS (length of the saved series)", 1, int(s["timesteps"]),
                                 min(int(base.get("TIMESTEPS", 121)), int(s["timesteps"])))
            cfg["TIMESTEPS"] = int(ts)
            if mm in ("single_window", "multi_window"):
                cfg["MAX_NAN_RATIO"] = st.slider("MAX_NAN_RATIO", 0.0, 1.0, float(base.get("MAX_NAN_RATIO") or 0.95), 0.01)
            if mm in ("multi_window", "multi_window_empty"):
                opts = sorted(set([w for w in (base.get("TIME_WINDOWS") or []) if w <= ts] + [1, 16, 31, 61, int(ts)]))
                cfg["TIME_WINDOWS"] = st.multiselect("TIME_WINDOWS (final timesteps checked)",
                                                     [o for o in opts if o <= ts],
                                                     [w for w in (base.get("TIME_WINDOWS") or []) if w <= ts])
            cfg["CMASK_FILTER"] = st.checkbox("CMASK quality filter (marked 'not fully tested' in the package)",
                                              bool(base.get("CMASK_FILTER")))
            if cfg["CMASK_FILTER"]:
                a, b, d = st.columns(3)
                cfg["CMASK_THRESHOLD"] = a.number_input("CMASK_THRESHOLD", 0.0, 1e7, float(base.get("CMASK_THRESHOLD", 5000)))
                cfg["CMASK_STRATEGY"] = b.selectbox("CMASK_STRATEGY", CMASK_STRATEGIES,
                                                    CMASK_STRATEGIES.index(base.get("CMASK_STRATEGY", "average")))
                cfg["CMASK_MAJORITY_THRESHOLD"] = d.slider("CMASK_MAJORITY_THRESHOLD", 0.0, 1.0,
                                                           float(base.get("CMASK_MAJORITY_THRESHOLD", 0.5)))
        with st.expander("Time intervals", expanded=True):
            tfm = st.selectbox("Time filter (TEMPORAL_FILTER_MODE)", TEMPORAL_FILTER_MODES,
                               format_func=lambda x: {None: "none", "select": "select: keep only the interval",
                                                      "filter": "filter: keep only the interval (variant)",
                                                      "exclude": "exclude: drop the interval"}[x])
            cfg["TEMPORAL_FILTER_MODE"] = tfm
            if tfm:
                tw = st.date_input("Interval (TEMPORAL_WINDOW)", (lo, hi), min_value=lo, max_value=hi, key="tw")
                cfg["TEMPORAL_WINDOW"] = [str(tw[0]), str(tw[1])] if len(tw) == 2 else None
            else:
                cfg["TEMPORAL_WINDOW"] = None
            cfg["MULTI_PERIOD_FILTER"] = st.checkbox("Exclude problematic periods (MULTI_PERIOD_FILTER)",
                                                     bool(base.get("MULTI_PERIOD_FILTER")))
            if cfg["MULTI_PERIOD_FILTER"]:
                pp = st.data_editor(pd.DataFrame(base.get("PROBLEMATIC_PERIODS") or [], columns=["start", "end"]),
                                    num_rows="dynamic", key="pp",
                                    column_config={"start": st.column_config.TextColumn(help="YYYY-MM-DD"),
                                                   "end": st.column_config.TextColumn(help="YYYY-MM-DD")})
                cfg["PROBLEMATIC_PERIODS"] = [[str(a), str(b)] for a, b in pp.dropna().values.tolist()]
            else:
                cfg["PROBLEMATIC_PERIODS"] = []
        with st.expander("Train / validation / test split", expanded=True):
            strat = st.radio("Strategy (SPLITTING_STRATEGY)", ["random", "chronological"], horizontal=True,
                             format_func=lambda x: {"random": "Random, grouped by Active Region",
                                                    "chronological": "Chronological (test in the future)"}[x])
            cfg["SPLITTING_STRATEGY"] = strat
            cfg["TEST_CUTOFF_DATE"], cfg["TEST_TIME_WINDOW"] = None, None
            if strat == "random":
                tsz = st.slider("TEST_SIZE", 0.05, 0.5, float(base.get("TEST_SIZE", 0.2)), 0.05)
                cfg["TEST_SIZE"] = tsz
                st.caption(f"The package uses StratifiedGroupKFold with {int(1 / tsz)} groups: actual test fraction "
                           f"≈ {1 / int(1 / tsz):.3f}. No AR is shared between test and train/val.")
            else:
                how = st.radio("Test definition", ["default", "cutoff", "window"], horizontal=True,
                               format_func=lambda x: {"default": "last TEST_SIZE of the period",
                                                      "cutoff": "after a date (TEST_CUTOFF_DATE)",
                                                      "window": "in a window (TEST_TIME_WINDOW)"}[x])
                cfg["TEST_SIZE"] = st.slider("TEST_SIZE", 0.05, 0.5, float(base.get("TEST_SIZE", 0.2)), 0.05)
                if how == "cutoff":
                    cfg["TEST_CUTOFF_DATE"] = str(st.date_input("TEST_CUTOFF_DATE", lo + (hi - lo) * 0.8, lo, hi))
                    st.caption("ARs ending before the cutoff go to train/val, those starting after it to test; "
                               "ARs spanning the cutoff are dropped.")
                elif how == "window":
                    w = st.date_input("TEST_TIME_WINDOW", (lo + (hi - lo) * 0.4, lo + (hi - lo) * 0.6), lo, hi, key="ttw")
                    cfg["TEST_TIME_WINDOW"] = [str(w[0]), str(w[1])] if len(w) == 2 else None
            a, b = st.columns(2)
            cfg["N_FOLDS"] = int(a.number_input("N_FOLDS (cross-validation folds)", 2, 10, int(base.get("N_FOLDS", 5))))
            cfg["RANDOM_STATE"] = int(b.number_input("RANDOM_STATE (seed)", 0, 2**31 - 1, int(base.get("RANDOM_STATE", 42))))
            cfg["FORCE_SEPARATION"] = st.checkbox("FORCE_SEPARATION (drop samples overlapping between train/val and test)",
                                                  bool(base.get("FORCE_SEPARATION", True)))
            cfg["SAVE_PLOTS"] = st.checkbox("Save distribution plots (SAVE_PLOTS)", bool(base.get("SAVE_PLOTS", True)))
            st.caption("Class balance: the package does not resample the data; imbalance is handled in "
                       "training (focal/weighted loss for PyTorch models, scale_pos_weight for XGBoost).")
        name = st.text_input("Name", f"Split {strat} from {ds['name']}", key="sp_name")
        desc = st.text_area("Description", "", key="sp_desc", height=70)
        ui.config_summary("filters and split", cfg, {"dataset": ds_id, "LABEL_TYPE": s["label_key"]})
        if ui.show_errors(validate_split(cfg, ds, info)) and st.button("Create and run split", type="primary"):
            ui.create_and_launch(lambda: c.pipeline.create_split(name, ds_id, cfg, desc))
    ui.job_monitor(["split"])

# ======================================================================= list
with t_list:
    kind = st.radio("Type", ["dataset", "split"], horizontal=True,
                    format_func=lambda x: {"dataset": "Dataset HDF5", "split": "Split"}[x])
    if kind == "dataset":
        sel = ui.artifact_list("dataset", lambda m: {
            "samples": m["summary"].get("n_samples"), "features": m["summary"].get("n_features"),
            "AR": m["summary"].get("n_active_regions"), "% YES": round(100 * m["summary"].get("positive_rate", 0), 2),
            "subsample": m["config"].get("subsample")}, key="dsl")
    else:
        sel = ui.artifact_list("split", lambda m: {
            "strategy": m["config"].get("SPLITTING_STRATEGY"), "folds": m["config"].get("N_FOLDS"),
            "samples kept": m["summary"].get("n_samples_kept"), "features": m["summary"].get("n_features")}, key="spl")
    if sel:
        m = c.store.get(sel)
        ui.artifact_status_box(m)
        s = m["summary"]
        if m["status"] == "completed" and m["type"] == "dataset":
            k = st.columns(5)
            k[0].metric("Samples", f"{s['n_samples']:,}")
            k[1].metric("Features", s["n_features"])
            k[2].metric("Active Regions", s["n_active_regions"])
            k[3].metric("Positives", f"{100 * s['positive_rate']:.2f}%")
            k[4].metric("Excluded events (no data)", s.get("n_excluded_events", 0))
            if s.get("n_positive", 1) == 0:
                st.error("This dataset contains **no positive (YES) sample**: no model can be trained on it. "
                         "Extract more events (no subsample or a larger one) or use a more frequent flare class.")
            st.markdown(f"Time range: **{s['time_start']} → {s['time_end']}** · timesteps {s['timesteps']}"
                        f" · label horizon Δ={s.get('label_horizon_hours')}h")
            c1, c2 = st.columns(2)
            cc = pd.DataFrame({"class": list(s["class_counts"]), "n": list(s["class_counts"].values())})
            c1.plotly_chart(px.bar(cc, x="class", y="n", title="Class distribution"), width="stretch")
            nr = pd.DataFrame({"feature": list(s["nan_ratio_per_feature"]),
                               "% NaN": [100 * v for v in s["nan_ratio_per_feature"].values()]})
            c2.plotly_chart(px.bar(nr, x="feature", y="% NaN", title=f"Missing values (on {s['nan_stats_on_samples']} samples)"),
                            width="stretch")
            if s.get("excluded_reasons"):
                st.markdown("**Exclusions during extraction:**")
                st.json(s["excluded_reasons"])
        if m["status"] == "completed" and m["type"] == "split":
            k = st.columns(4)
            k[0].metric("Input samples", f"{s['n_samples_input']:,}")
            k[1].metric("Samples after filters", f"{s['n_samples_kept']:,}")
            k[2].metric("Excluded by filters", f"{s['n_samples_excluded']:,}")
            k[3].metric("Features", s["n_features"])
            rows = [{"split": n, **{kk: v[kk] for kk in ("n_samples", "n_positive", "n_active_regions", "time_start", "time_end")},
                     "% YES": round(100 * v["positive_rate"], 2)} for n, v in s["files"].items()]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            lk = s["leakage"]
            ok = lk["test_vs_trainval_shared_harps"] == 0 and lk["test_vs_trainval_shared_samples"] == 0
            (st.success if ok else st.error)(
                f"Test ↔ train/val separation: shared ARs = {lk['test_vs_trainval_shared_harps']}, "
                f"shared samples = {lk['test_vs_trainval_shared_samples']}. "
                f"Shared train↔val ARs per fold: {lk['train_vs_val_shared_harps_per_fold']}")
            st.caption("Features: " + ", ".join(s["feature_names"]))
            plots = sorted((c.store.path(sel) / "split" / "plots").glob("*.png"))
            for p in plots[:6]:
                st.image(str(p), caption=p.name)
        with st.expander("Configuration"):
            st.json(m["config"])
        with st.expander("Log"):
            ui.log_viewer(sel)
