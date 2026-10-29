import pandas as pd
import streamlit as st

from cs_gui import ui
from cs_gui.pipeline import validate_inference

c = ui.page_header("Inference on unseen events",
                   "Applies a trained model to data not used for training and selection: saves the per-sample "
                   "predictions and, when the labels are known, the metrics.")

tab_new, tab_res = st.tabs(["New inference", "Results"])

with tab_new:
    tr_id = ui.artifact_picker("Trained model (training)", "training", key="inf_tr")
    if tr_id:
        tr = c.store.get(tr_id)
        split_id = tr["parents"]["split"]
        sp = c.store.get(split_id)
        kind = st.radio("Inference data", ["own_test", "other_split", "dataset"],
                        format_func=lambda x: {"own_test": "Test set of the model split (recommended)",
                                               "other_split": "Another split",
                                               "dataset": "An extracted HDF5 dataset (all samples)"}[x])
        source_file = "test"
        if kind == "own_test":
            src_id = split_id
        elif kind == "other_split":
            src_id = ui.artifact_picker("Split", "split", key="inf_src_sp")
            if src_id:
                files = list(c.store.get(src_id)["summary"]["files"])
                source_file = st.selectbox("File", files, index=files.index("test"))
        else:
            src_id = ui.artifact_picker("Dataset", "dataset", key="inf_src_ds")
        if src_id:
            src = c.store.get(src_id)
            need = sp["summary"]["feature_names"]
            have = src["summary"]["feature_names"]
            miss = [f for f in need if f not in have]
            if miss:
                st.error(f"Incompatible data: the model features {miss} are missing")
            elif have != need:
                st.info(f"The dataset has {len(have)} features: the {len(need)} of the model will be selected and ordered.")
            if src_id != split_id:
                st.warning("Samples whose Active Regions appear in the model train/validation are "
                           "**flagged** in the predictions and **excluded from the metrics** (option below).")
            a, b = st.columns(2)
            windows = a.multiselect("Windows", list(tr["summary"]["windows"]), list(tr["summary"]["windows"]))
            choice = b.radio("Model per window", ["best", "fold"], horizontal=True,
                             format_func=lambda x: {"best": "fold selected by the package", "fold": "specific fold"}[x])
            fold = int(b.number_input("Fold", 0, tr["summary"]["n_folds"] - 1, 0)) if choice == "fold" else None
            st.markdown("**Decision threshold**")
            tm = st.radio("Criterion", ["cv", "fixed"], horizontal=True,
                          format_func=lambda x: {"cv": "Threshold optimized on the CV validation (of the fold used)",
                                                 "fixed": "Fixed threshold"}[x])
            thr = st.slider("Threshold", 0.0, 1.0, 0.5, 0.01) if tm == "fixed" else None
            with st.expander("Event subset (optional)"):
                harps = st.text_input("HARPNUMs to analyse (comma-separated; empty = all)")
                d1, d2 = st.columns(2)
                t0 = d1.text_input("From (issue time, YYYY-MM-DD)", "")
                t1 = d2.text_input("To (YYYY-MM-DD)", "")
            excl = st.checkbox("Exclude from the metrics samples of Active Regions seen in train/validation", True)
            bn = st.select_slider("Bootstrap resamples for the confidence intervals", [0, 200, 500, 1000, 2000], 1000)
            cl = st.checkbox("Bootstrap by Active Region (cluster) — recommended", True)
            try:
                harp_list = [int(x) for x in harps.replace(" ", "").split(",") if x]
            except ValueError:
                harp_list = None
                st.error("Invalid HARPNUMs")
            cfg = {"windows": windows, "model_choice": choice, "fold": fold, "threshold_mode": tm, "threshold": thr,
                   "source_file": source_file, "harp_filter": harp_list or [], "time_start": t0 or None,
                   "time_end": (t1 + "T23:59:59") if t1 else None, "exclude_overlap": excl,
                   "bootstrap": {"n": int(bn), "cluster": cl}}
            name = st.text_input("Name", f"Inference {tr['name']} on {src['name']}")
            ui.config_summary("inference", cfg, {"model": tr_id, "source": f"{src_id} ({src['type']})"})
            errs = validate_inference(cfg, tr, src) + ([f"Missing features: {miss}"] if miss else []) \
                + (["Invalid HARPNUMs"] if harp_list is None else [])
            if ui.show_errors(errs) and st.button("Run inference", type="primary"):
                ui.create_and_launch(lambda: c.pipeline.create_inference(name, tr_id, src_id, cfg))
    ui.job_monitor(["inference"])

with tab_res:
    def cols(m):
        out = {"model": m["summary"].get("model"), "source": m["parents"]["source"],
               "threshold": m["config"].get("threshold_mode")}
        for w, e in (m["summary"].get("windows") or {}).items():
            if e.get("metrics"):
                out[f"TSS {w}"] = round(e["metrics"]["TSS"], 3)
        return out
    sel = ui.artifact_list("inference", cols, key="infl")
    if sel:
        m = c.store.get(sel)
        ui.artifact_status_box(m)
        if m["status"] == "completed":
            rows = []
            for w, e in m["summary"]["windows"].items():
                r = {"window": w, "threshold": round(e["threshold"], 4), "threshold criterion": e["threshold_source"],
                     "predictions": e["n_predictions"], "evaluated": e["n_evaluated"],
                     "excluded (AR in train/val)": e["n_excluded_trainval_harp"], "model file": e["model_file"]}
                if e.get("metrics"):
                    for k in ("TSS", "HSS", "Precision", "Recall", "Specificity", "F1", "ROC_AUC", "PR_AUC"):
                        r[k] = round(e["metrics"][k], 4)
                    ci = e.get("ci", {}).get("TSS")
                    if ci:
                        r["TSS 95% CI"] = f"[{ci['low']:.3f}, {ci['high']:.3f}]"
                else:
                    r["note"] = e.get("note")
                rows.append(r)
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            any_ci = next((e.get("ci") for e in m["summary"]["windows"].values() if e.get("ci")), None)
            if any_ci:
                st.caption("Confidence intervals: " + any_ci["_method"])
            p = c.store.path(sel) / "predictions.csv"
            pred = pd.read_csv(p)
            st.dataframe(pred.head(500), hide_index=True, width="stretch")
            st.download_button("Download predictions.csv", p.read_bytes(), file_name=f"{sel}_predictions.csv")
            st.download_button("Download metrics.json", (c.store.path(sel) / "metrics.json").read_bytes(),
                               file_name=f"{sel}_metrics.json")
        with st.expander("Log"):
            ui.log_viewer(sel)
