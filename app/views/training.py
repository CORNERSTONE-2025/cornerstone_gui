import json

import pandas as pd
import plotly.express as px
import streamlit as st

from cs_gui import ui
from cs_gui.pipeline import validate_training

c = ui.page_header("Training and validation",
                   "Package cross-validation (Cornerstone.train_models) on the split folds, for one or more "
                   "time windows. The job runs in the background.")
info = ui.package_info()
if not info:
    st.stop()

tab_new, tab_res = st.tabs(["New training", "Results"])

with tab_new:
    a, b = st.columns(2)
    with a:
        sp_id = ui.artifact_picker("Filtered and split dataset (split)", "split", key="tr_split")
    with b:
        mc_id = ui.artifact_picker("Model configuration", "model_config", key="tr_mc")
    if sp_id and mc_id:
        sp, mc = c.store.get(sp_id), c.store.get(mc_id)
        s = sp["summary"]
        st.markdown("**Time ranges and data separation** (defined by the split)")
        rows = [{"set": n, "samples": v["n_samples"], "positives": v["n_positive"], "AR": v["n_active_regions"],
                 "from": v["time_start"][:16], "to": v["time_end"][:16]} for n, v in s["files"].items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        strat = sp["config"]["SPLITTING_STRATEGY"]
        lk = s["leakage"]
        st.caption(f"Strategy: **{strat}**"
                   + (f", cutoff {sp['config'].get('TEST_CUTOFF_DATE')}" if sp["config"].get("TEST_CUTOFF_DATE") else "")
                   + (f", test window {sp['config'].get('TEST_TIME_WINDOW')}" if sp["config"].get("TEST_TIME_WINDOW") else "")
                   + f" · inner folds grouped by Active Region (StratifiedGroupKFold) · shared ARs test↔train/val: "
                     f"{lk['test_vs_trainval_shared_harps']}")
        if strat == "random":
            st.info("Random split by Active Region: test and training are not separated in time. For an "
                    "operational evaluation (past data only → future) create a **chronological** split in the Dataset page.")
        st.caption("To change the training/validation/test ranges create a new split (Dataset page): "
                   "the ranges are a property of the split, so the same split is reusable and comparable.")
        st.markdown("**Training parameters**")
        wm_default = info["default_window_map"]
        wm = {w: st_ for w, st_ in wm_default.items() if st_ <= s["timesteps"]}
        x, y, z = st.columns(3)
        windows = x.multiselect("Time windows (history used by the model)", list(wm), list(wm),
                                format_func=lambda w: f"{w} ({wm[w]} steps)")
        n_folds = int(y.number_input("Number of folds to train", 1, int(lk["n_folds"]), int(lk["n_folds"])))
        devices = ["auto", "cpu"] + (["cuda"] if info["cuda_available"] else [])
        device = z.radio("Device", devices, horizontal=True)
        cfg = {"windows": windows, "window_map": {w: wm[w] for w in windows}, "n_folds": n_folds, "device": device}
        tcfg = mc["config"]["training_config"]
        model = mc["config"]["model"]
        if model in info["sk_models"]:
            est = f"C grid = {len(mc['config'].get('regularization_params') or [])} values × {n_folds} folds × {len(windows)} windows"
        else:
            est = f"up to {tcfg['num_epochs']} epochs × {n_folds} folds × {len(windows)} windows (early stopping on {tcfg.get('early_stopping', {}).get('monitor')})"
        st.caption(f"Expected load: {est}.")
        name = st.text_input("Training experiment name", f"{model} on {sp['name']}")
        desc = st.text_area("Description", "", height=70)
        ui.config_summary("training", cfg, {"split": sp_id, "model": f"{model} ({mc_id})", "seed": mc["config"].get("seed")})
        if ui.show_errors(validate_training(cfg, sp, mc, info)) and st.button("Start training", type="primary"):
            ui.create_and_launch(lambda: c.pipeline.create_training(name, sp_id, mc_id, cfg, desc))
    ui.wait_hint()
    ui.job_monitor(["training"])

with tab_res:
    def cols(m):
        rows = [r for r in m["summary"].get("rows", []) if r.get("selected")]
        out = {"model": c.store.get(m["parents"]["model_config"])["config"]["model"]
               if c.store.exists(m["parents"]["model_config"]) else "?",
               "split": m["parents"]["split"], "duration [s]": (m.get("run") or {}).get("duration_s")}
        for r in rows:
            out[f"val TSS {r['window']}"] = round(r.get("val_TSS", float("nan")), 3)
        return out
    sel = ui.artifact_list("training", cols, key="trl")
    if sel:
        m = c.store.get(sel)
        ui.artifact_status_box(m)
        if m["status"] == "completed":
            s = m["summary"]
            st.markdown(f"Model **{s['model']}** ({s['family']}) on **{s['device']}** · windows {s['windows']} · "
                        f"{s['n_folds']} folds · training {s['training_seconds']} s")
            df = pd.DataFrame(s["rows"])
            st.markdown("**Metrics per fold** (val_* = validation of the fold; test_* = split test set, computed by the "
                        "package during training). The *selected* row is the model chosen by the package.")
            st.dataframe(df, hide_index=True, width="stretch")
            metric = st.selectbox("Metric", [c_ for c_ in df.columns if c_.startswith("val_")], index=0)
            st.plotly_chart(px.box(df, x="window", y=metric, points="all", color="window",
                                   title=f"{metric} across folds (CV variability)"), width="stretch")
            for n in s.get("notes", []):
                st.caption(n)
            if s.get("adapters"):
                st.caption("Adapters applied: " + json.dumps(s["adapters"]))
            plots = sorted((c.store.path(sel) / s["task_dir"] / "plots").glob("*.png"))
            if plots:
                with st.expander(f"Package plots ({len(plots)})"):
                    pick = st.multiselect("Plots", [p.name for p in plots], [p.name for p in plots][:4])
                    for p in plots:
                        if p.name in pick:
                            st.image(str(p), caption=p.name)
            with st.expander("Provenance and versions"):
                st.json(m.get("provenance", {}))
        with st.expander("Log"):
            ui.log_viewer(sel)
