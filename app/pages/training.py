import json

import pandas as pd
import plotly.express as px
import streamlit as st

from cs_gui import ui
from cs_gui.pipeline import validate_training

c = ui.page_header("Training e validazione",
                   "Cross-validation del pacchetto (Cornerstone.train_models) sui fold dello split, per una o più "
                   "finestre temporali. Il job gira in background.")
info = ui.package_info()
if not info:
    st.stop()

tab_new, tab_res = st.tabs(["Nuovo training", "Risultati"])

with tab_new:
    a, b = st.columns(2)
    with a:
        sp_id = ui.artifact_picker("Dataset filtrato e suddiviso (split)", "split", key="tr_split")
    with b:
        mc_id = ui.artifact_picker("Configurazione del modello", "model_config", key="tr_mc")
    if sp_id and mc_id:
        sp, mc = c.store.get(sp_id), c.store.get(mc_id)
        s = sp["summary"]
        st.markdown("**Intervalli temporali e separazione dei dati** (definiti dallo split)")
        rows = [{"insieme": n, "campioni": v["n_samples"], "positivi": v["n_positive"], "AR": v["n_active_regions"],
                 "da": v["time_start"][:16], "a": v["time_end"][:16]} for n, v in s["files"].items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        strat = sp["config"]["SPLITTING_STRATEGY"]
        lk = s["leakage"]
        st.caption(f"Strategia: **{strat}**"
                   + (f", cutoff {sp['config'].get('TEST_CUTOFF_DATE')}" if sp["config"].get("TEST_CUTOFF_DATE") else "")
                   + (f", finestra test {sp['config'].get('TEST_TIME_WINDOW')}" if sp["config"].get("TEST_TIME_WINDOW") else "")
                   + f" · fold interni raggruppati per Active Region (StratifiedGroupKFold) · AR condivise test↔train/val: "
                     f"{lk['test_vs_trainval_shared_harps']}")
        if strat == "random":
            st.info("Split casuale per Active Region: test e training non sono separati nel tempo. Per una "
                    "valutazione operativa (solo dati passati → futuro) crea uno split **cronologico** nella pagina Dataset.")
        st.caption("Per cambiare gli intervalli di training/validazione/test crea un nuovo split (pagina Dataset): "
                   "gli intervalli sono una proprietà dello split, così lo stesso split è riutilizzabile e confrontabile.")
        st.markdown("**Parametri del training**")
        wm_default = info["default_window_map"]
        wm = {w: st_ for w, st_ in wm_default.items() if st_ <= s["timesteps"]}
        x, y, z = st.columns(3)
        windows = x.multiselect("Finestre temporali (storia usata dal modello)", list(wm), list(wm),
                                format_func=lambda w: f"{w} ({wm[w]} step)")
        n_folds = int(y.number_input("Numero di fold da addestrare", 1, int(lk["n_folds"]), int(lk["n_folds"])))
        devices = ["auto", "cpu"] + (["cuda"] if info["cuda_available"] else [])
        device = z.radio("Dispositivo", devices, horizontal=True)
        cfg = {"windows": windows, "window_map": {w: wm[w] for w in windows}, "n_folds": n_folds, "device": device}
        tcfg = mc["config"]["training_config"]
        model = mc["config"]["model"]
        if model in info["sk_models"]:
            est = f"griglia C = {len(mc['config'].get('regularization_params') or [])} valori × {n_folds} fold × {len(windows)} finestre"
        else:
            est = f"fino a {tcfg['num_epochs']} epoche × {n_folds} fold × {len(windows)} finestre (early stopping su {tcfg.get('early_stopping', {}).get('monitor')})"
        st.caption(f"Carico previsto: {est}.")
        name = st.text_input("Nome dell'esperimento di training", f"{model} su {sp['name']}")
        desc = st.text_area("Descrizione", "", height=70)
        ui.config_summary("training", cfg, {"split": sp_id, "modello": f"{model} ({mc_id})", "seed": mc["config"].get("seed")})
        if ui.show_errors(validate_training(cfg, sp, mc, info)) and st.button("Avvia training", type="primary"):
            ui.create_and_launch(lambda: c.pipeline.create_training(name, sp_id, mc_id, cfg, desc))
    ui.wait_hint()
    ui.job_monitor(["training"])

with tab_res:
    def cols(m):
        rows = [r for r in m["summary"].get("rows", []) if r.get("selected")]
        out = {"modello": c.store.get(m["parents"]["model_config"])["config"]["model"]
               if c.store.exists(m["parents"]["model_config"]) else "?",
               "split": m["parents"]["split"], "durata [s]": (m.get("run") or {}).get("duration_s")}
        for r in rows:
            out[f"val TSS {r['window']}"] = round(r.get("val_TSS", float("nan")), 3)
        return out
    sel = ui.artifact_list("training", cols, key="trl")
    if sel:
        m = c.store.get(sel)
        ui.artifact_status_box(m)
        if m["status"] == "completed":
            s = m["summary"]
            st.markdown(f"Modello **{s['model']}** ({s['family']}) su **{s['device']}** · finestre {s['windows']} · "
                        f"{s['n_folds']} fold · training {s['training_seconds']} s")
            df = pd.DataFrame(s["rows"])
            st.markdown("**Metriche per fold** (val_* = validation del fold; test_* = test set dello split, calcolato dal "
                        "pacchetto durante il training). La riga *selected* è il modello scelto dal pacchetto.")
            st.dataframe(df, hide_index=True, width="stretch")
            metric = st.selectbox("Metrica", [c_ for c_ in df.columns if c_.startswith("val_")], index=0)
            st.plotly_chart(px.box(df, x="window", y=metric, points="all", color="window",
                                   title=f"{metric} tra i fold (variabilità della CV)"), width="stretch")
            for n in s.get("notes", []):
                st.caption(n)
            if s.get("adapters"):
                st.caption("Adattatori applicati: " + json.dumps(s["adapters"]))
            plots = sorted((c.store.path(sel) / s["task_dir"] / "plots").glob("*.png"))
            if plots:
                with st.expander(f"Grafici del pacchetto ({len(plots)})"):
                    pick = st.multiselect("Grafici", [p.name for p in plots], [p.name for p in plots][:4])
                    for p in plots:
                        if p.name in pick:
                            st.image(str(p), caption=p.name)
            with st.expander("Provenienza e versioni"):
                st.json(m.get("provenance", {}))
        with st.expander("Log"):
            ui.log_viewer(sel)
