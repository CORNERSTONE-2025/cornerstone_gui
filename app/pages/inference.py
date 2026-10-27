import pandas as pd
import streamlit as st

from cs_gui import ui
from cs_gui.pipeline import validate_inference

c = ui.page_header("Inferenza su eventi non visti",
                   "Applica un modello addestrato a dati non usati per training e selezione: salva le predizioni "
                   "per campione e, se le etichette sono note, le metriche.")

tab_new, tab_res = st.tabs(["Nuova inferenza", "Risultati"])

with tab_new:
    tr_id = ui.artifact_picker("Modello addestrato (training)", "training", key="inf_tr")
    if tr_id:
        tr = c.store.get(tr_id)
        split_id = tr["parents"]["split"]
        sp = c.store.get(split_id)
        kind = st.radio("Dati di inferenza", ["own_test", "other_split", "dataset"],
                        format_func=lambda x: {"own_test": "Test set dello split del modello (consigliato)",
                                               "other_split": "Un altro split",
                                               "dataset": "Un dataset HDF5 estratto (tutti i campioni)"}[x])
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
                st.error(f"Dati incompatibili: mancano le feature del modello {miss}")
            elif have != need:
                st.info(f"Il dataset ha {len(have)} feature: verranno selezionate e ordinate le {len(need)} del modello.")
            if src_id != split_id:
                st.warning("I campioni le cui Active Region compaiono nel train/validation del modello vengono "
                           "**marcati** nelle predizioni ed **esclusi dalle metriche** (opzione qui sotto).")
            a, b = st.columns(2)
            windows = a.multiselect("Finestre", list(tr["summary"]["windows"]), list(tr["summary"]["windows"]))
            choice = b.radio("Modello per finestra", ["best", "fold"], horizontal=True,
                             format_func=lambda x: {"best": "fold selezionato dal pacchetto", "fold": "fold specifico"}[x])
            fold = int(b.number_input("Fold", 0, tr["summary"]["n_folds"] - 1, 0)) if choice == "fold" else None
            st.markdown("**Soglia decisionale**")
            tm = st.radio("Criterio", ["cv", "fixed"], horizontal=True,
                          format_func=lambda x: {"cv": "Soglia ottimizzata sulla validation in CV (del fold usato)",
                                                 "fixed": "Soglia fissa"}[x])
            thr = st.slider("Soglia", 0.0, 1.0, 0.5, 0.01) if tm == "fixed" else None
            with st.expander("Sottoinsieme di eventi (opzionale)"):
                harps = st.text_input("HARPNUM da analizzare (separati da virgola; vuoto = tutti)")
                d1, d2 = st.columns(2)
                t0 = d1.text_input("Da (istante di emissione, YYYY-MM-DD)", "")
                t1 = d2.text_input("A (YYYY-MM-DD)", "")
            excl = st.checkbox("Escludi dalle metriche i campioni di Active Region viste in train/validation", True)
            bn = st.select_slider("Ricampionamenti bootstrap per gli intervalli di confidenza", [0, 200, 500, 1000, 2000], 1000)
            cl = st.checkbox("Bootstrap per Active Region (cluster) — consigliato", True)
            try:
                harp_list = [int(x) for x in harps.replace(" ", "").split(",") if x]
            except ValueError:
                harp_list = None
                st.error("HARPNUM non validi")
            cfg = {"windows": windows, "model_choice": choice, "fold": fold, "threshold_mode": tm, "threshold": thr,
                   "source_file": source_file, "harp_filter": harp_list or [], "time_start": t0 or None,
                   "time_end": (t1 + "T23:59:59") if t1 else None, "exclude_overlap": excl,
                   "bootstrap": {"n": int(bn), "cluster": cl}}
            name = st.text_input("Nome", f"Inferenza {tr['name']} su {src['name']}")
            ui.config_summary("inferenza", cfg, {"modello": tr_id, "sorgente": f"{src_id} ({src['type']})"})
            errs = validate_inference(cfg, tr, src) + ([f"Feature mancanti: {miss}"] if miss else []) \
                + (["HARPNUM non validi"] if harp_list is None else [])
            if ui.show_errors(errs) and st.button("Esegui inferenza", type="primary"):
                ui.create_and_launch(lambda: c.pipeline.create_inference(name, tr_id, src_id, cfg))
    ui.job_monitor(["inference"])

with tab_res:
    def cols(m):
        out = {"modello": m["summary"].get("model"), "sorgente": m["parents"]["source"],
               "soglia": m["config"].get("threshold_mode")}
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
                r = {"finestra": w, "soglia": round(e["threshold"], 4), "criterio soglia": e["threshold_source"],
                     "predizioni": e["n_predictions"], "valutati": e["n_evaluated"],
                     "esclusi (AR in train/val)": e["n_excluded_trainval_harp"], "file modello": e["model_file"]}
                if e.get("metrics"):
                    for k in ("TSS", "HSS", "Precision", "Recall", "Specificity", "F1", "ROC_AUC", "PR_AUC"):
                        r[k] = round(e["metrics"][k], 4)
                    ci = e.get("ci", {}).get("TSS")
                    if ci:
                        r["TSS 95% CI"] = f"[{ci['low']:.3f}, {ci['high']:.3f}]"
                else:
                    r["nota"] = e.get("note")
                rows.append(r)
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            any_ci = next((e.get("ci") for e in m["summary"]["windows"].values() if e.get("ci")), None)
            if any_ci:
                st.caption("Intervalli di confidenza: " + any_ci["_method"])
            p = c.store.path(sel) / "predictions.csv"
            pred = pd.read_csv(p)
            st.dataframe(pred.head(500), hide_index=True, width="stretch")
            st.download_button("Scarica predictions.csv", p.read_bytes(), file_name=f"{sel}_predictions.csv")
            st.download_button("Scarica metrics.json", (c.store.path(sel) / "metrics.json").read_bytes(),
                               file_name=f"{sel}_metrics.json")
        with st.expander("Log"):
            ui.log_viewer(sel)
