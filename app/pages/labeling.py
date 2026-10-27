import pandas as pd
import plotly.express as px
import streamlit as st

from cs_gui import db as dbq
from cs_gui import ui
from cs_gui.pipeline import LABEL_TYPES, TABLE_TYPES, validate_labels

c = ui.page_header("Labeling degli eventi",
                   "Etichette prodotte da Cornerstone.splitting_labeling: per ogni istante t_X e Active Region "
                   "(HARP) si contano i flare GOES prima e dopo t_X.")

with st.expander("📖 Definizione delle label (implementata nel pacchetto)", expanded=False):
    st.markdown("""
Per ogni t_X (griglia con passo **Δ**, dal 1996-01-01 al 2025-04-30 — intervallo fissato nel codice del pacchetto)
e per ogni HARP osservata in t_X, con soglia di classe *cut* (C: 1e-6, M: 1e-5, X: 1e-4 W/m²):

| finestra | intervallo | indicatore |
|---|---|---|
| prima | [t_X − Δ, t_X] | C_b |
| dopo (orizzonte di previsione) | [t_X, t_X + Δ] | C_a |
| dopo-dopo | [t_X + Δ, t_X + 2Δ] | C_aa |

- **YES**: almeno un flare ≥ cut nella finestra *dopo* (C_a = 1). `label_u` riporta la classe massima (C/M/X).
- **NO1**: nessun flare in nessuna finestra · **NO2**: solo *dopo-dopo* · **NO3**: *prima* e *dopo-dopo* · **NO4**: solo *prima*.
- Per training/valutazione il pacchetto usa il binario **YES vs NO\\*** (classi NO raggruppate).

Le regole per i casi ambigui e gli intervalli di esclusione non sono parametri del pacchetto: le finestre
temporali da escludere si applicano nel passo *Dataset → filtri e split*.
**CME:** il database non contiene cataloghi CME, quindi il labeling CME non è disponibile.
""")

tab_new, tab_list = st.tabs(["Nuovo labeling", "Label prodotte"])

with tab_new:
    existing = dbq.label_tables(c.settings.active_db_path) if not c.settings.problems() else pd.DataFrame()
    st.markdown("**Tabelle di label già presenti nel database** (create in precedenza con il pacchetto)")
    st.dataframe(existing, hide_index=True, width="stretch")

    source = st.radio("Sorgente delle label", ["existing", "generate"], horizontal=True,
                      index=0 if len(existing) else 1,
                      format_func=lambda x: {"existing": "Snapshot di una tabella esistente (veloce)",
                                             "generate": "Genera con splitting_labeling (lento)"}[x])
    col1, col2, col3 = st.columns(3)
    table_type = col1.selectbox("Tipo di dati SHARP (shtyp)", list(TABLE_TYPES), format_func=TABLE_TYPES.get)
    label_type = col2.selectbox("Classe minima del flare (class0)", list(LABEL_TYPES), format_func=LABEL_TYPES.get)
    delta = col3.number_input("Δ: orizzonte / passo in ore (delta_hours)", 1, 240, 24, step=1,
                              help="Lunghezza delle finestre prima/dopo/dopo-dopo e passo della griglia di t_X.")
    if source == "generate":
        st.info("La generazione esegue `splitting_labeling` su un **DB privato dell'artefatto** (copiando dal DB "
                "principale, aperto in sola lettura, gli eventi GOES e il primo/ultimo istante di ogni HARP). "
                "Il DB condiviso non viene modificato. Può richiedere decine di minuti.")
    name = st.text_input("Nome", f"Label {label_type} {table_type} Δ{delta}h")
    desc = st.text_area("Descrizione", "", height=70)
    cfg = {"source": source, "table_type": table_type, "label_type": label_type, "delta_hours": int(delta)}
    errs = validate_labels(cfg, c.settings) if not c.settings.problems() else ["Configurazione incompleta"]
    ui.config_summary("labeling", cfg, {"tabella risultante": f"DF_{table_type}_{label_type}_delta{delta}_three_classes"})
    if ui.show_errors(errs) and st.button("Crea ed esegui labeling", type="primary"):
        ui.create_and_launch(lambda: c.pipeline.create_labels(name, cfg, desc))
    ui.job_monitor(["labels"])

with tab_list:
    sel = ui.artifact_list("labels", lambda m: {
        "classe": m["config"]["label_type"], "tipo": m["config"]["table_type"], "Δh": m["config"]["delta_hours"],
        "sorgente": m["config"]["source"], "righe": m["summary"].get("n_rows"),
        "% YES": round(100 * m["summary"].get("positive_rate", 0), 2)}, key="labels")
    if sel:
        m = c.store.get(sel)
        ui.artifact_status_box(m)
        s = m["summary"]
        if m["status"] == "completed":
            k = st.columns(4)
            k[0].metric("Righe (t_X × HARP)", f"{s['n_rows']:,}")
            k[1].metric("Active Regions", s["n_active_regions"])
            k[2].metric("Positivi (YES)", f"{100 * s['positive_rate']:.2f}%")
            k[3].metric("Intervallo t_X", f"{s['t_X_start'][:10]} → {s['t_X_end'][:10]}")
            c1, c2 = st.columns(2)
            lc = pd.DataFrame({"label": list(s["label_counts"]), "n": list(s["label_counts"].values())})
            c1.plotly_chart(px.bar(lc, x="label", y="n", title="Distribuzione delle label"), width="stretch")
            py = pd.DataFrame([{"anno": y, "YES": v["YES"], "totale": v["n"]} for y, v in s["per_year"].items()])
            if len(py):
                py["% YES"] = 100 * py["YES"] / py["totale"]
                c2.plotly_chart(px.bar(py, x="anno", y="% YES", title="Frazione di positivi per anno"),
                                width="stretch")
            uc = pd.DataFrame({"label_u": list(s["label_u_counts"]), "n": list(s["label_u_counts"].values())})
            st.dataframe(uc, hide_index=True)
            f = c.store.path(sel) / "labels.csv"
            if f.exists() and f.stat().st_size < 200e6:
                st.download_button("Scarica labels.csv", f.read_bytes(), file_name=f"{sel}_labels.csv")
        with st.expander("Log"):
            ui.log_viewer(sel)
