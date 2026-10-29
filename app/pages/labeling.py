import pandas as pd
import plotly.express as px
import streamlit as st

from cs_gui import db as dbq
from cs_gui import ui
from cs_gui.pipeline import LABEL_TYPES, TABLE_TYPES, validate_labels

c = ui.page_header("Event labeling",
                   "Labels produced by Cornerstone.splitting_labeling: for every time t_X and Active Region "
                   "(HARP) the GOES flares before and after t_X are counted.")

with st.expander("📖 Label definition (implemented in the package)", expanded=False):
    st.markdown("""
For every t_X (grid with step **Δ**, from 1996-01-01 to 2025-04-30 — range fixed in the package code)
and every HARP observed at t_X, with class threshold *cut* (C: 1e-6, M: 1e-5, X: 1e-4 W/m²):

| window | interval | indicator |
|---|---|---|
| before | [t_X − Δ, t_X] | C_b |
| after (forecast horizon) | [t_X, t_X + Δ] | C_a |
| after-after | [t_X + Δ, t_X + 2Δ] | C_aa |

- **YES**: at least one flare ≥ cut in the *after* window (C_a = 1). `label_u` gives the maximum class (C/M/X).
- **NO1**: no flare in any window · **NO2**: *after-after* only · **NO3**: *before* and *after-after* · **NO4**: *before* only.
- For training/evaluation the package uses the binary **YES vs NO\\*** (NO classes grouped).

Rules for ambiguous cases and exclusion intervals are not package parameters: time windows to
exclude are applied in the *Dataset → filters and split* step.
**CME:** the database contains no CME catalogue, so CME labeling is not available.
""")

tab_new, tab_list = st.tabs(["New labeling", "Produced labels"])

with tab_new:
    existing = dbq.label_tables(c.settings.active_db_path) if not c.settings.problems() else pd.DataFrame()
    st.markdown("**Label tables already in the database** (created earlier with the package)")
    st.dataframe(existing, hide_index=True, width="stretch")

    source = st.radio("Label source", ["existing", "generate"], horizontal=True,
                      index=0 if len(existing) else 1,
                      format_func=lambda x: {"existing": "Snapshot of an existing table (fast)",
                                             "generate": "Generate with splitting_labeling (slow)"}[x])
    col1, col2, col3 = st.columns(3)
    table_type = col1.selectbox("SHARP data type (shtyp)", list(TABLE_TYPES), format_func=TABLE_TYPES.get)
    label_type = col2.selectbox("Minimum flare class (class0)", list(LABEL_TYPES), format_func=LABEL_TYPES.get)
    delta = col3.number_input("Δ: horizon / step in hours (delta_hours)", 1, 240, 24, step=1,
                              help="Length of the before/after/after-after windows and step of the t_X grid.")
    if source == "generate":
        st.info("Generation runs `splitting_labeling` on a **private DB of the artifact** (copying from the main "
                "DB, opened read-only, the GOES events and the first/last time of every HARP). "
                "The shared DB is not modified. It can take tens of minutes.")
    name = st.text_input("Name", f"Label {label_type} {table_type} Δ{delta}h")
    desc = st.text_area("Description", "", height=70)
    cfg = {"source": source, "table_type": table_type, "label_type": label_type, "delta_hours": int(delta)}
    errs = validate_labels(cfg, c.settings) if not c.settings.problems() else ["Incomplete configuration"]
    ui.config_summary("labeling", cfg, {"resulting table": f"DF_{table_type}_{label_type}_delta{delta}_three_classes"})
    if ui.show_errors(errs) and st.button("Create and run labeling", type="primary"):
        ui.create_and_launch(lambda: c.pipeline.create_labels(name, cfg, desc))
    ui.job_monitor(["labels"])

with tab_list:
    sel = ui.artifact_list("labels", lambda m: {
        "class": m["config"]["label_type"], "type": m["config"]["table_type"], "Δh": m["config"]["delta_hours"],
        "source": m["config"]["source"], "rows": m["summary"].get("n_rows"),
        "% YES": round(100 * m["summary"].get("positive_rate", 0), 2)}, key="labels")
    if sel:
        m = c.store.get(sel)
        ui.artifact_status_box(m)
        s = m["summary"]
        if m["status"] == "completed":
            k = st.columns(4)
            k[0].metric("Rows (t_X × HARP)", f"{s['n_rows']:,}")
            k[1].metric("Active Regions", s["n_active_regions"])
            k[2].metric("Positives (YES)", f"{100 * s['positive_rate']:.2f}%")
            k[3].metric("t_X range", f"{s['t_X_start'][:10]} → {s['t_X_end'][:10]}")
            c1, c2 = st.columns(2)
            lc = pd.DataFrame({"label": list(s["label_counts"]), "n": list(s["label_counts"].values())})
            c1.plotly_chart(px.bar(lc, x="label", y="n", title="Label distribution"), width="stretch")
            py = pd.DataFrame([{"year": y, "YES": v["YES"], "total": v["n"]} for y, v in s["per_year"].items()])
            if len(py):
                py["% YES"] = 100 * py["YES"] / py["total"]
                c2.plotly_chart(px.bar(py, x="year", y="% YES", title="Fraction of positives per year"),
                                width="stretch")
            uc = pd.DataFrame({"label_u": list(s["label_u_counts"]), "n": list(s["label_u_counts"].values())})
            st.dataframe(uc, hide_index=True)
            f = c.store.path(sel) / "labels.csv"
            if f.exists() and f.stat().st_size < 200e6:
                st.download_button("Download labels.csv", f.read_bytes(), file_name=f"{sel}_labels.csv")
        with st.expander("Log"):
            ui.log_viewer(sel)
