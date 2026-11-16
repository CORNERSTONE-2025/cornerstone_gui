import pandas as pd
import plotly.express as px
import streamlit as st

from cs_gui import ui
from cs_gui.pipeline import (AMBIGUOUS_RULES, CALIBRATIONS, FLARE_DEFAULTS, SERIES, THRESHOLDS, label_key,
                             quality_preset, validate_labels)

c = ui.page_header("Event labeling",
                   "Parametric labeling (package module cornerstone.labeling): for every issue time t_X and "
                   "Active Region (HARP) observed at t_X, the GOES flares in the flare window decide the label.")

with st.expander("📖 Label definition", expanded=False):
    st.markdown("""
For every **t_X** of a regular grid (step *Δ*) and every HARP observed at t_X, with flare window
**[t_X, t_X + W]**:

**Prediction label** (threshold **S_p**, e.g. M = 1e-5 W/m²) — the target of the models, binary:

- **YES**: at least one flare ≥ S_p peaks in the flare window;
- **NO**: otherwise.

**Stratification class** (threshold **S_s**, same flare window) — only used to stratify the
train/val/test splits, so that every split has the same mix of regions:

- the class of the strongest flare of the window (C/M/X…) when it is ≥ S_s;
- otherwise the history of the region, from the flares ≥ S_s *before* (≤ t_X) and *after* (≥ t_X + W):

| class | before | after |
|---|---|---|
| NO1 | no | no |
| NO2 | no | yes |
| NO3 | yes | yes |
| NO4 | yes | no |

"Before" and "after" are unlimited by default (whole past / whole future of the HARP); with a
**stratification window L** they become [t_X − L, t_X] and [t_X + W, t_X + W + L].
S_s and S_p may coincide or not.

The former static tables (`DF_*_three_classes`) are reproduced with W = L = 24 h, Δ = 24 h,
S_s = S_p. **CME:** no CME catalogue is used.
""")

tab_new, tab_list = st.tabs(["New labeling", "Produced labels"])


def _threshold(label: str, key: str, options: dict, allow_same: bool = False, index: int = 1):
    opts = (["same"] if allow_same else []) + list(options) + ["custom"]
    fmt = {"same": "same as the prediction threshold", "custom": "custom (class or W/m²)", **options}
    choice = st.selectbox(label, opts, index=index, format_func=fmt.get, key=key)
    if choice == "same":
        return None
    if choice == "custom":
        v = st.text_input("Value (e.g. M5, C2.5, 3e-6)", "M5", key=f"{key}_custom").strip()
        try:
            return float(v)
        except ValueError:
            return v.upper()
    return choice


def _default(key: str, value) -> None:
    """Initial value of a widget that a preset button may also set (no value= argument then)."""
    st.session_state.setdefault(key, value)


with tab_new:
    for k, v in {"lab_W": 24.0, "lab_step": 24.0, "lab_L_on": False, "lab_L": 24.0}.items():
        _default(k, v)
    if st.button("Load the parameters of the former static labels (W = L = Δ = 24 h)"):
        for k, v in {"lab_W": 24.0, "lab_step": 24.0, "lab_L_on": True, "lab_L": 24.0}.items():
            st.session_state[k] = v
    a, b = st.columns(2)
    with a:
        st.markdown("**Data and prediction label**")
        series = st.selectbox("SHARP series", list(SERIES), format_func=SERIES.get, key="lab_series")
        thr = _threshold("Prediction threshold S_p", "lab_thr", THRESHOLDS)
        W = st.number_input("Flare window W (hours)", 0.5, 720.0, step=1.0, key="lab_W")
        step = st.number_input("Step of the t_X grid Δ (hours)", 0.5, 720.0, step=1.0, key="lab_step",
                               help="t_X are aligned to midnight UTC (grid origin 1996-01-01).")
    with b:
        st.markdown("**Stratification class**")
        st.caption("Not a target: class of the window maximum (C/M/X…) or NO1–NO4 (flares ≥ S_s before/after "
                   "the window), used to stratify the splits. See *Label definition*.")
        s_thr = _threshold("Stratification threshold S_s", "lab_sthr", THRESHOLDS, allow_same=True, index=0)
        L_on = st.checkbox("Limit the before/after windows (stratification window L)", key="lab_L_on",
                           on_change=lambda: st.session_state.update(lab_L=st.session_state["lab_W"]))
        L = st.number_input("L (hours, default = W)", 0.5, 8760.0, step=1.0, key="lab_L") if L_on else None
        st.markdown("**Time range of t_X**")
        rng_on = st.checkbox("Restrict (default: the whole observation span)", key="lab_rng")
        start = end = None
        if rng_on:
            d1, d2 = st.columns(2)
            start = str(d1.date_input("From", pd.Timestamp("2010-05-01"), key="lab_start"))
            end = str(d2.date_input("To", pd.Timestamp.today(), key="lab_end"))
    with st.expander("Records used (quality filter)"):
        preset = st.radio("Preset", ["default", "legacy_clean", "custom"], horizontal=True, key="lab_qp",
                          format_func={"default": "every record with a patch position (former SHARPC_all)",
                                       "legacy_clean": "former SHARPC_clean selection",
                                       "custom": "custom"}.get)
        q = quality_preset("default" if preset == "custom" else preset, series)
        if preset == "custom":
            q1, q2, q3 = st.columns(3)
            mask = q1.text_input("Drop QUALITY bits (hex, empty = no check)", "", key="lab_qmask")
            q["quality_mask"] = int(mask, 16) if mask.strip() else None
            vals = q1.text_input("Keep only QUALITY values (hex, comma separated)", "", key="lab_qvals")
            q["quality_values"] = [int(v, 16) for v in vals.split(",") if v.strip()] or None
            cm = q2.number_input("CMASK > (0 = no check)", 0.0, 1e6, 0.0, key="lab_cmask")
            q["cmask_gt"] = cm or None
            q["meangbt_gt"] = 0.0 if q2.checkbox("MEANGBT > 0", key="lab_mgbt") else None
            lon = q3.number_input("|longitude| ≤ (0 = no check)", 0.0, 90.0, 0.0, key="lab_lon")
            q["max_abs_lon"] = lon or None
            yrs = q3.text_input("Exclude years (comma separated)", "", key="lab_years")
            q["exclude_years"] = [int(y) for y in yrs.split(",") if y.strip()]
        st.caption("The observation span of every HARP (when it is 'observed at t_X') and the flare → HARP "
                   "association use only these records.")
        st.json(q, expanded=False)
    with st.expander("Flares: calibration and association to the HARPs"):
        f1, f2 = st.columns(2)
        cal = f1.selectbox("GOES calibration", list(CALIBRATIONS), format_func=CALIBRATIONS.get, key="lab_cal")
        amb = f1.selectbox("Several candidate HARPs", list(AMBIGUOUS_RULES), format_func=AMBIGUOUS_RULES.get,
                           key="lab_amb")
        peak = f1.selectbox("Flares without observed peak", ["start", "drop"], key="lab_peak",
                            format_func={"start": "use the start time", "drop": "ignore"}.get)
        gap = f2.number_input("Flare outside the HARP span: nearest HARP within (hours)", 0.0, 720.0,
                              FLARE_DEFAULTS["max_gap_hours"], key="lab_gap")
        margin = f2.number_input("Tolerance on the patch box for the flare position (deg)", 0.0, 30.0,
                                 FLARE_DEFAULTS["position_margin_deg"], key="lab_margin")
    cfg = {"series": series, "threshold": thr, "flare_window_hours": float(W), "step_hours": float(step),
           "strat_threshold": s_thr, "strat_window_hours": float(L) if L else None, "start": start, "end": end,
           "quality": q, "flares": {"calibration": cal, "peak_fallback": peak, "ambiguous": amb,
                                    "position_margin_deg": float(margin), "max_gap_hours": float(gap)}}
    sthr_txt = "= S_p" if s_thr is None else s_thr
    name = st.text_input("Name", f"Label {thr} {series} W{W:g}h Δ{step:g}h S_s {sthr_txt}"
                         + (f" L{L:g}h" if L else ""), key="lab_name")
    desc = st.text_area("Description", "", height=70)
    errs = validate_labels(cfg, c.settings) if c.settings.has_data() else ["No data in the store: see the Data page"]
    ui.config_summary("labeling", cfg, {"HDF5 label name": label_key(cfg) if not errs else "—"})
    if ui.show_errors(errs) and st.button("Create and run labeling", type="primary"):
        ui.create_and_launch(lambda: c.pipeline.create_labels(name, cfg, desc))
    ui.job_monitor(["labels"])

with tab_list:
    sel = ui.artifact_list("labels", lambda m: {
        "series": m["config"].get("series"), "S_p": m["config"].get("threshold"),
        "W h": m["config"].get("flare_window_hours"), "Δ h": m["config"].get("step_hours"),
        "S_s": m["config"].get("strat_threshold") or "= S_p", "rows": m["summary"].get("n_rows"),
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
            c1, c2, c3 = st.columns(3)
            # labels of earlier versions kept NO1..NO4 in the prediction label: shown as NO
            lab_n = pd.Series(s["label_counts"]).groupby(lambda k: "YES" if k == "YES" else "NO").sum()
            lc = pd.DataFrame({"label": lab_n.index, "n": lab_n.values})
            c1.plotly_chart(px.bar(lc, x="label", y="n", title="Prediction label", log_y=True), width="stretch")
            sc = pd.DataFrame({"class": list(s.get("strat_counts", {})), "n": list(s.get("strat_counts", {}).values())})
            c2.plotly_chart(px.bar(sc, x="class", y="n", title="Stratification class", log_y=True), width="stretch")
            py = pd.DataFrame([{"year": y, "YES": v["YES"], "total": v["n"]} for y, v in s["per_year"].items()])
            if len(py):
                py["% YES"] = 100 * py["YES"] / py["total"]
                c3.plotly_chart(px.bar(py, x="year", y="% YES", title="Fraction of positives per year"),
                                width="stretch")
            fa = s.get("flare_association") or {}
            if fa:
                st.markdown(f"**Flares:** {fa.get('events', 0):,} GOES events, {fa.get('with_noaa_region', 0):,} "
                            f"with a NOAA region, {fa.get('associated', 0):,} associated to a HARP "
                            f"({fa.get('ambiguous', 0):,} with several candidates). Rules: {fa.get('rules')}")
            if s.get("sharp_data_until"):
                st.caption(f"SHARP data until {s['sharp_data_until'][:16]} · HDF5 label name `{s.get('label_key')}`")
            d = c.store.path(sel)
            if (d / "labels.csv").exists():
                st.download_button("Download labels.csv", (d / "labels.csv").read_bytes(), file_name=f"{sel}_labels.csv")
            elif (d / "labels.parquet").exists():
                st.download_button("Download labels.parquet", (d / "labels.parquet").read_bytes(),
                                   file_name=f"{sel}_labels.parquet")
        with st.expander("Configuration"):
            st.json(m["config"])
        with st.expander("Log"):
            ui.log_viewer(sel)
