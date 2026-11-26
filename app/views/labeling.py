import datetime as dt

import pandas as pd
import plotly.express as px
import streamlit as st

from cs_gui import ui
from cs_gui.label_timeline import LabelTimeline
from cs_gui.labeling_form import auto_name, form_errors, labeling_config, parse_threshold, threshold_flux
from cs_gui.pipeline import (AMBIGUOUS_RULES, CALIBRATIONS, FLARE_DEFAULTS, SERIES, label_key, quality_preset,
                             validate_labels)

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


LIMITS = {"W": [0.5, 720.0], "step": [0.5, 720.0], "L": [0.5, 8760.0]}    # hours, as the former number fields
RANGE_MIN = dt.date(2010, 5, 1)                                             # first SHARP data
# widget of the form holding each field of the timeline
TIMELINE_KEYS = {"W": "lab_W", "step": "lab_step", "L": "lab_L", "thr": "lab_thr", "sthr": "lab_sthr",
                 "start": "lab_start", "end": "lab_end"}
DEFAULTS = {"lab_W": 24.0, "lab_step": 24.0, "lab_L_on": False, "lab_L": 24.0, "lab_thr": "M",
            "lab_sthr_on": False, "lab_sthr": "M", "lab_rng": False, "lab_start": RANGE_MIN,
            "lab_end": dt.date.today(), "lab_name_manual": False}


def _flux_or(text: str, fallback: float) -> float:
    f = threshold_flux(parse_threshold(text))
    return f if f is not None else fallback


with tab_new:
    for k, v in DEFAULTS.items():
        st.session_state.setdefault(k, v)
    if st.button("Load the parameters of the former static labels (W = L = Δ = 24 h)"):
        st.session_state.update(lab_W=24.0, lab_step=24.0, lab_L_on=True, lab_L=24.0)
    # values dragged on the timeline go to the fields below before they are drawn
    timeline = LabelTimeline("lab_tl")
    for field, value in timeline.changes().items():
        st.session_state[TIMELINE_KEYS[field]] = value
    ss = st.session_state

    series = st.selectbox("SHARP series", list(SERIES), format_func=SERIES.get, key="lab_series")
    st.markdown("**Windows and thresholds** · flare window W, step Δ of the t_X grid, stratification "
                "window L, prediction threshold S_p and stratification threshold S_s")
    timeline.render(W=ss["lab_W"], step=ss["lab_step"], L_on=ss["lab_L_on"], L=ss["lab_L"],
                    thr=_flux_or(ss["lab_thr"], 1e-5), sthr_on=ss["lab_sthr_on"],
                    sthr=_flux_or(ss["lab_sthr"], 1e-5), rng_on=ss["lab_rng"], start=ss["lab_start"],
                    end=ss["lab_end"], range_min=RANGE_MIN, range_max=dt.date.today(), limits=LIMITS)

    f1, f2, f3 = st.columns(3)
    W = f1.number_input("Flare window W (h)", *LIMITS["W"], step=1.0, key="lab_W")
    step = f2.number_input("Step Δ of the t_X grid (h)", *LIMITS["step"], step=1.0, key="lab_step",
                           help="t_X are aligned to midnight UTC (grid origin 1996-01-01).")
    with f3:
        L_on = st.checkbox("Finite window L (h)", key="lab_L_on",
                           help="Stratification window L. Unchecked: the before/after windows are the whole past and future "
                                "of the HARP.",
                           on_change=lambda: st.session_state.update(lab_L=st.session_state["lab_W"]))
        if L_on:
            L = st.number_input("L (h)", *LIMITS["L"], step=1.0, key="lab_L", label_visibility="collapsed")
        else:
            L = None
            st.text_input("L (h)", "∞ (unlimited)", disabled=True, label_visibility="collapsed")
    t1, t2, _ = st.columns(3)
    thr = parse_threshold(t1.text_input("Prediction threshold S_p", key="lab_thr",
                                        help="GOES class (C, M, X, M5, C2.5…) or flux in W/m² (e.g. 3e-6)."))
    with t2:
        sthr_on = st.checkbox("S_s ≠ S_p", key="lab_sthr_on",
                              help="Stratification threshold S_s. Unchecked: S_s = S_p. GOES class or flux in W/m², as S_p.",
                              on_change=lambda: st.session_state.update(lab_sthr=st.session_state["lab_thr"]))
        s_thr = None
        if sthr_on:
            s_thr = parse_threshold(st.text_input("S_s", key="lab_sthr", label_visibility="collapsed"))
        else:
            st.text_input("S_s", "= S_p", disabled=True, label_visibility="collapsed")
    st.caption("The stratification class is not a target: class of the window maximum (C/M/X…) or NO1–NO4 "
               "(flares ≥ S_s before/after the window), used to stratify the splits. See *Label definition*.")

    rng_on = st.checkbox("Restrict the time range of t_X (default: the whole observation span)", key="lab_rng")
    r2, r3, _ = st.columns(3)
    start = end = None
    if rng_on:
        start = str(r2.date_input("From", key="lab_start"))
        end = str(r3.date_input("To", key="lab_end"))
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
    cfg = labeling_config(series, thr, W, step, s_thr, L, start, end, q, calibration=cal, peak_fallback=peak,
                          ambiguous=amb, position_margin_deg=margin, max_gap_hours=gap)
    # the name follows the parameters until the user edits it
    auto = auto_name(cfg)
    if not ss["lab_name_manual"]:
        ss["lab_name"] = auto
    n1, n2 = st.columns([6, 1], vertical_alignment="bottom")
    name = n1.text_input("Name", key="lab_name",
                         on_change=lambda a: ss.update(lab_name_manual=ss["lab_name"] != a), args=(auto,),
                         help="Follows the parameters until you edit it.")
    if ss["lab_name_manual"]:
        n2.button("↺ auto", help="Back to the automatic name", on_click=lambda: ss.update(lab_name_manual=False))
    desc = st.text_area("Description", "", height=70)
    errs = form_errors(W, step, L, thr, s_thr, start, end)
    errs += validate_labels(cfg, c.settings) if c.settings.has_data() else ["No data in the store: see the Data page"]
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
