"""The labeling form: same configuration and name as the former dropdown/number form, timeline sync."""
import itertools
import os

import pytest

from cs_gui.labeling_form import (auto_name, flux_class, form_errors, labeling_config, parse_threshold,
                                  threshold_flux)
from cs_gui.pipeline import quality_preset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ------------------------------------------------- the former form (app/views/labeling.py before the timeline)
def _former_threshold(choice: str, custom: str = "M5"):
    """Value of the former threshold selectbox: "same", a class of the list, or "custom" + text."""
    if choice == "same":
        return None
    if choice == "custom":
        v = custom.strip()
        try:
            return float(v)
        except ValueError:
            return v.upper()
    return choice


def _former_cfg_and_name(series, thr, W, step, s_thr, L, start, end, q, cal, peak, amb, margin, gap):
    cfg = {"series": series, "threshold": thr, "flare_window_hours": float(W), "step_hours": float(step),
           "strat_threshold": s_thr, "strat_window_hours": float(L) if L else None, "start": start, "end": end,
           "quality": q, "flares": {"calibration": cal, "peak_fallback": peak, "ambiguous": amb,
                                    "position_margin_deg": float(margin), "max_gap_hours": float(gap)}}
    sthr_txt = "= S_p" if s_thr is None else s_thr
    name = f"Label {thr} {series} W{W:g}h Δ{step:g}h S_s {sthr_txt}" + (f" L{L:g}h" if L else "")
    return cfg, name


# what the new form gives for the same choice: the text typed in (or set by the timeline) for S_p / S_s
NEW_TEXT = {("C", None): "C", ("M", None): "M", ("X", None): "X", ("custom", "M5"): "M5",
            ("custom", "c2.5"): "c2.5", ("custom", "3e-6"): "3e-6"}


@pytest.mark.parametrize("series,thr_choice,sthr_choice,W,step,L,rng", itertools.islice(itertools.product(
    ["DEF", "NRT"], list(NEW_TEXT), [("same", None), ("C", None), ("custom", "M5")],
    [24.0, 12.0, 0.5], [24.0, 1.0], [None, 24.0, 48.5], [None, ("2012-01-01", "2015-06-30")]), 0, None, 7))
def test_same_config_and_name_as_former_form(series, thr_choice, sthr_choice, W, step, L, rng):
    q = quality_preset("default", series)
    start, end = rng or (None, None)
    flares = ("science", "start", "position_then_flux", 2.0, 24.0)
    thr_old = _former_threshold(thr_choice[0], thr_choice[1] or "")
    sthr_old = _former_threshold(sthr_choice[0], sthr_choice[1] or "")
    cfg_old, name_old = _former_cfg_and_name(series, thr_old, W, step, sthr_old, L, start, end, q, *flares)

    thr_new = parse_threshold(NEW_TEXT[thr_choice])
    sthr_new = None if sthr_choice[0] == "same" else parse_threshold(NEW_TEXT.get(sthr_choice, sthr_choice[0]))
    cfg_new = labeling_config(series, thr_new, W, step, sthr_new, L, start, end, q, *flares)
    assert cfg_new == cfg_old
    assert [type(v) for v in cfg_new.values()] == [type(v) for v in cfg_old.values()]
    assert auto_name(cfg_new) == name_old
    assert form_errors(W, step, L, thr_new, sthr_new, start, end) == []


def test_threshold_classes():
    assert [threshold_flux(v) for v in ("C", "M", "X", "M5", "c2.5", 3e-6)] == \
        pytest.approx([1e-6, 1e-5, 1e-4, 5e-5, 2.5e-6, 3e-6])
    assert threshold_flux("Q1") is None and threshold_flux("") is None and threshold_flux(-1.0) is None
    # the timeline snaps to integer multiples; multiplier 1 gives the class of the former dropdown
    assert [flux_class(f) for f in (1e-6, 1e-5, 1e-4, 5e-5, 2e-6, 9e-5, 1e-3)] == \
        ["C", "M", "X", "M5", "C2", "M9", "X10"]
    assert all(threshold_flux(flux_class(f)) == pytest.approx(f) for f in (1e-7, 3e-6, 4e-5, 2e-4))


def test_form_errors():
    assert form_errors(24, 24, None, "M", None, "2015-01-01", "2012-01-01") == \
        ["Time range of t_X: the start (2015-01-01) must be before the end (2012-01-01)"]
    assert len(form_errors(24, 24, None, "Q", "Z9", None, None)) == 2
    assert form_errors(0.0, 24, -1.0, "M", None, None, None)[0].startswith("W:")


# ------------------------------------------------------------------------------------ the page
def _page():
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(os.path.join(ROOT, "app", "views", "labeling.py"), default_timeout=120)


def test_page_name_follows_parameters():
    at = _page().run()
    assert not at.exception
    assert at.text_input(key="lab_name").value == "Label M DEF W24h Δ24h S_s = S_p"
    at.number_input(key="lab_W").set_value(12.0).run()
    at.text_input(key="lab_thr").set_value("m5").run()
    at.checkbox(key="lab_L_on").check().run()          # L starts from W
    assert at.text_input(key="lab_name").value == "Label M5 DEF W12h Δ24h S_s = S_p L12h"
    # edited by hand: kept when the parameters change, until ↺ auto
    at.text_input(key="lab_name").set_value("my labels").run()
    at.number_input(key="lab_step").set_value(6.0).run()
    assert at.text_input(key="lab_name").value == "my labels"
    at.button[[b.label for b in at.button].index("↺ auto")].click().run()
    assert at.text_input(key="lab_name").value == "Label M5 DEF W12h Δ6h S_s = S_p L12h"
    assert not at.exception


def test_page_applies_timeline_changes():
    import datetime as dt
    at = _page().run()
    at.checkbox(key="lab_L_on").check().run()
    at.checkbox(key="lab_sthr_on").check().run()
    at.checkbox(key="lab_rng").check().run()
    # what the component sends after a drag (only the changed fields)
    at.session_state["lab_tl"] = {"value": {"W": 36, "step": 6, "L": 48, "thr": "X2", "sthr": "C",
                                            "start": "2014-02-03", "end": "2016-01-01", "seq": "a1"}}
    at.run()
    assert not at.exception
    assert at.number_input(key="lab_W").value == 36.0 and at.number_input(key="lab_step").value == 6.0
    assert at.number_input(key="lab_L").value == 48.0
    assert at.text_input(key="lab_thr").value == "X2" and at.text_input(key="lab_sthr").value == "C"
    assert at.date_input(key="lab_start").value == dt.date(2014, 2, 3)
    assert at.text_input(key="lab_name").value == "Label X2 DEF W36h Δ6h S_s C L48h"
    # the same message is applied once: a value typed afterwards is kept
    at.number_input(key="lab_W").set_value(10.0).run()
    assert at.number_input(key="lab_W").value == 10.0
