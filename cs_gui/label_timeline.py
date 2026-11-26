"""Graphical editor of the labeling windows (Streamlit component, JavaScript in ``components/``).

One timeline around an issue time t_X: flare window W, step Δ of the t_X grid, stratification
window L (or unlimited), thresholds S_p / S_s on a log flux axis; optionally a second row with the
time range of t_X. Usage, before the widgets that hold the same values are created::

    tl = LabelTimeline("lab_tl")
    for field, value in tl.changes().items(): ...   # values dragged since the last run
    tl.render(values)                                # values = the current ones (dict, see render)

The component sends only the fields the user changed (hours as numbers, thresholds as GOES
classes, dates as ISO days); the page applies them to its own widgets, so typed and dragged
values stay in sync.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import streamlit as st

_ASSETS = Path(__file__).parent / "components"
FIELDS = ("W", "step", "L", "thr", "sthr", "start", "end")


def _component():
    # registered at every use: registering the same definition again is a no-op, and a cached
    # handle would point to the registry of a runtime that no longer exists (tests, restarts)
    return st.components.v2.component(
        "cs_label_timeline",
        js=(_ASSETS / "label_timeline.js").read_text(),
        css=(_ASSETS / "label_timeline.css").read_text(),
    )


class LabelTimeline:
    def __init__(self, key: str):
        self.key = key
        self._seq_key = f"_{key}_seq"

    def changes(self) -> dict:
        """Fields changed in the widget since the last call (empty if none)."""
        v = (st.session_state.get(self.key) or {}).get("value")
        if not v or v.get("seq") == st.session_state.get(self._seq_key):
            return {}
        st.session_state[self._seq_key] = v["seq"]
        out = {k: v[k] for k in FIELDS if k in v}
        for k in ("W", "step", "L"):       # JSON may give 24 for 24.0: the number fields want floats
            if k in out:
                out[k] = float(out[k])
        for k in ("start", "end"):
            if k in out:
                out[k] = dt.date.fromisoformat(out[k])
        return out

    def render(self, *, W: float, step: float, L_on: bool, L: float, thr: float, sthr_on: bool, sthr: float,
               rng_on: bool, start: dt.date, end: dt.date, range_min: dt.date, range_max: dt.date,
               limits: dict) -> None:
        """Draw the timeline with these values (thresholds as fluxes in W/m²; limits: {"W": [lo, hi],
        "step": [lo, hi], "L": [lo, hi]} in hours)."""
        data = {"W": float(W), "step": float(step), "L_on": bool(L_on), "L": float(L), "thr": float(thr),
                "sthr_on": bool(sthr_on), "sthr": float(sthr), "rng_on": bool(rng_on),
                "start": start.isoformat(), "end": end.isoformat(),
                "range_min": min(range_min, start).isoformat(), "range_max": max(range_max, end).isoformat(),
                "limits": limits, "seq": st.session_state.get(self._seq_key)}
        _component()(key=self.key, data=data, default={"value": None}, on_value_change=lambda: None)
