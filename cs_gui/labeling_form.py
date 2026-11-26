"""Values of the labeling form: GOES thresholds, the labeling configuration and its automatic name.

Pure functions (no Streamlit), shared by the Labeling page and its tests. The configuration is the
contract with the pipeline (``Pipeline.create_labels``): the form may choose the values in any
way, but must build exactly this dictionary.
"""
from __future__ import annotations

import math
import re

# GOES X-ray classes: peak flux (W/m²) of the class with multiplier 1
GOES_CLASS_FLUX = {"A": 1e-8, "B": 1e-7, "C": 1e-6, "M": 1e-5, "X": 1e-4}
_CLASS_RE = re.compile(r"^([ABCMX])(\d+(?:\.\d+)?)?$")


def parse_threshold(text: str) -> str | float:
    """Value of a threshold typed by the user, as the former custom field gave it.

    A number (``3e-6``) is a flux in W/m²; anything else is a GOES class, upper case (``M``,
    ``M5``, ``C2.5``). The pipeline validation tells whether a class is well formed.
    """
    v = str(text).strip()
    try:
        return float(v)
    except ValueError:
        return v.upper()


def threshold_flux(value: str | float | None) -> float | None:
    """Peak flux (W/m²) of a threshold value (class or number), None if it is not valid."""
    if isinstance(value, (int, float)):
        return float(value) if value > 0 else None
    m = _CLASS_RE.match(str(value or "").strip().upper())
    if not m:
        return None
    return GOES_CLASS_FLUX[m.group(1)] * float(m.group(2) or 1)


def flux_class(flux: float) -> str:
    """GOES class of a flux with an integer multiplier: 1e-5 → ``M``, 5e-5 → ``M5``, 1e-3 → ``X10``.

    The multiplier 1 is left out, so that the classes of the former dropdown (C, M, X) come out
    exactly as the dropdown gave them.
    """
    letter = "A"
    for k, base in GOES_CLASS_FLUX.items():   # increasing order: last class whose base ≤ flux
        if flux >= base * (1 - 1e-9):
            letter = k
    mult = max(1, round(flux / GOES_CLASS_FLUX[letter]))
    return letter if mult == 1 else f"{letter}{mult}"


def labeling_config(series: str, threshold, flare_window_hours: float, step_hours: float,
                    strat_threshold, strat_window_hours: float | None, start: str | None, end: str | None,
                    quality: dict, calibration: str, peak_fallback: str, ambiguous: str,
                    position_margin_deg: float, max_gap_hours: float) -> dict:
    """The configuration of a labeling run (the dictionary given to ``Pipeline.create_labels``).

    ``strat_threshold`` None means "same as the prediction threshold"; ``strat_window_hours`` None
    means unlimited before/after windows.
    """
    L = strat_window_hours
    return {"series": series, "threshold": threshold, "flare_window_hours": float(flare_window_hours),
            "step_hours": float(step_hours), "strat_threshold": strat_threshold,
            "strat_window_hours": float(L) if L else None, "start": start, "end": end,
            "quality": quality, "flares": {"calibration": calibration, "peak_fallback": peak_fallback,
                                           "ambiguous": ambiguous,
                                           "position_margin_deg": float(position_margin_deg),
                                           "max_gap_hours": float(max_gap_hours)}}


def auto_name(cfg: dict) -> str:
    """Default name of a labeling run (the naming already used by the page), e.g.
    ``Label M DEF W24h Δ24h S_s = S_p`` or ``Label M5 NRT W12h Δ1h S_s C L48h``."""
    s_thr = cfg["strat_threshold"]
    L = cfg["strat_window_hours"]
    return (f"Label {cfg['threshold']} {cfg['series']} W{cfg['flare_window_hours']:g}h Δ{cfg['step_hours']:g}h "
            f"S_s {'= S_p' if s_thr is None else s_thr}" + (f" L{L:g}h" if L else ""))


def form_errors(W: float, step: float, L: float | None, thr, s_thr, start: str | None, end: str | None) -> list[str]:
    """Messages for values the form must not pass on (checked before the pipeline validation)."""
    errs = []
    if threshold_flux(thr) is None:
        errs.append(f"S_p {thr!r}: a GOES class (C, M5, X1.5…) or a flux in W/m² (e.g. 3e-6)")
    if s_thr is not None and threshold_flux(s_thr) is None:
        errs.append(f"S_s {s_thr!r}: a GOES class (C, M5, X1.5…) or a flux in W/m² (e.g. 3e-6)")
    for name, v in (("W", W), ("Δ", step), ("L", L)):
        if v is not None and (not math.isfinite(v) or v <= 0):
            errs.append(f"{name}: a positive number of hours")
    if start and end and start >= end:
        errs.append(f"Time range of t_X: the start ({start}) must be before the end ({end})")
    return errs


def tx_times(step_hours: float) -> str:
    """When the issue times fall, for a grid step: ``every day at 00:00 UTC``, ``every day at 00:00,
    06:00, 12:00, 18:00 UTC``… (the grid starts at 1996-01-01 00:00 UTC, see the package labeling)."""
    s = float(step_hours)
    if s <= 0:
        return "—"
    if s >= 24:
        days = s / 24
        if days == int(days):
            return "every day at 00:00 UTC" if days == 1 else f"every {int(days)} days at 00:00 UTC"
        return f"every {s:g} h (the time of day changes from one t_X to the next)"
    per_day = 24 / s
    if per_day != int(per_day):
        return f"every {s:g} h (the time of day changes from day to day)"
    times = [f"{int(k * s):02d}:{int(round(k * s % 1 * 60)):02d}" for k in range(int(per_day))]
    if len(times) > 6:
        return f"every {s:g} h from 00:00 UTC ({len(times)} per day)"
    return "every day at " + ", ".join(times) + " UTC"


def auto_description(cfg: dict, quality_preset_name: str) -> str:
    """The parameters of a labeling run that its automatic name leaves out, one per line."""
    q, f = cfg["quality"], cfg["flares"]
    lines = [f"t_X: {tx_times(cfg['step_hours'])}, "
             + (f"from {cfg['start']} to {cfg['end']}" if cfg.get("start") else "whole observation span")]
    if quality_preset_name == "custom":
        set_ = [f"{k}={v}" for k, v in q.items() if v not in (None, [], False)]
        lines.append("Records: custom filter (" + ", ".join(set_) + ")")
    else:
        lines.append(f"Records: preset {quality_preset_name}")
    lines.append(f"Flares: calibration {f['calibration']}, without peak → {f['peak_fallback']}, "
                 f"several HARPs → {f['ambiguous']}, position tolerance {f['position_margin_deg']:g}°, "
                 f"nearest HARP within {f['max_gap_hours']:g} h")
    return "\n".join(lines)
