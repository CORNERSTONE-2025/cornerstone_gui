import copy
import json

import streamlit as st

from cs_gui import ui
from cs_gui.pipeline import (CALIBRATION_TYPES, DEFAULT_TRAINING_CONFIG, LOSS_TYPES, MONITORS, SCHEDULERS,
                             THRESHOLD_METRICS, validate_model_config)

c = ui.page_header("Model configuration",
                   "Algorithms and hyperparameters read from the package (MODEL_CONFIGS, REGULARIZATION_PARAMS). "
                   "Every saved configuration is an immutable, versionable artifact.")
info = ui.package_info()
if not info:
    st.stop()

KNOWN_CHOICES = {"pooling": ["mask_mean", "last"], "embed": ["fixed", "learned", "timeF"],
                 "activation": ["relu", "gelu"], "temporal_encoding": ["none", "sinusoidal", "time_delta"],
                 "freq": ["h", "t", "s", "d"], "task_name": ["classification"]}
FIXED = {"model_name", "task_name", "num_class", "c_out"}   # identity of the architecture: not editable

tab_new, tab_list = st.tabs(["New configuration", "Saved configurations"])

with tab_new:
    base_id = ui.artifact_picker("Start from an existing configuration (duplicate and edit)", "model_config",
                                 key="mc_base", allow_none=True)
    base = c.store.get(base_id)["config"] if base_id else None
    models = info["trainable_models"]
    model = st.selectbox("Algorithm", models, index=models.index(base["model"]) if base else models.index("LSTM"),
                         format_func=lambda m: f"{m}  ({'sklearn' if m in info['sk_models'] else 'PyTorch'})", key=f"mc_model_{base_id}")
    is_sk = model in info["sk_models"]
    k = f"{base_id}_{model}"           # widget keys reset when base/model change
    defaults = info["model_configs"].get(model, {})
    current = {**defaults, **((base or {}).get("model_params", {}) if base and base["model"] == model else {})}

    st.subheader("Model hyperparameters")
    params = {}
    cols = st.columns(3)
    for i, (p, d) in enumerate(defaults.items()):
        v = current.get(p, d)
        col = cols[i % 3]
        if p in FIXED:
            col.text_input(p, str(v), disabled=True, key=f"{k}_{p}", help="Fixed for this architecture")
        elif d is None:
            col.text_input(p, "auto (dataset size)", disabled=True, key=f"{k}_{p}",
                           help="Set by the package from the number of features of the split")
        elif isinstance(d, bool):
            params[p] = col.checkbox(p, bool(v), key=f"{k}_{p}")
        elif isinstance(d, int):
            params[p] = int(col.number_input(p, value=int(v), step=1, key=f"{k}_{p}"))
        elif isinstance(d, float):
            params[p] = float(col.number_input(p, value=float(v), format="%.6g", key=f"{k}_{p}"))
        elif p in KNOWN_CHOICES:
            ch = KNOWN_CHOICES[p] if v in KNOWN_CHOICES[p] else KNOWN_CHOICES[p] + [v]
            params[p] = col.selectbox(p, ch, index=ch.index(v), key=f"{k}_{p}")
        else:
            params[p] = col.text_input(p, str(v), key=f"{k}_{p}")
    if "seq_len" in defaults:
        st.caption("seq_len is replaced by the package with the length of the time window during training.")
    if not defaults:
        st.caption("No hyperparameter exposed by MODEL_CONFIGS for this model.")

    tc = copy.deepcopy((base or {}).get("training_config") or info.get("default_training_config") or DEFAULT_TRAINING_CONFIG)
    for kk, vv in DEFAULT_TRAINING_CONFIG.items():
        tc.setdefault(kk, copy.deepcopy(vv))
    st.subheader("Training and validation")
    if is_sk:
        st.caption(f"For {model} (sklearn) the package uses: regularization grid C explored in CV, "
                   "threshold optimization on validation. Optimizer, loss and early stopping only apply to "
                   "PyTorch models.")
        reg_default = (base or {}).get("regularization_params") or info["regularization_params"].get(model, {}).get(
            "regularization_params", [])
        reg_txt = st.text_input("Regularization grid (regularization_params, comma-separated values)",
                                ", ".join(f"{x:g}" for x in reg_default), key=f"{k}_reg",
                                help="XGBoost: reg_lambda · ROCKET/MINIROCKET: C of the linear classifier")
        try:
            reg = [float(x) for x in reg_txt.split(",") if x.strip()]
        except ValueError:
            reg = []
            st.error("Invalid grid: use comma-separated numbers")
    else:
        reg = None
        a, b, d = st.columns(3)
        tc["batch_size"] = int(a.number_input("batch_size", 1, 65536, int(tc["batch_size"]), key=f"{k}_bs"))
        tc["learning_rate"] = float(b.number_input("learning_rate", 1e-9, 10.0, float(tc["learning_rate"]), format="%.2e", key=f"{k}_lr"))
        tc["weight_decay"] = float(d.number_input("weight_decay", 0.0, 10.0, float(tc["weight_decay"]), format="%.2e", key=f"{k}_wd"))
        tc["num_epochs"] = int(a.number_input("num_epochs", 1, 10000, int(tc["num_epochs"]), key=f"{k}_ep"))
        tc["patience"] = int(b.number_input("patience", 1, 10000, int(tc["patience"]), key=f"{k}_pat"))
        with st.expander("Learning rate scheduler"):
            stype = st.selectbox("lr_scheduler.type", SCHEDULERS,
                                 index=SCHEDULERS.index((tc.get("lr_scheduler") or {}).get("type")), key=f"{k}_sch")
            if stype == "CosineAnnealingLR":
                tc["lr_scheduler"] = {"type": stype,
                                      "T_max": int(st.number_input("T_max", 1, 100000, int(tc["lr_scheduler"].get("T_max", tc["num_epochs"])), key=f"{k}_tmax")),
                                      "eta_min": float(st.number_input("eta_min", 0.0, 1.0, float(tc["lr_scheduler"].get("eta_min", 1e-6)), format="%.1e", key=f"{k}_eta"))}
            elif stype == "ReduceLROnPlateau":
                tc["lr_scheduler"] = {"type": stype, "mode": st.selectbox("mode", ["min", "max"], key=f"{k}_rmode"),
                                      "factor": float(st.number_input("factor", 0.01, 0.99, 0.5, key=f"{k}_rf")),
                                      "patience": int(st.number_input("patience scheduler", 1, 1000, 5, key=f"{k}_rp")),
                                      "min_lr": float(st.number_input("min_lr", 0.0, 1.0, 1e-6, format="%.1e", key=f"{k}_rml"))}
            else:
                tc["lr_scheduler"] = None
        with st.expander("Loss and class imbalance"):
            lf = dict(tc.get("loss_function") or {"type": "bce"})
            lf["type"] = st.selectbox("loss_function.type", LOSS_TYPES, index=LOSS_TYPES.index(lf.get("type", "bce")),
                                      key=f"{k}_loss", help="bce uses pos_weight = n_neg/n_pos computed by the package; "
                                                            "focal/asymmetric_focal down-weight easy examples.")
            if lf["type"] in ("focal", "asymmetric_focal", "curriculum"):
                sub = dict(lf.get(lf["type"]) or {})
                for pk, pv in {"focal": {"alpha": 0.96, "gamma": 2.0}, "asymmetric_focal": {"alpha": 0.96, "gamma_pos": 5.0, "gamma_neg": 2.0},
                               "curriculum": {"alpha": 0.5, "gamma": 2.0, "warmup_epochs": 5, "transition_epochs": 10}}[lf["type"]].items():
                    val = sub.get(pk, pv)
                    sub[pk] = (int(st.number_input(pk, 0, 10000, int(val), key=f"{k}_l_{pk}")) if isinstance(pv, int)
                               else float(st.number_input(pk, 0.0, 100.0, float(val), key=f"{k}_l_{pk}")))
                sub.setdefault("eps", 1e-6)
                lf[lf["type"]] = sub
            tc["loss_function"] = lf
        with st.expander("Early stopping"):
            es = dict(tc.get("early_stopping") or {})
            es["monitor"] = st.selectbox("monitor", MONITORS, index=MONITORS.index(str(es.get("monitor", "tss")).lower()),
                                         key=f"{k}_esm")
            es["mode"] = st.selectbox("mode", ["max", "min"], index=0 if es.get("mode", "max") == "max" else 1, key=f"{k}_esmode")
            es["min_delta"] = float(st.number_input("min_delta", 0.0, 1.0, float(es.get("min_delta", 1e-3)), format="%.1e", key=f"{k}_esd"))
            es["patience"] = int(st.number_input("patience (early stopping)", 1, 10000, int(es.get("patience", tc["patience"])), key=f"{k}_esp"))
            tc["early_stopping"] = es
        with st.expander("Probability calibration"):
            cal = dict(tc.get("calibration") or {})
            cal["enabled"] = st.checkbox("Post-hoc calibration on validation", bool(cal.get("enabled")), key=f"{k}_cal")
            ctype = cal.get("type") or cal.get("kind") or "platt"
            cal["type"] = st.selectbox("calibration.type", CALIBRATION_TYPES, index=CALIBRATION_TYPES.index(ctype)
                                       if ctype in CALIBRATION_TYPES else 0, key=f"{k}_calt",
                                       help="The package reads the 'type' key (the 'kind' key is ignored).")
            cal.pop("kind", None)
            st.caption("Warning: with barely trained models the package can stop on the AUC invariance "
                       "check after Platt calibration.")
            tc["calibration"] = cal
    with st.expander("Decision threshold optimization (on validation)"):
        th = dict(tc.get("threshold_optimization") or {})
        th["enabled"] = st.checkbox("Enabled", bool(th.get("enabled", True)), key=f"{k}_th")
        th["metric"] = st.selectbox("Metric to maximize", THRESHOLD_METRICS,
                                    index=THRESHOLD_METRICS.index(th.get("metric", "TSS")), key=f"{k}_thm")
        th["n_thresholds"] = int(st.number_input("n_thresholds", 2, 10000, int(th.get("n_thresholds", 50)), key=f"{k}_thn"))
        if th["metric"] == "precision_at_recall":
            th["min_recall"] = float(st.slider("min_recall", 0.0, 1.0, float(th.get("min_recall", 0.6)), key=f"{k}_thr"))
        if th["metric"] == "f1_at_precision":
            th["min_precision"] = float(st.slider("min_precision", 0.0, 1.0, float(th.get("min_precision", 0.3)), key=f"{k}_thp"))
        th.setdefault("threshold_range", [0.0, 1.0])
        tc["threshold_optimization"] = th
    seed = int(st.number_input("Random seed (python, numpy, torch)", 0, 2**32 - 1, int((base or {}).get("seed", 42)), key=f"{k}_seed"))
    st.caption("The number of folds and the time windows are chosen at training time (they depend on the split).")

    cfg = {"model": model, "model_params": params, "training_config": tc, "regularization_params": reg, "seed": seed}
    name = st.text_input("Name", f"{model} " + ("(variant)" if base else "base"), key=f"{k}_name")
    desc = st.text_area("Description", "", key=f"{k}_desc", height=70)
    changed = {p: v for p, v in params.items() if defaults.get(p) != v}
    ui.config_summary("model", {"model": model, "seed": seed, **({"regularization_params": reg} if is_sk else {})},
                      {"hyperparameters changed from the package defaults": changed or "none"})
    with st.expander("Full configuration (JSON)"):
        st.json(cfg)
    if ui.show_errors(validate_model_config(cfg, info)) and st.button("Save configuration", type="primary"):
        m = ui.create_and_launch(lambda: c.pipeline.create_model_config(name, cfg, desc,
                                                                        derived_from=base_id), launch=False)
        if m:
            st.success(f"Configuration saved: {m['id']}")

with tab_list:
    sel = ui.artifact_list("model_config", lambda m: {"model": m["config"]["model"], "seed": m["config"].get("seed"),
                                                      "derived from": m.get("derived_from") or ""}, key="mcl")
    if sel:
        m = c.store.get(sel)
        st.json(m["config"])
        ui.download_json("Export configuration (JSON)", m["config"], f"{sel}.json", key="mc_dl")
        other = ui.artifact_picker("Compare with", "model_config", key="mc_diff", allow_none=True)
        if other and other != sel:
            st.dataframe(c.pipeline.diff(sel, other), hide_index=True, width="stretch")
