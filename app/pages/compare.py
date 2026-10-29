import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from cs_gui import metrics as gm
from cs_gui import ui

c = ui.page_header("Algorithm and experiment comparison",
                   "Comparison table of inferences (models, configurations, dataset versions) with a check of "
                   "methodological validity.")
infs = [m for m in c.store.list("inference") if m["status"] == "completed"]
if len(infs) < 1:
    st.info("At least two completed inferences are needed.")
    st.stop()
sel = st.multiselect("Inferences to compare", [m["id"] for m in infs], [m["id"] for m in infs[:2]],
                     format_func=lambda x: c.store.get(x)["name"])
if not sel:
    st.stop()

info = ui.package_info() or {}
rows, keysets, preds = [], {}, {}
for aid in sel:
    m = c.store.get(aid)
    tr = c.store.get(m["parents"]["training"])
    mc = c.store.get(tr["parents"]["model_config"])["config"]
    sp = c.store.get(tr["parents"]["split"])
    defaults = info.get("model_configs", {}).get(mc["model"], {})
    changed = {k: v for k, v in mc.get("model_params", {}).items() if defaults.get(k) != v}
    tv = [v for k, v in sp["summary"]["files"].items() if k != "test"]
    df = pd.read_csv(c.store.path(aid) / "predictions.csv")
    preds[aid] = df
    for w, e in m["summary"]["windows"].items():
        g = df[(df["window"] == w) & (~df["in_trainval_harp"] if m["config"].get("exclude_overlap", True) else True)]
        keysets[(aid, w)] = frozenset(zip(g["HARPNUM"], g["t_issue"]))
        r = {"experiment": m["name"], "artifact": aid, "window": w, "algorithm": mc["model"],
             "changed hyperparameters": json.dumps(changed) if changed else "package defaults",
             "seed": mc.get("seed"), "n features": sp["summary"]["n_features"],
             "features": ",".join(sp["summary"]["feature_names"]), "split": sp["id"],
             "split strategy": sp["config"]["SPLITTING_STRATEGY"],
             "train/val": f"{min(v['time_start'] for v in tv)[:10]} → {max(v['time_end'] for v in tv)[:10]}",
             "inference source": f"{m['parents']['source']}:{m['config'].get('source_file', 'all')}",
             "evaluated period": f"{g['t_issue'].min()[:10]} → {g['t_issue'].max()[:10]}" if len(g) else "",
             "n evaluated": e["n_evaluated"], "threshold": round(e["threshold"], 4), "threshold criterion": e["threshold_source"],
             "training [s]": tr["summary"].get("training_seconds"), "inference [s]": (m.get("run") or {}).get("duration_s")}
        if e.get("metrics"):
            for k in ("TSS", "HSS", "Precision", "Recall", "Specificity", "F1", "ROC_AUC", "PR_AUC"):
                r[k] = e["metrics"][k]
            ci = (e.get("ci") or {}).get("TSS")
            r["TSS CI low"], r["TSS CI high"] = (ci["low"], ci["high"]) if ci else (None, None)
        rows.append(r)
res = pd.DataFrame(rows)
st.dataframe(res, hide_index=True, width="stretch")
st.download_button("Export CSV", res.to_csv(index=False), "comparison.csv", "text/csv")

st.subheader("Comparison validity")
for w, g in res.groupby("window"):
    sets = {keysets[(a, w)] for a in g["artifact"]}
    hard, soft = [], []      # hard issues invalidate the comparison, soft ones are only reported
    if len(sets) > 1:
        hard.append("the evaluated samples differ")
    if g["inference source"].nunique() > 1:
        hard.append("different inference sources")
    if g["threshold criterion"].nunique() > 1:
        soft.append("different threshold criteria (threshold-dependent metrics are not homogeneous)")
    if g["split"].nunique() > 1:
        soft.append("different splits (training on different data: allowed to compare dataset versions, "
                    "but it must be stated)")
    issues = hard + soft
    if "TSS" not in g or g["TSS"].isna().all():
        st.info(f"Window {w}: metrics not available.")
        continue
    if hard:
        st.error(f"Window **{w}**: comparison NOT valid — " + "; ".join(issues) + ". No model is declared the best.")
    else:
        best = g.loc[g["TSS"].idxmax()]
        overlap = ""
        if g["TSS CI low"].notna().all() and len(g) > 1:
            others = g.drop(best.name)
            if (others["TSS CI high"] >= best["TSS CI low"]).any():
                overlap = " The confidence intervals overlap: the difference is not significant."
        msg = (f"Window **{w}**: same {len(next(iter(sets)))} samples for all models. Highest TSS: "
               f"**{best['esperimento']}** ({best['TSS']:.3f}).{overlap}")
        if issues:
            msg += " Note: " + "; ".join(issues) + "."
        st.success(msg)

if "TSS" in res:
    fig = go.Figure()
    for exp, g in res.groupby("experiment"):
        err = None
        if g["TSS CI low"].notna().all():
            err = dict(type="data", symmetric=False, array=(g["TSS CI high"] - g["TSS"]).tolist(),
                       arrayminus=(g["TSS"] - g["TSS CI low"]).tolist())
        fig.add_trace(go.Bar(x=g["window"], y=g["TSS"], name=exp, error_y=err))
    fig.update_layout(barmode="group", title="TSS per window (error bars = 95% bootstrap CI)", yaxis_title="TSS")
    st.plotly_chart(fig, width="stretch")

st.subheader("ROC and Precision-Recall curves")
w = st.selectbox("Window", sorted(res["window"].unique()))
fr, fp = go.Figure(), go.Figure()
for aid in sel:
    m = c.store.get(aid)
    df = preds[aid]
    g = df[(df["window"] == w) & (~df["in_trainval_harp"])]
    if len(g) and 0 < g["y_true"].sum() < len(g):
        roc, pr = gm.roc_curve(g["y_true"], g["score"]), gm.pr_curve(g["y_true"], g["score"])
        fr.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", name=m["name"]))
        fp.add_trace(go.Scatter(x=pr["recall"], y=pr["precision"], mode="lines", name=m["name"]))
fr.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dot", color="#999"))
fr.update_layout(title=f"ROC · {w}", xaxis_title="FPR", yaxis_title="TPR")
fp.update_layout(title=f"Precision-Recall · {w}", xaxis_title="Recall", yaxis_title="Precision")
a, b = st.columns(2)
a.plotly_chart(fr, width="stretch")
b.plotly_chart(fp, width="stretch")
