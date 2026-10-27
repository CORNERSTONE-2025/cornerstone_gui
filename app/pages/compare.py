import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from cs_gui import metrics as gm
from cs_gui import ui

c = ui.page_header("Confronto tra algoritmi ed esperimenti",
                   "Tabella comparativa di inferenze (modelli, configurazioni, versioni del dataset) con controllo "
                   "di validità metodologica.")
infs = [m for m in c.store.list("inference") if m["status"] == "completed"]
if len(infs) < 1:
    st.info("Servono almeno due inferenze completate.")
    st.stop()
sel = st.multiselect("Inferenze da confrontare", [m["id"] for m in infs], [m["id"] for m in infs[:2]],
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
        r = {"esperimento": m["name"], "artefatto": aid, "finestra": w, "algoritmo": mc["model"],
             "iperparametri modificati": json.dumps(changed) if changed else "default del pacchetto",
             "seed": mc.get("seed"), "n feature": sp["summary"]["n_features"],
             "feature": ",".join(sp["summary"]["feature_names"]), "split": sp["id"],
             "strategia split": sp["config"]["SPLITTING_STRATEGY"],
             "train/val": f"{min(v['time_start'] for v in tv)[:10]} → {max(v['time_end'] for v in tv)[:10]}",
             "sorgente inferenza": f"{m['parents']['source']}:{m['config'].get('source_file', 'all')}",
             "periodo valutato": f"{g['t_issue'].min()[:10]} → {g['t_issue'].max()[:10]}" if len(g) else "",
             "n valutati": e["n_evaluated"], "soglia": round(e["threshold"], 4), "criterio soglia": e["threshold_source"],
             "training [s]": tr["summary"].get("training_seconds"), "inferenza [s]": (m.get("run") or {}).get("duration_s")}
        if e.get("metrics"):
            for k in ("TSS", "HSS", "Precision", "Recall", "Specificity", "F1", "ROC_AUC", "PR_AUC"):
                r[k] = e["metrics"][k]
            ci = (e.get("ci") or {}).get("TSS")
            r["TSS CI low"], r["TSS CI high"] = (ci["low"], ci["high"]) if ci else (None, None)
        rows.append(r)
res = pd.DataFrame(rows)
st.dataframe(res, hide_index=True, width="stretch")
st.download_button("Esporta CSV", res.to_csv(index=False), "confronto.csv", "text/csv")

st.subheader("Validità del confronto")
for w, g in res.groupby("finestra"):
    sets = {keysets[(a, w)] for a in g["artefatto"]}
    issues = []
    if len(sets) > 1:
        issues.append("i campioni valutati non coincidono")
    if g["sorgente inferenza"].nunique() > 1:
        issues.append("sorgenti di inferenza diverse")
    if g["criterio soglia"].nunique() > 1:
        issues.append("criteri di soglia diversi (le metriche dipendenti dalla soglia non sono omogenee)")
    if g["split"].nunique() > 1:
        issues.append("split diversi (training su dati diversi: ammesso per confrontare versioni del dataset, "
                      "ma va dichiarato)")
    hard = [i for i in issues if not i.startswith("split diversi") and not i.startswith("criteri")]
    if "TSS" not in g or g["TSS"].isna().all():
        st.info(f"Finestra {w}: metriche non disponibili.")
        continue
    if hard:
        st.error(f"Finestra **{w}**: confronto NON valido — " + "; ".join(issues) + ". Nessun modello viene indicato come migliore.")
    else:
        best = g.loc[g["TSS"].idxmax()]
        overlap = ""
        if g["TSS CI low"].notna().all() and len(g) > 1:
            others = g.drop(best.name)
            if (others["TSS CI high"] >= best["TSS CI low"]).any():
                overlap = " Gli intervalli di confidenza si sovrappongono: la differenza non è significativa."
        msg = (f"Finestra **{w}**: stessi {len(next(iter(sets)))} campioni per tutti i modelli. TSS più alto: "
               f"**{best['esperimento']}** ({best['TSS']:.3f}).{overlap}")
        if issues:
            msg += " Nota: " + "; ".join(issues) + "."
        st.success(msg)

if "TSS" in res:
    fig = go.Figure()
    for exp, g in res.groupby("esperimento"):
        err = None
        if g["TSS CI low"].notna().all():
            err = dict(type="data", symmetric=False, array=(g["TSS CI high"] - g["TSS"]).tolist(),
                       arrayminus=(g["TSS"] - g["TSS CI low"]).tolist())
        fig.add_trace(go.Bar(x=g["finestra"], y=g["TSS"], name=exp, error_y=err))
    fig.update_layout(barmode="group", title="TSS per finestra (barre di errore = CI 95% bootstrap)", yaxis_title="TSS")
    st.plotly_chart(fig, width="stretch")

st.subheader("Curve ROC e Precision-Recall")
w = st.selectbox("Finestra", sorted(res["finestra"].unique()))
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
