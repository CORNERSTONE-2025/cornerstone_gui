import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from cs_gui import metrics as gm
from cs_gui import ui

c = ui.page_header("Metriche globali e valutazione",
                   "Calcolate dalle predizioni salvate negli artefatti di inferenza. TSS = Sensitivity + Specificity − 1.")

infs = [m for m in c.store.list("inference") if m["status"] == "completed"]
if not infs:
    st.info("Nessuna inferenza completata: esegui prima un'inferenza.")
    st.stop()

sel = st.multiselect("Esperimenti (inferenze)", [m["id"] for m in infs], [infs[0]["id"]],
                     format_func=lambda x: c.store.get(x)["name"])
if not sel:
    st.stop()


@st.cache_data
def _load(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["t_issue"])


frames = []
for aid in sel:
    p = c.store.path(aid) / "predictions.csv"
    df = _load(str(p), p.stat().st_mtime)
    m = c.store.get(aid)
    tr = c.store.get(m["parents"]["training"])
    df["esperimento"] = m["name"]
    df["artefatto"] = aid
    df["modello"] = m["summary"]["model"]
    df["dataset"] = m["parents"]["source"]
    df["label"] = df["label"].astype(str)
    df["evento"] = c.store.get(c.store.get(tr["parents"]["split"])["parents"]["dataset"])["summary"]["label_key"]
    frames.append(df)
data = pd.concat(frames, ignore_index=True)

with st.container(border=True):
    st.markdown("**Filtri**")
    f1, f2, f3, f4 = st.columns(4)
    windows = f1.multiselect("Finestra", sorted(data["window"].unique()), sorted(data["window"].unique()))
    events = f2.multiselect("Tipologia di evento", sorted(data["evento"].unique()), sorted(data["evento"].unique()))
    tmin, tmax = data["t_issue"].min().date(), data["t_issue"].max().date()
    tr_ = f3.date_input("Intervallo temporale (emissione)", (tmin, tmax), tmin, tmax)
    harps_txt = f4.text_input("Active Region (HARPNUM, separati da virgola)")
    g1, g2, g3 = st.columns(3)
    thr_mode = g1.radio("Soglia", ["artifact", "custom"], horizontal=True,
                        format_func=lambda x: {"artifact": "quella salvata nell'esperimento", "custom": "personalizzata"}[x])
    thr_custom = g2.slider("Soglia personalizzata", 0.0, 1.0, 0.5, 0.01, disabled=thr_mode != "custom")
    include_tv = g3.checkbox("Includi campioni di AR viste in train/val (sconsigliato)", False)

flt = data[data["window"].isin(windows) & data["evento"].isin(events)]
if len(tr_) == 2:
    flt = flt[(flt["t_issue"].dt.date >= tr_[0]) & (flt["t_issue"].dt.date <= tr_[1])]
if harps_txt.strip():
    try:
        flt = flt[flt["HARPNUM"].isin([int(x) for x in harps_txt.replace(" ", "").split(",") if x])]
    except ValueError:
        st.error("HARPNUM non validi")
if not include_tv:
    flt = flt[~flt["in_trainval_harp"]]
if flt.empty:
    st.warning("Nessun campione dopo i filtri.")
    st.stop()

rows = []
for (exp, aid, w), g in flt.groupby(["esperimento", "artefatto", "window"]):
    thr = float(g["threshold"].iloc[0]) if thr_mode == "artifact" else thr_custom
    crit = (c.store.get(aid)["summary"]["windows"][w]["threshold_source"] if thr_mode == "artifact"
            else "soglia personalizzata (dashboard)")
    pm = gm.point_metrics(g["y_true"].values, g["score"].values, thr)
    rows.append({"esperimento": exp, "artefatto": aid, "finestra": w, "modello": g["modello"].iloc[0],
                 "dataset": g["dataset"].iloc[0], "soglia": round(thr, 4), "criterio soglia": crit,
                 "periodo": f"{g['t_issue'].min():%Y-%m-%d} → {g['t_issue'].max():%Y-%m-%d}",
                 "n": pm["n"], "positivi": pm["n_pos"],
                 **{k: pm[k] for k in ("TSS", "HSS", "Precision", "Recall", "Specificity", "F1", "Accuracy",
                                       "ROC_AUC", "PR_AUC", "FPR", "FNR", "TP", "FP", "TN", "FN")}})
res = pd.DataFrame(rows)
st.subheader("Metriche")
fmt = {k: "{:.4f}" for k in ("TSS", "HSS", "Precision", "Recall", "Specificity", "F1", "Accuracy", "ROC_AUC", "PR_AUC", "FPR", "FNR")}
st.dataframe(res.style.format(fmt, na_rep="n.d."), hide_index=True, width="stretch")
if res["dataset"].nunique() > 1 or res["periodo"].nunique() > 1 or res["n"].nunique() > 1:
    st.warning("⚠️ Le righe sono calcolate su **campioni diversi** (dataset, periodo o numero di campioni diversi): "
               "le metriche non sono direttamente confrontabili. Usa la pagina *Confronto* per confronti controllati.")
if res["ROC_AUC"].isna().any():
    st.caption("n.d. = metrica non definita (manca una delle due classi o divisione per zero).")
x1, x2 = st.columns(2)
x1.download_button("Esporta CSV", res.to_csv(index=False), "metriche.csv", "text/csv")
x2.download_button("Esporta JSON", res.to_json(orient="records", indent=2), "metriche.json", "application/json")

st.subheader("Dettaglio")
key = st.selectbox("Esperimento e finestra", list(zip(res["artefatto"], res["finestra"])),
                   format_func=lambda k: f"{c.store.get(k[0])['name']} · {k[1]}")
g = flt[(flt["artefatto"] == key[0]) & (flt["window"] == key[1])]
r = res[(res["artefatto"] == key[0]) & (res["finestra"] == key[1])].iloc[0]
y, s = g["y_true"].values, g["score"].values
a, b = st.columns(2)
cm = np.array([[r["TN"], r["FP"]], [r["FN"], r["TP"]]])
a.plotly_chart(px.imshow(cm, text_auto=True, x=["pred NO", "pred YES"], y=["vero NO", "vero YES"],
                         color_continuous_scale="Blues", title=f"Matrice di confusione (soglia {r['soglia']})"),
               width="stretch")
tv = gm.tss_vs_threshold(y, s)
figt = px.line(tv, x="threshold", y="TSS", title="TSS al variare della soglia")
figt.add_vline(x=r["soglia"], line_dash="dash", annotation_text="soglia usata")
b.plotly_chart(figt, width="stretch")
if 0 < y.sum() < len(y):
    roc, pr = gm.roc_curve(y, s), gm.pr_curve(y, s)
    c1, c2 = st.columns(2)
    fr = go.Figure(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", name=f"AUC {r['ROC_AUC']:.3f}"))
    fr.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dot", color="#999"))
    fr.update_layout(title="Curva ROC", xaxis_title="FPR", yaxis_title="TPR (Sensitivity)")
    c1.plotly_chart(fr, width="stretch")
    fp = go.Figure(go.Scatter(x=pr["recall"], y=pr["precision"], mode="lines", name=f"AP {r['PR_AUC']:.3f}"))
    fp.add_hline(y=y.mean(), line_dash="dot", annotation_text="tasso di positivi (baseline)")
    fp.update_layout(title="Curva Precision-Recall", xaxis_title="Recall", yaxis_title="Precision")
    c2.plotly_chart(fp, width="stretch")

with st.expander("Incertezza (bootstrap)"):
    n = st.select_slider("Ricampionamenti", [200, 500, 1000, 2000], 1000)
    cluster = st.checkbox("Per Active Region (cluster bootstrap)", True)
    if st.button("Calcola intervalli di confidenza al 95%"):
        ci = gm.bootstrap_ci(y, s, float(r["soglia"]), metrics=("TSS", "HSS", "F1", "ROC_AUC", "PR_AUC"), n_boot=n,
                             group=g["HARPNUM"].values if cluster else None)
        st.dataframe(pd.DataFrame([{"metrica": k, "valore": r[k], "CI 2.5%": v["low"], "CI 97.5%": v["high"]}
                                   for k, v in ci.items() if not k.startswith("_")]), hide_index=True)
        st.caption("Metodo: " + ci["_method"])
st.caption("Distribuzione delle classi nei campioni valutati: " +
           json.dumps(g["label"].value_counts().to_dict()))
