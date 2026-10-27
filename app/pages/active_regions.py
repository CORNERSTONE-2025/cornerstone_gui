import h5py
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from cs_gui import db as dbq
from cs_gui import h5info, ui

c = ui.page_header("Storia di una Active Region",
                   "Serie temporali SHARP, flare GOES osservati, label e predizioni sulla stessa asse temporale "
                   "(zoom e pan sincronizzati).")
if c.settings.problems():
    st.stop()
DB = c.settings.active_db_path


@st.cache_data(show_spinner="Indicizzo le Active Region…")
def _harp_index(db, table, mt):
    return dbq.harp_index(db, table, mt)


@st.cache_data
def _split_membership(split_dir: str, harp: int) -> pd.DataFrame:
    rows = []
    from pathlib import Path
    for p in sorted(Path(split_dir).glob("split_*.h5")):
        role = p.stem.replace("split_", "")
        with h5py.File(p, "r") as f:
            h = f["HARPNUM"][:]
            idx = np.flatnonzero(h == harp)
            if idx.size:
                it = h5info.issue_times(f["timestamps"][np.sort(idx)])
                for t in it:
                    rows.append({"t_issue": t, "insieme": role})
    return pd.DataFrame(rows)


ftables = dbq.feature_tables(DB)
a, b, d = st.columns([2, 2, 3])
table = a.selectbox("Tabella feature", ftables, index=ftables.index("SHARPC_clean_DEF") if "SHARPC_clean_DEF" in ftables else 0)
table_type = table.split("_")[-1]
idx = _harp_index(DB, table, dbq._mtime(DB))

inf_all = [m for m in c.store.list("inference") if m["status"] == "completed"]
only_pred = b.checkbox("Solo AR con predizioni", bool(inf_all))
harps = idx["HARPNUM"].tolist()
if only_pred and inf_all:
    s = set()
    for m in inf_all:
        s |= set(pd.read_csv(c.store.path(m["id"]) / "predictions.csv", usecols=["HARPNUM"])["HARPNUM"].unique())
    harps = [h for h in harps if h in s] or harps
row_of = idx.set_index("HARPNUM")
harp = d.selectbox("Active Region (HARPNUM)", harps,
                   format_func=lambda h: f"HARP {h} · NOAA {row_of.loc[h, 'NOAA_AR']} · "
                                         f"{str(row_of.loc[h, 'start'])[:10]} → {str(row_of.loc[h, 'end'])[:10]}")

cols = [x for x in dbq.columns(DB, table) if x not in ("T_OBS", "T_OBS1", "HARPNUM", "QUALITY", "NOAA_AR", "NOAA_NUM", "NOAA_ARS")]
default_f = [f for f in ("USFLUX_N1", "USFLUX", "TOTUSJH_N1", "TOTUSJH", "R_VALUE") if f in cols][:3]
feats = st.multiselect("Feature HMI/SHARP", cols, default_f)

e1, e2, e3 = st.columns(3)
lab_opts = [None] + [m["id"] for m in c.store.list("labels") if m["status"] == "completed"
                     and m["config"]["table_type"] == table_type]
lab_db = [t for t in dbq.label_tables(DB).get("table", []) if f"_{table_type}_" in t] if len(dbq.label_tables(DB)) else []
lab_src = e1.selectbox("Label", lab_opts + lab_db,
                       format_func=lambda x: "— nessuna —" if x is None else (f"artefatto {x}" if not str(x).startswith("DF_") else f"tabella DB {x}"))
inf_sel = e2.multiselect("Predizioni (artefatti di inferenza)", [m["id"] for m in inf_all],
                         format_func=lambda x: c.store.get(x)["name"])
split_opts = [None] + [m["id"] for m in c.store.list("split") if m["status"] == "completed"]
split_sel = e3.selectbox("Mostra appartenenza a train/val/test dello split", split_opts,
                         format_func=lambda x: "— nessuno —" if x is None else c.store.get(x)["name"])

series = dbq.harp_series(DB, table, harp, feats)
flares = dbq.harp_flares(DB, table_type, harp)
labels = pd.DataFrame()
if lab_src:
    if str(lab_src).startswith("DF_"):
        labels = dbq.harp_labels(DB, lab_src, harp)
    else:
        lm = c.store.get(lab_src)
        labels = dbq.labels_from_artifact(str(c.store.path(lab_src) / "labels.sqlite"), lm["summary"]["table"], harp)
preds = []
for iid in inf_sel:
    df = pd.read_csv(c.store.path(iid) / "predictions.csv")
    df = df[df["HARPNUM"] == harp].copy()
    df["t_issue"] = pd.to_datetime(df["t_issue"])
    df["artefatto"] = c.store.get(iid)["name"]
    preds.append(df)
preds = pd.concat(preds) if preds else pd.DataFrame()
member = _split_membership(str(c.store.path(split_sel) / "split" / "splits"), int(harp)) if split_sel else pd.DataFrame()

n_rows = len(feats) + 1 + (1 if len(labels) else 0) + (1 if len(preds) else 0) + (1 if len(member) else 0)
titles = feats + ["Flare GOES osservati (log10 flusso)"] + (["Label (YES=1)"] if len(labels) else []) \
    + (["Probabilità predetta e soglia"] if len(preds) else []) + (["Insieme dello split"] if len(member) else [])
fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True, vertical_spacing=0.025, subplot_titles=titles)
r = 1
for f in feats:
    fig.add_trace(go.Scattergl(x=series["time"], y=series[f], mode="lines", name=f, showlegend=False), row=r, col=1)
    r += 1
if len(flares):
    fig.add_trace(go.Scatter(x=flares["tmax"], y=np.log10(flares["fflux"].astype(float)), mode="markers",
                             marker=dict(size=9, symbol="star", color="#d62728"), name="flare (picco)",
                             text=flares["part1"], hovertemplate="%{text}<br>%{x}<extra></extra>"), row=r, col=1)
for lvl, cls in ((-6, "C"), (-5, "M"), (-4, "X")):
    fig.add_hline(y=lvl, line_dash="dot", line_color="#999", annotation_text=cls, row=r, col=1)
r += 1
if len(labels):
    fig.add_trace(go.Scatter(x=labels["t_X"], y=(labels["label"] == "YES").astype(int), mode="lines+markers",
                             line_shape="hv", name="label", text=labels["label_u"],
                             hovertemplate="t_X %{x}<br>%{text}<extra></extra>"), row=r, col=1)
    r += 1
if len(preds):
    for (art, w), g in preds.groupby(["artefatto", "window"]):
        g = g.sort_values("t_issue")
        fig.add_trace(go.Scatter(x=g["t_issue"], y=g["score"], mode="lines+markers", name=f"{art} · {w}",
                                 marker=dict(symbol=np.where(g["in_trainval_harp"], "x", "circle")),
                                 hovertemplate="score %{y:.3f}<br>%{x}<extra></extra>"), row=r, col=1)
        fig.add_trace(go.Scatter(x=g["t_issue"], y=g["threshold"], mode="lines", line=dict(dash="dash", width=1),
                                 name=f"soglia {art} · {w}"), row=r, col=1)
    r += 1
if len(member):
    member["t"] = pd.to_datetime(member["t_issue"])
    for role, g in member.groupby("insieme"):
        fig.add_trace(go.Scatter(x=g["t"], y=[role] * len(g), mode="markers", name=role, marker=dict(size=7)), row=r, col=1)
fig.update_layout(height=230 * n_rows, hovermode="x unified", margin=dict(t=40, b=20),
                  legend=dict(orientation="h", y=-0.05))
st.plotly_chart(fig, width="stretch")

st.caption("Convenzioni: la label in t_X indica se c'è almeno un flare ≥ soglia di classe in [t_X, t_X+Δ]; "
           "la predizione è emessa in t_X usando solo dati SHARP fino a t_X. "
           "Le predizioni marcate con ✕ appartengono ad Active Region usate in train/validation del modello "
           "(non sono previsioni su eventi non visti).")
k = st.columns(4)
k[0].metric("Osservazioni SHARP", len(series))
k[1].metric("Flare associati", len(flares))
k[2].metric("Label YES", int((labels["label"] == "YES").sum()) if len(labels) else 0)
k[3].metric("Predizioni", len(preds))
if len(flares):
    with st.expander("Cronologia dei flare"):
        st.dataframe(flares, hide_index=True, width="stretch")
st.info("Eventi CME: non disponibili nel database del pacchetto.")
