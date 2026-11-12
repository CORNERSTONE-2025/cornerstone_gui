import os

import pandas as pd
import streamlit as st

import subprocess

from cs_gui import datastore, ui
from cs_gui.demo import is_demo_store
from cs_gui.jobs import runner_env
from cs_gui.settings import _ENV, SETTINGS_FILE, save_settings
from cs_gui.store import ARTIFACT_TYPES

c = ui.page_header("CORNERSTONE — Home and project status",
                   "GUI for the solar flare forecasting pipeline of the CORNERSTONE package "
                   "(SHARP → labeling → dataset → training → inference → evaluation).")

col1, col2 = st.columns([3, 2])
with col1:
    st.subheader("Workflow")
    st.graphviz_chart("""
    digraph { rankdir=LR; node [shape=box, style="rounded", fontname="Helvetica", fontsize=11];
      DB [label="Parquet store\\n(SHARP + GOES)", shape=cylinder];
      L [label="Labeling\\ncornerstone.labeling"]; D [label="Dataset HDF5\\ncornerstone.dataset"];
      S [label="Filters and split\\ncreate_dataset_splits"]; M [label="Model config"];
      T [label="Training CV\\ntrain_models"]; I [label="Inference\\n(unseen events)"];
      V [label="Metrics and comparison"];
      DB -> L -> D -> S -> T -> I -> V; M -> T; DB -> D; }""")
    st.markdown("Every step produces a **persistent artifact** (manifest + files + log) that the next "
                "step uses without re-running the chain. Completed artifacts are never overwritten: "
                "changing a parameter creates a **variant**.")

with col2:
    st.subheader("Artifacts")
    rows = []
    for t, (_, name) in ARTIFACT_TYPES.items():
        items = c.store.list(t)
        rows.append({"type": name, "total": len(items),
                     "completed": sum(m["status"] == "completed" for m in items),
                     "failed": sum(m["status"] == "failed" for m in items),
                     "in progress": sum(m["status"] in ("pending", "running") for m in items)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    ui.job_monitor()

st.divider()
st.subheader("Environment status")
s = c.settings
a, b, d = st.columns(3)
with a:
    st.markdown("**CORNERSTONE package**")
    st.code(f"root:   {s.cornerstone_root}\npython: {s.python}", language="text")
    info = ui.package_info() if not s.problems() else None
    if info:
        cs = info.get("cornerstone") or {}
        st.markdown(f"commit `{(cs.get('commit') or '?')[:10]}` · branch `{cs.get('branch')}`"
                    + (" · :orange[uncommitted local changes]" if cs.get("dirty") else ""))
        st.markdown(f"CUDA available: **{'yes' if info['cuda_available'] else 'no'}** · "
                    f"trainable models: {len(info['trainable_models'])}")
        if st.button("Reload package features"):
            ui.package_info(refresh=True)
            st.rerun()
with b:
    st.markdown("**Data store**")
    st.code(s.active_data_root, language="text")
    try:
        t = datastore.status(s.active_data_root)
        st.markdown(f"{len(t)} datasets · " + ("**synthetic DEMO**" if is_demo_store(s.active_data_root) else "real"))
        for _, r in t.iterrows():
            st.caption(f"{r['dataset']}: {r['rows']:,} rows · until {r['data until'] or '—'}")
        st.caption("CME events: **not present** (GOES flares only). Manage the data in the **Data** page.")
    except Exception as e:  # noqa: BLE001
        st.error(f"Cannot read the data store: {e}")
with d:
    st.markdown("**Workspace**")
    st.code(str(s.active_workspace), language="text")
    import shutil
    free = shutil.disk_usage(s.active_workspace).free / 1e9
    st.markdown(f"Free space: **{free:.1f} GB** (minimum to start a job: {s.min_free_disk_gb} GB)")

st.divider()
with st.expander("⚙️ GUI configuration", expanded=bool(s.problems())):
    overridden = [env for env in _ENV if os.environ.get(env)]
    if overridden:
        st.caption("Set through environment variables (they take precedence over the values saved here): "
                   + ", ".join(f"`{e}`" for e in overridden))
    with st.form("settings"):
        root = st.text_input("CORNERSTONE package folder", s.cornerstone_root)
        py = st.text_input("CORNERSTONE Python interpreter (empty = <root>/.venv/bin/python, else the GUI one)", s.cornerstone_python)
        dr = st.text_input("Data store folder (empty = data_root of data_config.json)", s.data_root)
        ws = st.text_input("Real mode workspace", s.workspace)
        mode = st.radio("Mode", ["real", "demo"], index=0 if s.mode == "real" else 1, horizontal=True,
                        format_func=lambda x: "Real (scientific data)" if x == "real" else "Demo (synthetic data)")
        mfree = st.number_input("Minimum free disk space to start a job (GB)", 0.5, 1000.0, float(s.min_free_disk_gb))
        if st.form_submit_button("Save configuration"):
            s.cornerstone_root, s.cornerstone_python, s.data_root, s.workspace, s.mode, s.min_free_disk_gb = \
                root, py, dr, ws, mode, mfree
            save_settings(s)
            ui.reset_ctx()
            st.success(f"Configuration saved to {SETTINGS_FILE.name}")
            st.rerun()
    if s.mode == "demo":
        st.markdown("**Demo data** (synthetic, same layout as the real store, for trying out/testing only)")
        n = st.slider("Number of synthetic Active Regions", 10, 120, 36)
        if st.button("Generate / regenerate demo data"):
            r = subprocess.run([s.python, "-m", "cs_gui.demo", "--root", str(s.demo_data_root), "--harps", str(n)],
                               env=runner_env(s), capture_output=True, text=True, timeout=600)
            if r.returncode:
                st.error("Generation failed:\n" + (r.stderr or r.stdout)[-3000:])
            else:
                st.success(f"Demo data created: {r.stdout.strip()}")
