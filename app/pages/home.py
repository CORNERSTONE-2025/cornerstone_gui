import os

import pandas as pd
import streamlit as st

from cs_gui import db as dbq
from cs_gui import ui
from cs_gui.demo import build_demo_db, is_demo_db
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
      DB [label="DB SQLite\\n(SHARP + GOES)", shape=cylinder];
      L [label="Labeling\\nsplitting_labeling"]; D [label="Dataset HDF5\\ncreate_hdf5_from_db"];
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
        if info["gpu_only_models"] and not info["cuda_available"]:
            st.caption(f"Require a GPU (not usable here): {', '.join(info['gpu_only_models'])}")
        if st.button("Reload package features"):
            ui.package_info(refresh=True)
            st.rerun()
with b:
    st.markdown("**Database**")
    st.code(s.active_db_path, language="text")
    try:
        tabs = dbq.tables(s.active_db_path)
        st.markdown(f"{len(tabs)} tables · " + ("**synthetic DEMO**" if is_demo_db(s.active_db_path) else "real"))
        lt = dbq.label_tables(s.active_db_path)
        st.caption("Label tables: " + (", ".join(lt["table"]) if len(lt) else "none"))
        st.caption("Feature tables: " + ", ".join(dbq.feature_tables(s.active_db_path)))
        st.caption("CME events: **not present** in the database (GOES flares only).")
    except Exception as e:  # noqa: BLE001
        st.error(f"Cannot read the DB: {e}")
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
        dbp = st.text_input("Real SQLite database (empty = local_db_path from db_config.json)", s.db_path)
        ws = st.text_input("Real mode workspace", s.workspace)
        mode = st.radio("Mode", ["real", "demo"], index=0 if s.mode == "real" else 1, horizontal=True,
                        format_func=lambda x: "Real (scientific data)" if x == "real" else "Demo (synthetic data)")
        mfree = st.number_input("Minimum free disk space to start a job (GB)", 0.5, 1000.0, float(s.min_free_disk_gb))
        if st.form_submit_button("Save configuration"):
            s.cornerstone_root, s.cornerstone_python, s.db_path, s.workspace, s.mode, s.min_free_disk_gb = \
                root, py, dbp, ws, mode, mfree
            save_settings(s)
            ui.reset_ctx()
            st.success(f"Configuration saved to {SETTINGS_FILE.name}")
            st.rerun()
    if s.mode == "demo":
        st.markdown("**Demo database** (synthetic, same schema as the real DB, for trying out/testing only)")
        n = st.slider("Number of synthetic Active Regions", 10, 120, 36)
        if st.button("Generate / regenerate demo DB"):
            stats = build_demo_db(s.demo_db_path, n_harps=n)
            dbq._tables.cache_clear()
            st.success(f"Demo DB created: {stats}")
