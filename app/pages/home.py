import pandas as pd
import streamlit as st

from cs_gui import db as dbq
from cs_gui import ui
from cs_gui.demo import build_demo_db, is_demo_db
from cs_gui.settings import save_settings
from cs_gui.store import ARTIFACT_TYPES

c = ui.page_header("CORNERSTONE — Home e stato del progetto",
                   "GUI per la pipeline di previsione dei flare solari del pacchetto CORNERSTONE "
                   "(SHARP → labeling → dataset → training → inferenza → valutazione).")

col1, col2 = st.columns([3, 2])
with col1:
    st.subheader("Flusso di lavoro")
    st.graphviz_chart("""
    digraph { rankdir=LR; node [shape=box, style="rounded", fontname="Helvetica", fontsize=11];
      DB [label="DB SQLite\\n(SHARP + GOES)", shape=cylinder];
      L [label="Labeling\\nsplitting_labeling"]; D [label="Dataset HDF5\\ncreate_hdf5_from_db"];
      S [label="Filtri e split\\ncreate_dataset_splits"]; M [label="Config modello"];
      T [label="Training CV\\ntrain_models"]; I [label="Inferenza\\n(eventi non visti)"];
      V [label="Metriche e confronto"];
      DB -> L -> D -> S -> T -> I -> V; M -> T; DB -> D; }""")
    st.markdown("Ogni passo produce un **artefatto persistente** (manifest + file + log) che il passo "
                "successivo usa senza rieseguire la catena. Gli artefatti completati non vengono mai "
                "sovrascritti: cambiare un parametro crea una **variante**.")

with col2:
    st.subheader("Artefatti")
    rows = []
    for t, (_, name) in ARTIFACT_TYPES.items():
        items = c.store.list(t)
        rows.append({"tipo": name, "totale": len(items),
                     "completati": sum(m["status"] == "completed" for m in items),
                     "falliti": sum(m["status"] == "failed" for m in items),
                     "in corso": sum(m["status"] in ("pending", "running") for m in items)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    ui.job_monitor()

st.divider()
st.subheader("Stato dell'ambiente")
s = c.settings
a, b, d = st.columns(3)
with a:
    st.markdown("**Pacchetto CORNERSTONE**")
    st.code(f"root:   {s.cornerstone_root}\npython: {s.python}", language="text")
    info = ui.package_info() if not s.problems() else None
    if info:
        cs = info.get("cornerstone") or {}
        st.markdown(f"commit `{(cs.get('commit') or '?')[:10]}` · branch `{cs.get('branch')}`"
                    + (" · :orange[modifiche locali non committate]" if cs.get("dirty") else ""))
        st.markdown(f"CUDA disponibile: **{'sì' if info['cuda_available'] else 'no'}** · "
                    f"modelli addestrabili: {len(info['trainable_models'])}")
        if info["gpu_only_models"] and not info["cuda_available"]:
            st.caption(f"Richiedono GPU (non utilizzabili qui): {', '.join(info['gpu_only_models'])}")
        if st.button("Rileggi funzionalità del pacchetto"):
            ui.package_info(refresh=True)
            st.rerun()
with b:
    st.markdown("**Database**")
    st.code(s.active_db_path, language="text")
    try:
        tabs = dbq.tables(s.active_db_path)
        st.markdown(f"{len(tabs)} tabelle · " + ("**DEMO sintetico**" if is_demo_db(s.active_db_path) else "reale"))
        lt = dbq.label_tables(s.active_db_path)
        st.caption("Tabelle di label: " + (", ".join(lt["table"]) if len(lt) else "nessuna"))
        st.caption("Tabelle feature: " + ", ".join(dbq.feature_tables(s.active_db_path)))
        st.caption("Eventi CME: **non presenti** nel database (solo flare GOES).")
    except Exception as e:  # noqa: BLE001
        st.error(f"DB non leggibile: {e}")
with d:
    st.markdown("**Workspace**")
    st.code(str(s.active_workspace), language="text")
    import shutil
    free = shutil.disk_usage(s.active_workspace).free / 1e9
    st.markdown(f"Spazio libero: **{free:.1f} GB** (minimo per avviare job: {s.min_free_disk_gb} GB)")

st.divider()
with st.expander("⚙️ Configurazione della GUI", expanded=bool(s.problems())):
    with st.form("settings"):
        root = st.text_input("Cartella del pacchetto CORNERSTONE", s.cornerstone_root)
        py = st.text_input("Interprete Python di CORNERSTONE (vuoto = <root>/.venv/bin/python)", s.cornerstone_python)
        dbp = st.text_input("Database SQLite reale (vuoto = local_db_path di db_config.json)", s.db_path)
        ws = st.text_input("Workspace modalità reale", s.workspace)
        mode = st.radio("Modalità", ["real", "demo"], index=0 if s.mode == "real" else 1, horizontal=True,
                        format_func=lambda x: "Reale (dati scientifici)" if x == "real" else "Demo (dati sintetici)")
        mfree = st.number_input("Spazio disco minimo per avviare un job (GB)", 0.5, 1000.0, float(s.min_free_disk_gb))
        if st.form_submit_button("Salva configurazione"):
            s.cornerstone_root, s.cornerstone_python, s.db_path, s.workspace, s.mode, s.min_free_disk_gb = \
                root, py, dbp, ws, mode, mfree
            save_settings(s)
            ui.reset_ctx()
            st.success("Configurazione salvata in gui_settings.json")
            st.rerun()
    if s.mode == "demo":
        st.markdown("**Database demo** (sintetico, stesso schema del DB reale, solo per prova/test)")
        n = st.slider("Numero di Active Region sintetiche", 10, 120, 36)
        if st.button("Genera / rigenera DB demo"):
            stats = build_demo_db(s.demo_db_path, n_harps=n)
            dbq._tables.cache_clear()
            st.success(f"DB demo creato: {stats}")
