import json

import pandas as pd
import streamlit as st

from cs_gui import ui
from cs_gui.store import ARTIFACT_TYPES, StoreError

c = ui.page_header("Esperimenti, pipeline e riproducibilità",
                   "Catalogo degli artefatti, grafo delle dipendenze, integrità, varianti, riproduzione e ripresa.")

tab_cat, tab_det, tab_jobs = st.tabs(["Catalogo", "Dettaglio e azioni", "Job"])

with tab_cat:
    a, b, d = st.columns(3)
    types = a.multiselect("Tipo", list(ARTIFACT_TYPES), list(ARTIFACT_TYPES), format_func=lambda t: ARTIFACT_TYPES[t][1])
    stats = b.multiselect("Stato", ["completed", "failed", "running", "pending", "cancelled", "corrupted"],
                          ["completed", "failed", "running", "pending", "cancelled", "corrupted"])
    q = d.text_input("Cerca (nome / id)")
    rows = []
    for m in c.store.list():
        if m["type"] not in types and m["type"] != "?" or m["status"] not in stats:
            continue
        if q and q.lower() not in (m["name"] + m["id"]).lower():
            continue
        rows.append({"id": m["id"], "tipo": m["type"], "nome": m["name"], "stato": m["status"],
                     "creato": m["created_at"][:19].replace("T", " "),
                     "dimensione [MB]": round(c.store.size_bytes(m["id"]) / 1e6, 2),
                     "dipende da": ", ".join(f"{r}:{p}" for r, p in (m.get("parents") or {}).items()),
                     "variante di": m.get("derived_from") or ""})
    df = pd.DataFrame(rows)
    ev = st.dataframe(df, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
                      key="cat")
    if ev.selection.rows:
        st.session_state["exp_sel"] = df.iloc[ev.selection.rows[0]]["id"]
        st.info(f"Selezionato {st.session_state['exp_sel']}: apri la scheda *Dettaglio e azioni*.")
    with st.expander("Grafo completo delle dipendenze"):
        if len(df):
            st.graphviz_chart(c.pipeline.lineage_dot())

with tab_det:
    ids = [m["id"] for m in c.store.list()]
    if not ids:
        st.info("Nessun artefatto.")
        st.stop()
    cur = st.session_state.get("exp_sel")
    aid = st.selectbox("Artefatto", ids, index=ids.index(cur) if cur in ids else 0,
                       format_func=lambda x: ui.label_of(c.store.get(x)) if c.store.exists(x) else x)
    m = c.store.get(aid)
    st.markdown(f"### {m['name']}  \n`{aid}` · {ARTIFACT_TYPES[m['type']][1]} · modalità {m['mode']}")
    if m.get("description"):
        st.caption(m["description"])
    ui.artifact_status_box(m)
    st.graphviz_chart(c.pipeline.lineage_dot(aid))

    t1, t2, t3, t4, t5 = st.tabs(["Configurazione e risultati", "Log", "Integrità", "Azioni", "Confronta versioni"])
    with t1:
        st.markdown("**Configurazione**")
        st.json(m["config"])
        st.markdown("**Riepilogo dei risultati**")
        st.json(m.get("summary") or {}, expanded=False)
        st.markdown("**Provenienza** (commit del codice e versioni delle dipendenze)")
        st.json(m.get("provenance") or {}, expanded=False)
        st.markdown("**File prodotti**")
        files = (m.get("outputs") or {}).get("files", {})
        st.dataframe(pd.DataFrame([{"file": k, "MB": round(v["size"] / 1e6, 3), "sha256": v.get("sha256", "")[:16]}
                                   for k, v in files.items()]), hide_index=True, width="stretch")
        st.markdown("**Storia**")
        st.dataframe(pd.DataFrame(m.get("history", [])), hide_index=True)
    with t2:
        ui.log_viewer(aid, 1000)
    with t3:
        full = st.checkbox("Verifica completa dei checksum (lenta per file grandi)", True)
        if st.button("Verifica integrità dell'artefatto e delle dipendenze"):
            problems = {x: c.store.verify(x, full_hash=full) for x in c.store.ancestors(aid) + [aid]}
            bad = {k: v for k, v in problems.items() if v}
            if bad:
                st.error("Problemi trovati:")
                st.json(bad)
            else:
                st.success(f"Tutti i {len(problems)} artefatti della catena sono integri.")
        cs = (m.get("provenance") or {}).get("cornerstone") or {}
        info = ui.package_info() or {}
        now = (info.get("cornerstone") or {}).get("commit")
        if cs.get("commit") and now and cs["commit"] != now:
            st.warning(f"Il pacchetto è cambiato dall'esecuzione: commit allora {cs['commit'][:10]}, ora {now[:10]}. "
                       "Una riproduzione userà il codice attuale.")
        if cs.get("dirty"):
            st.warning("Al momento dell'esecuzione il pacchetto aveva modifiche locali non committate: "
                       "la riproducibilità non è garantita.")
        st.caption("Limiti noti: training su CPU/GPU con thread multipli e operazioni non deterministiche di PyTorch "
                   "possono dare risultati non identici bit a bit anche con lo stesso seed.")
    with t4:
        st.markdown("**Riproduci / riprendi**")
        x, y, z = st.columns(3)
        if m["type"] != "model_config" and x.button("Riproduci questo passo",
                                                    help="Nuovo artefatto con la stessa configurazione e gli stessi input"):
            try:
                new = c.pipeline.reproduce_step(aid)
                ui.launch_existing([new["id"]])
            except StoreError as e:
                st.error(str(e))
        if y.button("Riproduci l'intera catena", help="Nuove copie di tutti i passi a monte, collegate tra loro"):
            try:
                new_ids = c.pipeline.reproduce_chain(aid)
                ui.launch_existing(new_ids)
            except StoreError as e:
                st.error(str(e))
        plan = c.pipeline.resume_plan(aid)
        if plan and z.button(f"Riprendi dall'ultimo artefatto valido ({len(plan)} passi)"):
            ui.launch_existing(plan)
        if plan:
            st.caption("Passi ancora da eseguire: " + " → ".join(plan))
        st.markdown("**Variante (modifica la configurazione senza toccare l'originale)**")
        txt = st.text_area("Configurazione (JSON)", json.dumps(m["config"], indent=2), height=260, key=f"var_{aid}")
        vname = st.text_input("Nome della variante", f"{m['name']} (variante)", key=f"vn_{aid}")
        launch = st.checkbox("Avvia subito", m["type"] != "model_config", key=f"vl_{aid}")
        if st.button("Crea variante"):
            try:
                new = c.pipeline.variant(aid, config=json.loads(txt), name=vname)
                st.success(f"Creata variante {new['id']}")
                if launch and m["type"] != "model_config":
                    ui.launch_existing([new["id"]])
            except (StoreError, json.JSONDecodeError) as e:
                st.error(str(e))
        st.markdown("**Esporta**")
        ui.download_json("Esporta configurazione completa della catena (JSON)", c.pipeline.export(aid),
                         f"{aid}_chain.json", key=f"exp_{aid}")
        st.markdown("**Cancella**")
        deps = c.store.descendants(aid)
        if deps:
            st.warning(f"{len(deps)} artefatti dipendono da questo: {deps}")
        cascade = st.checkbox("Cancella anche gli artefatti dipendenti", False, disabled=not deps, key=f"cas_{aid}")
        sure = st.checkbox(f"Confermo la cancellazione definitiva di {aid}"
                           + (" e dei dipendenti" if cascade else ""), key=f"sure_{aid}")
        if st.button("Cancella", type="secondary", disabled=not sure):
            try:
                gone = c.store.delete(aid, confirm=True, cascade=cascade)
                st.success(f"Cancellati: {gone}")
                st.session_state.pop("exp_sel", None)
            except StoreError as e:
                st.error(str(e))
    with t5:
        same = [x for x in ids if x != aid and c.store.get(x)["type"] == m["type"]]
        if not same:
            st.caption("Nessun altro artefatto dello stesso tipo.")
        else:
            other = st.selectbox("Confronta con", same, format_func=lambda x: ui.label_of(c.store.get(x)))
            d = c.pipeline.diff(aid, other)
            st.dataframe(pd.DataFrame(d), hide_index=True, width="stretch") if d else st.success("Configurazioni identiche.")
            sa, sb = m.get("summary") or {}, c.store.get(other).get("summary") or {}
            if m["type"] == "inference":
                comp = [{"finestra": w, f"TSS {aid}": (sa["windows"][w].get("metrics") or {}).get("TSS"),
                         f"TSS {other}": ((sb.get("windows") or {}).get(w, {}).get("metrics") or {}).get("TSS")}
                        for w in sa.get("windows", {})]
                st.dataframe(pd.DataFrame(comp), hide_index=True)

with tab_jobs:
    ui.job_monitor()
    jobs = c.jobs.list()
    if jobs:
        st.dataframe(pd.DataFrame([{"job": j["id"], "attivo": j["alive"], "artefatti": ", ".join(j["artifacts"]),
                                    "avviato": j["started_at"][:19]} for j in jobs]), hide_index=True, width="stretch")
        jid = st.selectbox("Log del job", [j["id"] for j in jobs])
        from pathlib import Path
        st.code(ui.tail(Path(c.jobs.get(jid)["log"]), 200), language="text")
