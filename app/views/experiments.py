import json
from pathlib import Path

import pandas as pd
import streamlit as st

from cs_gui import bundle, ui
from cs_gui.store import ARTIFACT_TYPES, StoreError

c = ui.page_header("Experiments, pipeline and reproducibility",
                   "Artifact catalogue, dependency graph, integrity, variants, reproduction and resume.")

tab_cat, tab_det, tab_share, tab_jobs = st.tabs(["Catalogue", "Details and actions", "Share (export / import)", "Jobs"])

with tab_cat:
    a, b, d = st.columns(3)
    types = a.multiselect("Type", list(ARTIFACT_TYPES), list(ARTIFACT_TYPES), format_func=lambda t: ARTIFACT_TYPES[t][1])
    stats = b.multiselect("Status", ["completed", "failed", "running", "pending", "cancelled", "corrupted"],
                          ["completed", "failed", "running", "pending", "cancelled", "corrupted"])
    q = d.text_input("Search (name / id)")
    rows = []
    for m in c.store.list():
        if m["type"] not in types and m["type"] != "?" or m["status"] not in stats:
            continue
        if q and q.lower() not in (m["name"] + m["id"]).lower():
            continue
        rows.append({"id": m["id"], "type": m["type"], "name": m["name"], "status": m["status"],
                     "created": m["created_at"][:19].replace("T", " "),
                     "size [MB]": round(c.store.size_bytes(m["id"]) / 1e6, 2),
                     "depends on": ", ".join(f"{r}:{p}" for r, p in (m.get("parents") or {}).items()),
                     "variant of": m.get("derived_from") or ""})
    df = pd.DataFrame(rows)
    ev = st.dataframe(df, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
                      key="cat")
    if ev.selection.rows:
        st.session_state["exp_sel"] = df.iloc[ev.selection.rows[0]]["id"]
        st.info(f"Selected {st.session_state['exp_sel']}: open the *Details and actions* tab.")
    with st.expander("Full dependency graph"):
        if len(df):
            st.graphviz_chart(c.pipeline.lineage_dot())

with tab_det:
    ids = [m["id"] for m in c.store.list()]
    if not ids:
        st.info("No artifacts.")
        st.stop()
    cur = st.session_state.get("exp_sel")
    aid = st.selectbox("Artifact", ids, index=ids.index(cur) if cur in ids else 0,
                       format_func=lambda x: ui.label_of(c.store.get(x)) if c.store.exists(x) else x)
    m = c.store.get(aid)
    st.markdown(f"### {m['name']}  \n`{aid}` · {ARTIFACT_TYPES[m['type']][1]} · mode {m['mode']}")
    if m.get("description"):
        st.caption(m["description"])
    ui.artifact_status_box(m)
    st.graphviz_chart(c.pipeline.lineage_dot(aid))

    t1, t2, t3, t4, t5 = st.tabs(["Configuration and results", "Log", "Integrity", "Actions", "Compare versions"])
    with t1:
        st.markdown("**Configuration**")
        st.json(m["config"])
        st.markdown("**Results summary**")
        st.json(m.get("summary") or {}, expanded=False)
        st.markdown("**Provenance** (code commit and dependency versions)")
        st.json(m.get("provenance") or {}, expanded=False)
        st.markdown("**Produced files**")
        files = (m.get("outputs") or {}).get("files", {})
        st.dataframe(pd.DataFrame([{"file": k, "MB": round(v["size"] / 1e6, 3), "sha256": v.get("sha256", "")[:16]}
                                   for k, v in files.items()]), hide_index=True, width="stretch")
        st.markdown("**History**")
        st.dataframe(pd.DataFrame(m.get("history", [])), hide_index=True)
    with t2:
        ui.log_viewer(aid, 520)
    with t3:
        full = st.checkbox("Full checksum verification (slow for large files)", True)
        if st.button("Verify the integrity of the artifact and its dependencies"):
            problems = {x: c.store.verify(x, full_hash=full) for x in c.store.ancestors(aid) + [aid]}
            bad = {k: v for k, v in problems.items() if v}
            if bad:
                st.error("Problems found:")
                st.json(bad)
            else:
                st.success(f"All {len(problems)} artifacts of the chain are intact.")
        cs = (m.get("provenance") or {}).get("cornerstone") or {}
        info = ui.package_info() or {}
        now = (info.get("cornerstone") or {}).get("commit")
        if cs.get("commit") and now and cs["commit"] != now:
            st.warning(f"The package changed since the run: commit then {cs['commit'][:10]}, now {now[:10]}. "
                       "A reproduction will use the current code.")
        if cs.get("dirty"):
            st.warning("At run time the package had uncommitted local changes: "
                       "reproducibility is not guaranteed.")
        st.caption("Known limits: training on CPU/GPU with multiple threads and non-deterministic PyTorch operations "
                   "can give results that are not bit-for-bit identical even with the same seed.")
    with t4:
        st.markdown("**Reproduce / resume**")
        x, y, z = st.columns(3)
        if m["type"] != "model_config" and x.button("Reproduce this step",
                                                    help="New artifact with the same configuration and inputs"):
            try:
                new = c.pipeline.reproduce_step(aid)
                ui.launch_existing([new["id"]])
            except StoreError as e:
                st.error(str(e))
        if y.button("Reproduce the whole chain", help="New copies of all upstream steps, wired to each other"):
            try:
                new_ids = c.pipeline.reproduce_chain(aid)
                ui.launch_existing(new_ids)
            except StoreError as e:
                st.error(str(e))
        plan = c.pipeline.resume_plan(aid)
        if plan and z.button(f"Resume from the last valid artifact ({len(plan)} steps)"):
            ui.launch_existing(plan)
        if plan:
            st.caption("Steps still to run: " + " → ".join(plan))
        st.markdown("**Variant (change the configuration without touching the original)**")
        txt = st.text_area("Configuration (JSON)", json.dumps(m["config"], indent=2), height=260, key=f"var_{aid}")
        vname = st.text_input("Variant name", f"{m['name']} (variant)", key=f"vn_{aid}")
        launch = st.checkbox("Start immediately", m["type"] != "model_config", key=f"vl_{aid}")
        if st.button("Create variant"):
            try:
                new = c.pipeline.variant(aid, config=json.loads(txt), name=vname)
                st.success(f"Created variant {new['id']}")
                if launch and m["type"] != "model_config":
                    ui.launch_existing([new["id"]])
            except (StoreError, json.JSONDecodeError) as e:
                st.error(str(e))
        st.markdown("**Export**")
        ui.download_json("Export the full chain configuration (JSON)", c.pipeline.export(aid),
                         f"{aid}_chain.json", key=f"exp_{aid}")
        st.markdown("**Delete**")
        deps = c.store.descendants(aid)
        if deps:
            st.warning(f"{len(deps)} artifacts depend on this one: {deps}")
        cascade = st.checkbox("Also delete the dependent artifacts", False, disabled=not deps, key=f"cas_{aid}")
        sure = st.checkbox(f"I confirm the permanent deletion of {aid}"
                           + (" and its dependents" if cascade else ""), key=f"sure_{aid}")
        if st.button("Delete", type="secondary", disabled=not sure):
            try:
                gone = c.store.delete(aid, confirm=True, cascade=cascade)
                st.success(f"Deleted: {gone}")
                st.session_state.pop("exp_sel", None)
            except StoreError as e:
                st.error(str(e))
    with t5:
        same = [x for x in ids if x != aid and c.store.get(x)["type"] == m["type"]]
        if not same:
            st.caption("No other artifact of the same type.")
        else:
            other = st.selectbox("Compare with", same, format_func=lambda x: ui.label_of(c.store.get(x)))
            d = c.pipeline.diff(aid, other)
            st.dataframe(pd.DataFrame(d), hide_index=True, width="stretch") if d else st.success("Identical configurations.")
            sa, sb = m.get("summary") or {}, c.store.get(other).get("summary") or {}
            if m["type"] == "inference":
                comp = [{"window": w, f"TSS {aid}": (sa["windows"][w].get("metrics") or {}).get("TSS"),
                         f"TSS {other}": ((sb.get("windows") or {}).get(w, {}).get("metrics") or {}).get("TSS")}
                        for w in sa.get("windows", {})]
                st.dataframe(pd.DataFrame(comp), hide_index=True)

with tab_share:
    st.markdown("Artifacts can be moved to another installation **with their results**: whoever imports "
                "them sees labels, datasets, models and metrics without re-running the chain, and can use "
                "them as input of new steps (e.g. inference with an imported model). A bundle contains the "
                "chosen artifacts **and all their ancestors**; every file is checked (sha256) on import.")
    ex, im = st.columns(2)
    with ex:
        st.markdown("#### Export")
        done = [m for m in c.store.list() if m["status"] == "completed"]
        pick = st.multiselect("Artifacts to export", [m["id"] for m in done],
                              [st.session_state["exp_sel"]] if st.session_state.get("exp_sel") in
                              {m["id"] for m in done} else [],
                              format_func=lambda x: ui.label_of(c.store.get(x)), key="bundle_pick")
        if pick:
            todo = bundle.bundle_ids(c.store, pick)
            mb = sum(c.store.size_bytes(x) for x in todo) / 1e6
            st.caption(f"{len(todo)} artifacts with the ancestors ({mb:,.1f} MB before compression): "
                       + ", ".join(todo))
            note = st.text_input("Note for whoever receives it", "", key="bundle_note")
            if st.button("Create the bundle", type="primary"):
                out = c.store.workspace / "exports" / f"cornerstone_{pick[0]}{'_and_more' if len(pick) > 1 else ''}.zip"
                try:
                    with st.spinner("Packing…"):
                        st.session_state["bundle_out"] = bundle.export_bundle(c.store, pick, out, note)
                except StoreError as e:
                    st.error(str(e))
        res = st.session_state.get("bundle_out")
        if res and Path(res["file"]).exists():
            st.success(f"Bundle ready: `{res['file']}` ({res['size'] / 1e6:,.1f} MB)")
            if res["size"] < 2e9:
                with open(res["file"], "rb") as fh:
                    st.download_button("Download the bundle", fh, file_name=Path(res["file"]).name,
                                       mime="application/zip")
            else:
                st.caption("Too large to download from the browser: copy the file from the workspace.")
    with im:
        st.markdown("#### Import")
        up = st.file_uploader("Bundle (.zip)", type=["zip"], key="bundle_up")
        srv = st.text_input("…or path of a bundle on this machine (e.g. in the workspace)", "", key="bundle_path")
        if (up or srv) and st.button("Import", type="primary"):
            src = None
            try:
                if up:
                    tmp_dir = c.store.workspace / "imports"
                    tmp_dir.mkdir(parents=True, exist_ok=True)
                    src = tmp_dir / up.name
                    with open(src, "wb") as fh:
                        fh.write(up.getbuffer())
                else:
                    src = Path(srv)
                info = bundle.read_bundle(src)
                with st.spinner(f"Checking and importing {len(info['artifacts'])} artifacts…"):
                    r = bundle.import_bundle(c.store, src, mode=c.settings.mode)
                st.success(f"Imported {len(r['imported'])} artifacts" + (f", {len(r['skipped'])} already present"
                                                                          if r["skipped"] else "")
                           + (f". Note: {r['note']}" if r["note"] else ""))
                st.json(r, expanded=False)
            except (StoreError, OSError, ValueError) as e:
                st.error(f"Import failed, nothing was changed: {e}")
            finally:
                if up and src is not None and src.exists():
                    src.unlink()
        st.caption("Artifacts keep their id, so their dependencies stay wired. An artifact already present "
                   "with the same contents is skipped; same id with different contents refuses the import. "
                   "Demo and real artifacts never mix. With Docker, large bundles can also be copied into the "
                   "container (`docker compose cp file.zip gui:/workspace/real/`) and imported by path.")

with tab_jobs:
    ui.job_monitor()
    jobs = c.jobs.list()
    if jobs:
        st.dataframe(pd.DataFrame([{"job": j["id"], "active": j["alive"], "artifacts": ", ".join(j["artifacts"]),
                                    "started": j["started_at"][:19]} for j in jobs]), hide_index=True, width="stretch")
        jid = st.selectbox("Job log", [j["id"] for j in jobs])
        ui.log_box(c.jobs.get(jid)["log"], key=f"job_{jid}")
