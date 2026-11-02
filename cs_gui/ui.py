"""Shared Streamlit components (context, pickers, job monitor, summaries)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from . import dbsync
from .jobs import JobManager
from .pipeline import Pipeline
from .settings import Settings, load_settings, save_settings
from .store import ARTIFACT_TYPES, ArtifactStore, StoreError

STATUS_STYLE = {"completed": ("green", "completed"), "failed": ("red", "failed"), "running": ("blue", "running"),
                "pending": ("gray", "queued"), "cancelled": ("orange", "cancelled"), "draft": ("gray", "draft"),
                "corrupted": ("red", "corrupted manifest")}


# --------------------------------------------------------------------------- context
class Ctx:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.store = ArtifactStore(settings.active_workspace)
        self.jobs = JobManager(settings, self.store)
        self.pipeline = Pipeline(settings, self.store)


def ctx() -> Ctx:
    if "cs_ctx" not in st.session_state:
        st.session_state.cs_ctx = Ctx(load_settings())
    c: Ctx = st.session_state.cs_ctx
    c.jobs.reconcile()
    return c


def reset_ctx() -> None:
    st.session_state.pop("cs_ctx", None)


def package_info(refresh: bool = False) -> dict | None:
    c = ctx()
    try:
        if refresh:
            c.pipeline._info = None
            return c.pipeline.package_info_refresh()
        return c.pipeline.info
    except Exception as e:  # noqa: BLE001
        st.error(f"Cannot read the features of the CORNERSTONE package: {e}")
        return None


# --------------------------------------------------------------------------- chrome
def page_header(title: str, help_text: str = "") -> Ctx:
    c = ctx()
    if c.settings.mode == "demo":
        st.warning("**DEMO MODE** — synthetic database generated for testing: the results have NO scientific "
                   "value. Demo artifacts live in a separate workspace.", icon="🧪")
    st.title(title)
    if help_text:
        st.caption(help_text)
    problems = c.settings.problems()
    if problems:
        st.error("Incomplete configuration:\n\n- " + "\n- ".join(problems) + "\n\nGo to the **Home** page to fix it.")
    return c


def badge(status: str) -> str:
    color, text = STATUS_STYLE.get(status, ("gray", status))
    return f":{color}[● {text}]"


def label_of(m: dict) -> str:
    return f"{m['name']} · {m['id']} · {STATUS_STYLE.get(m['status'], ('', m['status']))[1]}"


def artifact_picker(label: str, types, key: str, completed_only: bool = True, help: str | None = None,
                    allow_none: bool = False, default: str | None = None) -> str | None:
    c = ctx()
    types = [types] if isinstance(types, str) else list(types)
    items = [m for t in types for m in c.store.list(t) if (m["status"] == "completed" or not completed_only)]
    if not items:
        names = ", ".join(ARTIFACT_TYPES[t][1] for t in types)
        st.info(f"No artifact available for: {names}. Create one in the corresponding page.")
        return None
    ids = [m["id"] for m in items]
    opts = ([None] if allow_none else []) + ids
    idx = opts.index(default) if default in opts else 0
    by = {m["id"]: m for m in items}
    return st.selectbox(label, opts, index=idx, key=key, help=help,
                        format_func=lambda a: "— none —" if a is None else label_of(by[a]))


def show_errors(errs: list[str]) -> bool:
    if errs:
        st.error("Fix the parameters before proceeding:\n\n- " + "\n- ".join(errs))
        return False
    return True


def config_summary(title: str, cfg: dict, extra: dict | None = None) -> None:
    with st.container(border=True):
        st.markdown(f"**Summary — {title}**")
        rows = [{"parameter": k, "value": json.dumps(v) if isinstance(v, (list, dict)) else str(v)}
                for k, v in {**(extra or {}), **cfg}.items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def create_and_launch(create_fn, launch: bool = True) -> dict | None:
    """Run a creator, then launch its job; errors are shown, never swallowed."""
    c = ctx()
    try:
        m = create_fn()
    except StoreError as e:
        st.error(str(e))
        return None
    except Exception as e:  # noqa: BLE001
        st.exception(e)
        return None
    if launch:
        try:
            c.jobs.launch([m["id"]])
            st.success(f"Started **{m['id']}** in the background. You can switch page: the job keeps running.")
        except StoreError as e:
            st.error(f"Artifact {m['id']} created but not started:\n\n{e}")
    st.session_state["last_created"] = m["id"]
    return m


def launch_existing(ids: list[str]) -> None:
    try:
        ctx().jobs.launch(ids)
        st.success(f"Job started for: {', '.join(ids)}")
    except StoreError as e:
        st.error(str(e))


# --------------------------------------------------------------------------- monitoring
def _elapsed(m: dict) -> str:
    run = m.get("run") or {}
    if run.get("duration_s") is not None:
        return f"{run['duration_s']:.0f}s"
    if run.get("started_at"):
        try:
            t0 = pd.Timestamp(run["started_at"])
            return f"{(pd.Timestamp.now(tz=t0.tz) - t0).total_seconds():.0f}s"
        except (ValueError, TypeError):
            return ""
    return ""


def tail(path: Path, n: int = 60) -> str:
    if not path.exists():
        return "(no log yet)"
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 200_000))
        data = f.read().decode(errors="replace")
    lines = [l for l in data.replace("\r", "\n").splitlines() if l.strip()]
    return "\n".join(lines[-n:])


@st.fragment(run_every=3)
def job_monitor(types: list[str] | None = None) -> None:
    """Live view of the artifacts being executed (auto-refresh every 3 s)."""
    c = ctx()
    active = [m for m in c.store.list() if m["status"] in ("pending", "running")
              and (types is None or m["type"] in types)]
    if not active:
        st.caption("No job running for this section.")
        return
    for m in active:
        with st.container(border=True):
            cols = st.columns([4, 2, 1])
            cols[0].markdown(f"**{m['name']}** · `{m['id']}` · {badge(m['status'])} · ⏱ {_elapsed(m)}")
            job_id = (m.get("run") or {}).get("job_id")
            if job_id and cols[2].button("Cancel", key=f"cancel_{m['id']}"):
                c.jobs.cancel(job_id)
                st.rerun()
            with st.expander("Live log", expanded=m["status"] == "running"):
                st.code(tail(c.store.log_path(m["id"]), 25), language="text")


@st.fragment(run_every=3)
def db_download_panel() -> None:
    """Download / update of the real database (``Cornerstone.db_update``), auto-refresh every 3 s."""
    c = ctx()
    s = c.settings
    stt = dbsync.status(s)
    job = stt["job"] or {}
    if stt["exists"]:
        st.markdown(f"`{stt['db_path']}` · {stt['size_gb']:.1f} GB · synchronized to "
                    f"**{stt['last_sync'] or 'unknown date'}**")
    else:
        st.markdown(f"Not downloaded yet: it will be saved to `{stt['db_path']}` "
                    f"(about {dbsync.FULL_DOWNLOAD_GB:.0f} GB of free space needed).")
    if stt["running"]:
        st.info(f"Download/update running since {job['started_at'][:19].replace('T', ' ')}: the first "
                "download takes a while (1.5 GB from Google Drive, then the 4 GB database is built).")
        if st.button("Cancel the download", key="db_cancel"):
            dbsync.cancel(s)
            st.rerun()
    else:
        if job.get("status") == "completed":
            st.success(f"Last download/update completed ({job.get('finished_at', '')[:19].replace('T', ' ')}).")
        elif job.get("status") in ("failed", "cancelled"):
            st.error(f"Last download/update {job['status']}: {job.get('error')}")
        problems = dbsync.preflight(s)
        if problems:
            st.warning("\n\n".join(problems))
        label = "Check for updates" if stt["exists"] else "Download the database"
        if st.button(label, key="db_launch", disabled=bool(problems)):
            try:
                dbsync.launch(s)
            except RuntimeError as e:
                st.error(str(e))
            st.rerun()
        if stt["exists"] and s.mode == "demo" and st.button("Switch to Real mode", key="db_real"):
            s.mode = "real"
            save_settings(s)
            reset_ctx()
            st.rerun(scope="app")
    if job.get("log"):
        with st.expander("Download log", expanded=stt["running"]):
            st.code(tail(Path(job["log"]), 25), language="text")


def artifact_status_box(m: dict) -> None:
    run = m.get("run") or {}
    st.markdown(f"{badge(m['status'])} · created {m['created_at'][:19]} · duration {_elapsed(m) or '—'}")
    if m["status"] == "failed":
        st.error(f"**Error:** {run.get('error')}")
        if run.get("traceback"):
            with st.expander("Traceback"):
                st.code(run["traceback"], language="text")
    if m["status"] == "cancelled":
        st.warning(run.get("error") or "Cancelled")


def log_viewer(aid: str, lines: int = 300) -> None:
    st.code(tail(ctx().store.log_path(aid), lines), language="text")


def artifact_list(types, columns_fn=None, key: str = "list") -> str | None:
    """Table of artifacts of the given type(s) + selection; returns selected id."""
    c = ctx()
    types = [types] if isinstance(types, str) else types
    items = [m for t in types for m in c.store.list(t)]
    if not items:
        st.caption("No artifacts.")
        return None
    rows = []
    for m in items:
        r = {"id": m["id"], "name": m["name"], "status": STATUS_STYLE.get(m["status"], ("", m["status"]))[1],
             "created": m["created_at"][:19].replace("T", " ")}
        if columns_fn:
            try:
                r.update(columns_fn(m))
            except (KeyError, TypeError, IndexError):
                pass
        rows.append(r)
    df = pd.DataFrame(rows)
    ev = st.dataframe(df, hide_index=True, width="stretch", on_select="rerun",
                      selection_mode="single-row", key=f"tbl_{key}")
    sel = ev.selection.rows if ev and ev.selection else []
    ids = df["id"].tolist()
    picked = df.iloc[sel[0]]["id"] if sel else None
    by = {m["id"]: m for m in items}
    return st.selectbox("Open the details (or select a table row)", [None] + ids,
                        index=ids.index(picked) + 1 if picked else 0, key=f"open_{key}",
                        format_func=lambda a: "—" if a is None else label_of(by[a]))


def download_json(label: str, data: dict, file_name: str, key: str) -> None:
    st.download_button(label, json.dumps(data, indent=2, default=str), file_name=file_name,
                       mime="application/json", key=key)


def wait_hint() -> None:
    st.caption("Long processes run in the background: the page refreshes itself every few seconds.")


def sleep_rerun(seconds: float = 0.5) -> None:
    time.sleep(seconds)
    st.rerun()
