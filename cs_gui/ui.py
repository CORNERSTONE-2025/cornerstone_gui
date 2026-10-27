"""Shared Streamlit components (context, pickers, job monitor, summaries)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from .jobs import JobManager
from .pipeline import Pipeline
from .settings import Settings, load_settings
from .store import ARTIFACT_TYPES, ArtifactStore, StoreError

STATUS_STYLE = {"completed": ("green", "completato"), "failed": ("red", "fallito"), "running": ("blue", "in esecuzione"),
                "pending": ("gray", "in coda"), "cancelled": ("orange", "annullato"), "draft": ("gray", "bozza"),
                "corrupted": ("red", "manifest corrotto")}


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
        st.error(f"Impossibile leggere le funzionalità del pacchetto CORNERSTONE: {e}")
        return None


# --------------------------------------------------------------------------- chrome
def page_header(title: str, help_text: str = "") -> Ctx:
    c = ctx()
    if c.settings.mode == "demo":
        st.warning("**MODALITÀ DEMO** — database sintetico generato per prova: i risultati NON hanno valore "
                   "scientifico. Gli artefatti demo sono in un workspace separato.", icon="🧪")
    st.title(title)
    if help_text:
        st.caption(help_text)
    problems = c.settings.problems()
    if problems:
        st.error("Configurazione incompleta:\n\n- " + "\n- ".join(problems) + "\n\nVai alla **Home** per sistemarla.")
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
        st.info(f"Nessun artefatto disponibile per: {names}. Crealo nella pagina corrispondente.")
        return None
    ids = [m["id"] for m in items]
    opts = ([None] if allow_none else []) + ids
    idx = opts.index(default) if default in opts else 0
    by = {m["id"]: m for m in items}
    return st.selectbox(label, opts, index=idx, key=key, help=help,
                        format_func=lambda a: "— nessuno —" if a is None else label_of(by[a]))


def show_errors(errs: list[str]) -> bool:
    if errs:
        st.error("Correggi i parametri prima di procedere:\n\n- " + "\n- ".join(errs))
        return False
    return True


def config_summary(title: str, cfg: dict, extra: dict | None = None) -> None:
    with st.container(border=True):
        st.markdown(f"**Riepilogo — {title}**")
        rows = [{"parametro": k, "valore": json.dumps(v) if isinstance(v, (list, dict)) else str(v)}
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
            st.success(f"Avviato **{m['id']}** in background. Puoi cambiare pagina: il job continua.")
        except StoreError as e:
            st.error(f"Artefatto {m['id']} creato ma non avviato:\n\n{e}")
    st.session_state["last_created"] = m["id"]
    return m


def launch_existing(ids: list[str]) -> None:
    try:
        ctx().jobs.launch(ids)
        st.success(f"Job avviato per: {', '.join(ids)}")
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
        return "(nessun log ancora)"
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
        st.caption("Nessun job in esecuzione per questa sezione.")
        return
    for m in active:
        with st.container(border=True):
            cols = st.columns([4, 2, 1])
            cols[0].markdown(f"**{m['name']}** · `{m['id']}` · {badge(m['status'])} · ⏱ {_elapsed(m)}")
            job_id = (m.get("run") or {}).get("job_id")
            if job_id and cols[2].button("Annulla", key=f"cancel_{m['id']}"):
                c.jobs.cancel(job_id)
                st.rerun()
            with st.expander("Log in tempo reale", expanded=m["status"] == "running"):
                st.code(tail(c.store.log_path(m["id"]), 25), language="text")


def artifact_status_box(m: dict) -> None:
    run = m.get("run") or {}
    st.markdown(f"{badge(m['status'])} · creato {m['created_at'][:19]} · durata {_elapsed(m) or '—'}")
    if m["status"] == "failed":
        st.error(f"**Errore:** {run.get('error')}")
        if run.get("traceback"):
            with st.expander("Traceback"):
                st.code(run["traceback"], language="text")
    if m["status"] == "cancelled":
        st.warning(run.get("error") or "Annullato")


def log_viewer(aid: str, lines: int = 300) -> None:
    st.code(tail(ctx().store.log_path(aid), lines), language="text")


def artifact_list(types, columns_fn=None, key: str = "list") -> str | None:
    """Table of artifacts of the given type(s) + selection; returns selected id."""
    c = ctx()
    types = [types] if isinstance(types, str) else types
    items = [m for t in types for m in c.store.list(t)]
    if not items:
        st.caption("Nessun artefatto.")
        return None
    rows = []
    for m in items:
        r = {"id": m["id"], "nome": m["name"], "stato": STATUS_STYLE.get(m["status"], ("", m["status"]))[1],
             "creato": m["created_at"][:19].replace("T", " ")}
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
    return st.selectbox("Apri il dettaglio (o seleziona una riga della tabella)", [None] + ids,
                        index=ids.index(picked) + 1 if picked else 0, key=f"open_{key}",
                        format_func=lambda a: "—" if a is None else label_of(by[a]))


def download_json(label: str, data: dict, file_name: str, key: str) -> None:
    st.download_button(label, json.dumps(data, indent=2, default=str), file_name=file_name,
                       mime="application/json", key=key)


def wait_hint() -> None:
    st.caption("I processi lunghi girano in background: la pagina si aggiorna da sola ogni pochi secondi.")


def sleep_rerun(seconds: float = 0.5) -> None:
    time.sleep(seconds)
    st.rerun()
