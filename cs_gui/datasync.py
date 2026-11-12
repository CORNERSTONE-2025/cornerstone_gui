"""Download / update of the data store through ``python -m cornerstone.data`` (background job).

The GUI starts the package command line with the CORNERSTONE interpreter, like the pipeline jobs,
in its own process group; this is the only place where the GUI writes to the data store. Actions:

- ``update``: incremental download from the sources (JSOC SHARP DEF/NRT, JSOC HARP-NOAA map,
  NOAA SWPC GOES events); the first run downloads everything (hours for SHARP);
- ``snapshot``: create the store from the published Parquet snapshot (``bootstrap_url``);
- ``import``: create the store from a database of the former MySQL dump (``bootstrap-sqlite``).

The state of the last run lives in ``<real workspace>/jobs/data_update.json``.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from .jobs import pid_alive, runner_env
from .settings import Settings
from .store import now_iso

SOURCES = {"harp_noaa": "HARP ↔ NOAA map (JSOC)", "goes": "GOES flares (NOAA SWPC)",
           "sharp_DEF": "SHARP DEF (JSOC)", "sharp_NRT": "SHARP NRT (JSOC)"}
NEED_GB = {"update": 2.0, "snapshot": 2.0, "import": 2.0}


def _jobs_dir(settings: Settings) -> Path:
    d = Path(settings.workspace) / "jobs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _state_file(settings: Settings) -> Path:
    return _jobs_dir(settings) / "data_update.json"


def status(settings: Settings) -> dict:
    out = {"data_root": settings.real_data_root, "exists": (Path(settings.real_data_root) / "manifest.json").exists(),
           "job": None, "running": False}
    try:
        job = json.loads(_state_file(settings).read_text())
    except (OSError, json.JSONDecodeError):
        return out
    alive = pid_alive(job.get("pid"))
    if job.get("status") == "running" and not alive:
        rc = _exit_code(job)
        job["status"] = "completed" if rc == 0 else "failed"
        if rc != 0:
            job.setdefault("error", "The process ended with an error. See the log." if rc is not None
                           else "The process ended unexpectedly. See the log.")
        job["finished_at"] = job.get("finished_at") or now_iso()
        _write_state(_state_file(settings), job)
    job["running"] = job.get("status") == "running" and alive
    out["job"], out["running"] = job, job["running"]
    return out


def _exit_code(job: dict) -> int | None:
    try:
        return int(Path(job["rc_file"]).read_text().strip())
    except (OSError, KeyError, ValueError):
        return None


def command(settings: Settings, action: str, sources: list[str] | None = None, full: bool = False,
            url: str | None = None, sqlite: str | None = None) -> list[str]:
    base = [settings.python, "-m", "cornerstone.data", "--root", settings.real_data_root]
    if action == "update":
        return base + ["update", "--sources", *(sources or list(SOURCES))] + (["--full"] if full else [])
    if action == "snapshot":
        return base + ["bootstrap-snapshot"] + (["--url", url] if url else [])
    if action == "import":
        if not sqlite:
            raise ValueError("import: the path of the former SQLite database is required")
        return base + ["bootstrap-sqlite", sqlite]
    raise ValueError(f"Unknown action {action!r}")


def preflight(settings: Settings, action: str) -> list[str]:
    """Problems that prevent starting (empty = ok)."""
    problems = []
    root = Path(settings.cornerstone_root)
    if not (root / "cornerstone" / "data" / "__main__.py").exists():
        problems.append(f"CORNERSTONE package with the data layer (cornerstone/data) not found in '{root}'.")
    if not Path(settings.python).exists():
        problems.append(f"CORNERSTONE Python interpreter not found: '{settings.python}'.")
    dr = Path(settings.real_data_root)
    folder = next((p for p in [dr, *dr.parents] if p.exists()), None)
    if folder is None or not os.access(folder, os.W_OK):
        problems.append(f"The data folder '{dr}' is not writable.")
    elif (free := shutil.disk_usage(folder).free / 1e9) < NEED_GB[action]:
        problems.append(f"Not enough disk space in '{folder}': {free:.1f} GB free, {NEED_GB[action]:.1f} GB needed.")
    if status(settings)["running"]:
        problems.append("A data update is already running.")
    return problems


def launch(settings: Settings, action: str, **kw) -> dict:
    problems = preflight(settings, action)
    if problems:
        raise RuntimeError("Cannot start:\n- " + "\n- ".join(problems))
    cmd = command(settings, action, **kw)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log = _jobs_dir(settings) / f"data_{action}-{stamp}.log"
    rc_file = _jobs_dir(settings) / f"data_{action}-{stamp}.rc"
    # the shell records the exit code, so a completed run is told apart from a crash
    shell = " ".join(_quote(c) for c in cmd) + f"; echo $? > {_quote(str(rc_file))}"
    with open(log, "ab") as fh:
        fh.write(f"===== [cs-gui] {now_iso()} {' '.join(cmd[2:])} =====\n".encode())
        fh.flush()
        proc = subprocess.Popen(["/bin/sh", "-c", shell], cwd=str(_jobs_dir(settings)), env=runner_env(settings),
                                stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
    job = {"pid": proc.pid, "status": "running", "action": action, "cmd": cmd, "data_root": settings.real_data_root,
           "started_at": now_iso(), "log": str(log), "rc_file": str(rc_file)}
    _write_state(_state_file(settings), job)
    return job


def cancel(settings: Settings) -> None:
    job = status(settings)["job"]
    if not job or not job["running"]:
        return
    try:
        os.killpg(job["pid"], signal.SIGTERM)   # files are written atomically: a stopped run leaves no partial file
    except ProcessLookupError:
        pass
    for _ in range(100):
        if not pid_alive(job["pid"]):
            break
        time.sleep(0.1)
    if pid_alive(job["pid"]):
        os.killpg(job["pid"], signal.SIGKILL)
    job.update(status="cancelled", finished_at=now_iso(), error="Cancelled by the user")
    job.pop("running", None)
    _write_state(_state_file(settings), job)


def _quote(s: str) -> str:
    import shlex
    return shlex.quote(s)


def _write_state(path: Path, job: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({k: v for k, v in job.items() if k != "running"}, indent=2))
    os.replace(tmp, path)
