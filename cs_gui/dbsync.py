"""Download / update of the real database through ``Cornerstone.db_update`` (background job).

    python -m cs_gui.dbsync --db DB_PATH --state STATE_JSON [--cornerstone-root ROOT]

The GUI starts it with the CORNERSTONE interpreter, like the pipeline jobs, in its own process
group. ``db_update`` downloads the full dump from Google Drive when ``DB_PATH`` does not exist
(or a newer full dump is published), rebuilds the SQLite database and applies the incremental
updates. Files are written next to ``DB_PATH``: this is the only place where the GUI writes to
the database folder. The state of the last run lives in ``<real workspace>/jobs/db_update.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from .jobs import pid_alive, runner_env
from .settings import Settings
from .store import now_iso

# free space needed next to the database for a first download: dump (1.5 GB) + database (4.3 GB)
FULL_DOWNLOAD_GB = 7.0
UPDATE_GB = 0.5


def _jobs_dir(settings: Settings) -> Path:
    d = Path(settings.workspace) / "jobs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _state_file(settings: Settings) -> Path:
    return _jobs_dir(settings) / "db_update.json"


def last_sync(db_path: str) -> str | None:
    """Date of the remote version the database was last synchronized to (written by db_update)."""
    meta = Path(db_path).parent / "local_meta" / "local_db_metadata.json"
    try:
        return json.loads(meta.read_text()).get("last_sync") or None
    except (OSError, json.JSONDecodeError):
        return None


def status(settings: Settings) -> dict:
    db = settings.real_db_path
    out = {"db_path": db, "exists": Path(db).exists(), "last_sync": last_sync(db), "job": None, "running": False}
    if out["exists"]:
        out["size_gb"] = Path(db).stat().st_size / 1e9
    try:
        job = json.loads(_state_file(settings).read_text())
    except (OSError, json.JSONDecodeError):
        return out
    job["running"] = job.get("status") == "running" and pid_alive(job.get("pid"))
    if job.get("status") == "running" and not job["running"]:
        # the process died without recording its end (kill, container restart)
        job["status"] = "failed"
        job.setdefault("error", "The process ended unexpectedly. See the log.")
    out["job"], out["running"] = job, job["running"]
    return out


def preflight(settings: Settings) -> list[str]:
    """Problems that prevent starting a download (empty = ok)."""
    problems = []
    root = Path(settings.cornerstone_root)
    if not (root / "cornerstone" / "__init__.py").exists():
        problems.append(f"CORNERSTONE package not found in '{root}'.")
    if not Path(settings.python).exists():
        problems.append(f"CORNERSTONE Python interpreter not found: '{settings.python}'.")
    db = Path(settings.real_db_path)
    folder = next((p for p in [db.parent, *db.parent.parents] if p.exists()), None)
    if folder is None or not os.access(folder, os.W_OK):
        problems.append(f"The database folder '{db.parent}' is not writable"
                        + (" (with Docker: see 'Using a database you already have' in docs/DOCKER.md)." if folder else "."))
    elif (free := shutil.disk_usage(folder).free / 1e9) < (need := UPDATE_GB if db.exists() else FULL_DOWNLOAD_GB):
        problems.append(f"Not enough disk space in '{folder}': {free:.1f} GB free, {need:.1f} GB needed.")
    if status(settings)["running"]:
        problems.append("A database download is already running.")
    return problems


def launch(settings: Settings) -> dict:
    problems = preflight(settings)
    if problems:
        raise RuntimeError("Cannot start the download:\n- " + "\n- ".join(problems))
    log = _jobs_dir(settings) / f"db_update-{time.strftime('%Y%m%d-%H%M%S')}.log"
    cmd = [settings.python, "-m", "cs_gui.dbsync", "--db", settings.real_db_path,
           "--cornerstone-root", settings.cornerstone_root, "--state", str(_state_file(settings))]
    with open(log, "ab") as fh:
        proc = subprocess.Popen(cmd, cwd=str(_jobs_dir(settings)), env=runner_env(settings),
                                stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
    job = {"pid": proc.pid, "status": "running", "db_path": settings.real_db_path, "cmd": cmd,
           "started_at": now_iso(), "log": str(log)}
    _write_state(_state_file(settings), job)
    return job


def cancel(settings: Settings) -> None:
    job = status(settings)["job"]
    if not job or not job["running"]:
        return
    try:
        os.killpg(job["pid"], signal.SIGTERM)    # db_update removes its partial database on SIGTERM
    except ProcessLookupError:
        pass
    for _ in range(100):
        if not pid_alive(job["pid"]):
            break
        time.sleep(0.1)
    if pid_alive(job["pid"]):
        os.killpg(job["pid"], signal.SIGKILL)


def _write_state(path: Path, job: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(job, indent=2))
    os.replace(tmp, path)


def _update_state(path: Path, **fields) -> None:
    try:
        job = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        job = {}
    job.update(fields)
    _write_state(path, job)


def run(db: str, cornerstone_root: str, state: Path) -> int:
    # SIGTERM (Cancel) becomes an exception, so db_update removes its partial files
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    sys.path.insert(0, cornerstone_root)
    t0 = time.time()
    tmp = state.parent / "db_update_tmp"
    print(f"===== [cs-gui] {now_iso()} database download/update: {db} =====", flush=True)
    try:
        from cornerstone import Cornerstone
        cs = Cornerstone({"local_db_path": db}, output_folder=str(tmp), project_name="cornerstone-gui")
        cs.db_update()
    except BaseException as e:  # noqa: BLE001 - every failure must be recorded
        import traceback
        traceback.print_exc()
        cancelled = isinstance(e, SystemExit) and e.code == 143
        _update_state(state, status="cancelled" if cancelled else "failed", finished_at=now_iso(),
                      error="Cancelled by the user" if cancelled else f"{type(e).__name__}: {e}")
        return 143 if cancelled else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    _update_state(state, status="completed", finished_at=now_iso(), duration_s=round(time.time() - t0, 1),
                  last_sync=last_sync(db))
    print(f"===== [cs-gui] completed in {time.time() - t0:.0f}s =====", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="cs_gui.dbsync")
    ap.add_argument("--db", required=True)
    ap.add_argument("--cornerstone-root", default=None)
    ap.add_argument("--state", required=True)
    a = ap.parse_args(argv)
    root = a.cornerstone_root or next((p for p in sys.path if (Path(p) / "cornerstone" / "__init__.py").exists()), "")
    return run(os.path.abspath(a.db), root, Path(a.state))


if __name__ == "__main__":
    sys.exit(main())
