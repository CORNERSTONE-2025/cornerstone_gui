"""Background execution of pipeline steps.

A *job* runs one or more artifacts (in order) through ``python -m cs_gui.runner`` using the
CORNERSTONE virtualenv interpreter, in its own process group, so the GUI never blocks and the
heavy scientific dependencies (torch, cupy, ...) are not needed in the GUI environment.

Job records live in ``<workspace>/jobs/<job_id>.json``. The runner owns the manifests of the
artifacts it executes; the GUI only reads them, except when cancelling.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from .settings import PROJECT_ROOT, Settings
from .store import ArtifactStore, StoreError, now_iso

RUNNABLE = ("labels", "dataset", "split", "training", "inference")


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        # reap our own finished children (jobs launched by this process) so they do not linger as zombies
        if os.waitpid(pid, os.WNOHANG)[0] == pid:
            return False
    except ChildProcessError:
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # a zombie child still answers to kill(0): check /proc
    try:
        state = Path(f"/proc/{pid}/stat").read_text().split()[2]
        return state != "Z"
    except OSError:
        return True


def runner_env(settings: Settings) -> dict:
    env = dict(os.environ)
    paths = [str(PROJECT_ROOT), settings.cornerstone_root]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env.update({
        "PYTHONPATH": os.pathsep.join(paths),
        "PYTHONUNBUFFERED": "1",
        "CLEARML_OFFLINE_MODE": "1",   # never push runs to app.clear.ml from the GUI
        "MPLBACKEND": "Agg",
    })
    env.pop("VIRTUAL_ENV", None)
    return env


class JobManager:
    def __init__(self, settings: Settings, store: ArtifactStore):
        self.settings = settings
        self.store = store
        self.dir = store.workspace / "jobs"
        self.dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ launch
    def preflight(self, artifact_ids: list[str]) -> list[str]:
        """Problems that prevent launching (empty = ok)."""
        problems = list(self.settings.problems())
        free_gb = shutil.disk_usage(self.store.workspace).free / 1e9
        if free_gb < self.settings.min_free_disk_gb:
            problems.append(f"Not enough disk space: {free_gb:.1f} GB free "
                            f"(minimum {self.settings.min_free_disk_gb} GB)")
        for aid in artifact_ids:
            m = self.store.get(aid)
            if m["type"] not in RUNNABLE:
                problems.append(f"{aid}: type '{m['type']}' is not runnable")
            if m["status"] == "running":
                problems.append(f"{aid} is already running")
            if m["status"] == "completed":
                problems.append(f"{aid} is already completed: create a new variant to run it again")
        return problems

    def launch(self, artifact_ids: list[str]) -> dict:
        problems = self.preflight(artifact_ids)
        if problems:
            raise StoreError("Cannot start the job:\n- " + "\n- ".join(problems))
        job_id = f"job-{time.strftime('%Y%m%d-%H%M%S')}-{os.urandom(2).hex()}"
        log = self.dir / f"{job_id}.log"
        cmd = [self.settings.python, "-m", "cs_gui.runner",
               "--workspace", str(self.store.workspace),
               "--db", self.settings.active_db_path,
               "--mode", self.settings.mode,
               "--job", job_id, *artifact_ids]
        for aid in artifact_ids:
            # failed/cancelled steps go back to pending (history keeps the previous attempts)
            self.store.set_status(aid, "pending", job_id=job_id, queued_at=now_iso(),
                                  error=None, traceback=None)
        with open(log, "ab") as fh:
            proc = subprocess.Popen(cmd, cwd=str(self.store.workspace), env=runner_env(self.settings),
                                    stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        job = {"id": job_id, "pid": proc.pid, "artifacts": artifact_ids, "cmd": cmd,
               "started_at": now_iso(), "log": str(log)}
        (self.dir / f"{job_id}.json").write_text(json.dumps(job, indent=2))
        return job

    # ------------------------------------------------------------------ query
    def get(self, job_id: str) -> dict:
        return json.loads((self.dir / f"{job_id}.json").read_text())

    def list(self) -> list[dict]:
        jobs = []
        for p in sorted(self.dir.glob("job-*.json"), reverse=True):
            try:
                j = json.loads(p.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            j["alive"] = pid_alive(j.get("pid"))
            jobs.append(j)
        return jobs

    def active(self) -> list[dict]:
        return [j for j in self.list() if j["alive"]]

    # ------------------------------------------------------------------ control
    def cancel(self, job_id: str) -> None:
        job = self.get(job_id)
        if pid_alive(job["pid"]):
            try:
                os.killpg(job["pid"], signal.SIGTERM)
            except ProcessLookupError:
                pass
            for _ in range(50):
                if not pid_alive(job["pid"]):
                    break
                time.sleep(0.1)
            if pid_alive(job["pid"]):
                try:
                    os.killpg(job["pid"], signal.SIGKILL)
                except ProcessLookupError:
                    pass
                for _ in range(50):
                    if not pid_alive(job["pid"]):
                        break
                    time.sleep(0.1)
        for aid in job["artifacts"]:
            if self.store.exists(aid) and self.store.get(aid)["status"] in ("pending", "running"):
                self.store.set_status(aid, "cancelled", finished_at=now_iso(),
                                      error="Cancelled by the user")

    def reconcile(self) -> list[str]:
        """Mark as failed the artifacts whose job process died without updating them."""
        fixed = []
        for m in self.store.list():
            if m.get("status") not in ("pending", "running"):
                continue
            job_id = (m.get("run") or {}).get("job_id")
            if not job_id or not (self.dir / f"{job_id}.json").exists():
                continue
            if not pid_alive(self.get(job_id).get("pid")):
                self.store.set_status(m["id"], "failed", finished_at=now_iso(),
                                      error="The process ended unexpectedly "
                                            "(crash, kill or restart). See the job log.")
                fixed.append(m["id"])
        return fixed
