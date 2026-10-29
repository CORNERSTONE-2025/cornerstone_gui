"""Artifact store: persistent, versioned intermediate products of the pipeline.

Layout::

    <workspace>/artifacts/<artifact_id>/
        manifest.json      # identity, config, parents, status, outputs (+sha256), summary, provenance
        logs/run.log       # stdout/stderr of the step
        ...                # step outputs (HDF5, SQLite, models, predictions, ...)

Rules enforced here:
- an artifact's ``config`` and ``parents`` are frozen once it leaves the ``draft``/``pending`` state;
- completed artifacts are never overwritten: changing a parameter creates a *new* artifact
  (``derived_from`` keeps the link to the original);
- deletion is refused while other artifacts depend on it, unless explicitly forced.

Only the standard library is used, so the module is importable from both virtualenvs.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import secrets
import shutil
from pathlib import Path
from typing import Iterable

# artifact type -> (id prefix, human readable name)
ARTIFACT_TYPES = {
    "labels": ("lbl", "Label (labeling)"),
    "dataset": ("ds", "HDF5 dataset (extraction)"),
    "split": ("spl", "Filtered and split dataset"),
    "model_config": ("mdl", "Model configuration"),
    "training": ("trn", "Training"),
    "inference": ("inf", "Inference"),
}
# step artifact type -> required parent roles
PARENT_ROLES = {
    "labels": [],
    "dataset": ["labels"],
    "split": ["dataset"],
    "model_config": [],
    "training": ["split", "model_config"],
    "inference": ["training", "source"],
}
STATUSES = ("draft", "pending", "running", "completed", "failed", "cancelled")
FINAL_OK = "completed"
EDITABLE = ("draft", "pending")


class StoreError(Exception):
    """Raised on invalid store operations (missing artifacts, forbidden overwrite, ...)."""


def now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _atomic_write_json(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    os.replace(tmp, path)


class ArtifactStore:
    def __init__(self, workspace: str | Path):
        self.workspace = Path(workspace)
        self.root = self.workspace / "artifacts"
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ paths
    def path(self, artifact_id: str) -> Path:
        return self.root / artifact_id

    def manifest_path(self, artifact_id: str) -> Path:
        return self.path(artifact_id) / "manifest.json"

    def log_path(self, artifact_id: str) -> Path:
        return self.path(artifact_id) / "logs" / "run.log"

    # ------------------------------------------------------------------ create / read
    def new_id(self, artifact_type: str) -> str:
        prefix = ARTIFACT_TYPES[artifact_type][0]
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{prefix}-{stamp}-{secrets.token_hex(2)}"

    def create(
        self,
        artifact_type: str,
        name: str,
        config: dict,
        parents: dict | None = None,
        description: str = "",
        derived_from: str | None = None,
        status: str = "pending",
        mode: str = "real",
    ) -> dict:
        if artifact_type not in ARTIFACT_TYPES:
            raise StoreError(f"Unknown artifact type: {artifact_type}")
        if status not in EDITABLE + (FINAL_OK,):
            raise StoreError(f"Invalid initial status: {status}")
        parents = parents or {}
        missing = [r for r in PARENT_ROLES[artifact_type] if r not in parents]
        if missing:
            raise StoreError(f"Missing dependencies for '{artifact_type}': {missing}")
        for role, pid in parents.items():
            if not self.exists(pid):
                raise StoreError(f"Parent artifact '{pid}' ({role}) not found")
        if not name.strip():
            raise StoreError("The artifact name is required")

        aid = self.new_id(artifact_type)
        d = self.path(aid)
        (d / "logs").mkdir(parents=True)
        ts = now_iso()
        manifest = {
            "id": aid,
            "type": artifact_type,
            "name": name.strip(),
            "description": description,
            "created_at": ts,
            "updated_at": ts,
            "status": status,
            "mode": mode,
            "parents": parents,
            "derived_from": derived_from,
            "config": config,
            "outputs": {"files": {}},
            "summary": {},
            "provenance": {},
            "run": {},
            "history": [{"at": ts, "event": "created", "status": status}],
        }
        _atomic_write_json(self.manifest_path(aid), manifest)
        return manifest

    def exists(self, artifact_id: str) -> bool:
        return self.manifest_path(artifact_id).exists()

    def get(self, artifact_id: str) -> dict:
        p = self.manifest_path(artifact_id)
        if not p.exists():
            raise StoreError(f"Artifact '{artifact_id}' not found")
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError as e:
            raise StoreError(f"Corrupted manifest for '{artifact_id}': {e}") from e

    def list(self, artifact_type: str | None = None, status: str | None = None) -> list[dict]:
        out = []
        for p in sorted(self.root.glob("*/manifest.json")):
            try:
                m = json.loads(p.read_text())
            except (OSError, json.JSONDecodeError):
                out.append({"id": p.parent.name, "type": "?", "name": p.parent.name, "status": "corrupted",
                            "created_at": "", "parents": {}, "config": {}, "summary": {}})
                continue
            if artifact_type and m.get("type") != artifact_type:
                continue
            if status and m.get("status") != status:
                continue
            out.append(m)
        out.sort(key=lambda m: m.get("created_at", ""), reverse=True)
        return out

    # ------------------------------------------------------------------ updates
    def update(self, artifact_id: str, **fields) -> dict:
        """Update manifest fields. ``config``/``parents`` are frozen after the editable states."""
        m = self.get(artifact_id)
        frozen = m["status"] not in EDITABLE
        for key in ("config", "parents", "type", "id", "created_at"):
            if key in fields and (frozen or key in ("type", "id", "created_at")):
                raise StoreError(f"'{key}' cannot be changed for artifact {artifact_id} (status {m['status']})")
        new_status = fields.get("status")
        if new_status and new_status not in STATUSES:
            raise StoreError(f"Invalid status: {new_status}")
        if m["status"] == FINAL_OK and new_status and new_status != FINAL_OK:
            raise StoreError(f"Artifact {artifact_id} is completed: its status cannot change")
        m.update(fields)
        m["updated_at"] = now_iso()
        if new_status:
            m.setdefault("history", []).append({"at": m["updated_at"], "event": "status", "status": new_status})
        _atomic_write_json(self.manifest_path(artifact_id), m)
        return m

    def set_status(self, artifact_id: str, status: str, **run_fields) -> dict:
        m = self.get(artifact_id)
        run = dict(m.get("run") or {})
        run.update(run_fields)
        return self.update(artifact_id, status=status, run=run)

    def register_outputs(self, artifact_id: str, hash_files: bool = True) -> dict:
        """Scan the artifact folder and record every output file with size (and sha256)."""
        d = self.path(artifact_id)
        files = {}
        for p in sorted(d.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(d).as_posix()
            if rel == "manifest.json" or rel.startswith("logs/") or ".tmp" in p.name:
                continue
            entry = {"size": p.stat().st_size}
            if hash_files:
                entry["sha256"] = sha256_file(p)
            files[rel] = entry
        m = self.get(artifact_id)
        outputs = dict(m.get("outputs") or {})
        outputs["files"] = files
        return self.update(artifact_id, outputs=outputs)

    # ------------------------------------------------------------------ lineage
    def ancestors(self, artifact_id: str) -> list[str]:
        """All ancestors in topological order (roots first), artifact excluded."""
        seen: dict[str, None] = {}

        def visit(aid: str) -> None:
            for pid in self.get(aid).get("parents", {}).values():
                if pid not in seen and self.exists(pid):
                    visit(pid)
                    seen[pid] = None

        visit(artifact_id)
        return list(seen)

    def dependents(self, artifact_id: str) -> list[dict]:
        """Artifacts that list ``artifact_id`` among their parents (direct children)."""
        return [m for m in self.list() if artifact_id in (m.get("parents") or {}).values()]

    def descendants(self, artifact_id: str) -> list[str]:
        out, todo = [], [artifact_id]
        while todo:
            cur = todo.pop()
            for child in self.dependents(cur):
                if child["id"] not in out:
                    out.append(child["id"])
                    todo.append(child["id"])
        return out

    # ------------------------------------------------------------------ integrity
    def verify(self, artifact_id: str, full_hash: bool = True) -> list[str]:
        """Return a list of integrity problems (empty list = artifact intact)."""
        problems = []
        try:
            m = self.get(artifact_id)
        except StoreError as e:
            return [str(e)]
        for role, pid in (m.get("parents") or {}).items():
            if not self.exists(pid):
                problems.append(f"Missing dependency: {role} → {pid}")
            elif self.get(pid).get("status") != FINAL_OK:
                problems.append(f"Dependency not completed: {role} → {pid} ({self.get(pid).get('status')})")
        if m.get("status") == FINAL_OK:
            d = self.path(artifact_id)
            for rel, info in (m.get("outputs") or {}).get("files", {}).items():
                p = d / rel
                if not p.exists():
                    problems.append(f"Missing file: {rel}")
                    continue
                if p.stat().st_size != info.get("size"):
                    problems.append(f"Dimensione cambiata: {rel}")
                elif full_hash and info.get("sha256") and sha256_file(p) != info["sha256"]:
                    problems.append(f"Checksum mismatch (file modified or corrupted): {rel}")
        return problems

    def size_bytes(self, artifact_id: str) -> int:
        return sum(p.stat().st_size for p in self.path(artifact_id).rglob("*") if p.is_file())

    # ------------------------------------------------------------------ delete
    def delete(self, artifact_id: str, confirm: bool = False, cascade: bool = False) -> list[str]:
        """Delete an artifact. Refuses if dependents exist unless ``cascade``; always needs ``confirm``."""
        if not confirm:
            raise StoreError("Deletion not confirmed")
        m = self.get(artifact_id)
        if m.get("status") == "running":
            raise StoreError("Cannot delete a running artifact: cancel the job first")
        victims = [artifact_id]
        desc = self.descendants(artifact_id)
        if desc and not cascade:
            raise StoreError(f"Other artifacts depend on {artifact_id}: {desc}. "
                             "Delete them first or use cascade deletion.")
        victims += desc
        for aid in reversed(victims):
            if self.exists(aid) and self.get(aid).get("status") == "running":
                raise StoreError(f"{aid} is running: cancel the job first")
        for aid in reversed(victims):
            shutil.rmtree(self.path(aid), ignore_errors=False)
        return victims

    def iter_files(self, artifact_id: str) -> Iterable[Path]:
        return (p for p in self.path(artifact_id).rglob("*") if p.is_file())
