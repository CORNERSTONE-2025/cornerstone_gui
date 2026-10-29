"""Provenance information recorded in every executed artifact."""
from __future__ import annotations

import platform
import subprocess
import sys
from importlib import metadata

from ..settings import PROJECT_ROOT

PACKAGES = ["numpy", "pandas", "h5py", "scikit-learn", "scipy", "torch", "xgboost", "xgboost-cpu", "sktime",
            "clearml", "cupy-cuda12x"]


def git_state(path: str) -> dict:
    def run(*args):
        return subprocess.run(["git", "-C", path, *args], capture_output=True, text=True, timeout=20)
    head = run("rev-parse", "HEAD")
    if head.returncode != 0:
        return {"commit": None, "dirty": None}
    dirty = run("status", "--porcelain", "--untracked-files=no").stdout.strip()
    upstream = run("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}").stdout.strip()   # e.g. cornerstone/main
    remote_name = upstream.split("/")[0] if upstream else "origin"
    remote = run("config", "--get", f"remote.{remote_name}.url").stdout.strip()
    branch = run("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    return {"commit": head.stdout.strip(), "dirty": bool(dirty), "branch": branch,
            "dirty_files": dirty.splitlines()[:50], "remote": remote}


def collect(cornerstone_root: str) -> dict:
    versions = {}
    for p in PACKAGES:
        try:
            versions[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "host": platform.node(),
        "packages": versions,
        "cornerstone": git_state(cornerstone_root),
        "cornerstone_gui": git_state(str(PROJECT_ROOT)),
    }
