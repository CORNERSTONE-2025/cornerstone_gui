"""Database download job (cs_gui.dbsync) with a stub ``cornerstone`` package: no network, no real DB."""
import json
import sys
import time
from pathlib import Path

import pytest

from cs_gui import dbsync
from cs_gui.settings import Settings, _db_path_from_cornerstone

STUB = '''
import json, os, sys, time
class Cornerstone:
    def __init__(self, db_config, output_folder=".", project_name=""):
        self.db_config = db_config
        os.makedirs(output_folder, exist_ok=True)
    def db_update(self):
        db = self.db_config["local_db_path"]
        behaviour = open(os.path.join(os.path.dirname(db), "behaviour")).read().strip()
        print("stub db_update", behaviour, flush=True)
        if behaviour == "fail":
            raise RuntimeError("remote folder not reachable")
        if behaviour == "hang":
            time.sleep(60)
        open(db, "w").write("db")
        meta = os.path.join(os.path.dirname(db), "local_meta")
        os.makedirs(meta, exist_ok=True)
        json.dump({"last_sync": "2025-11-05"}, open(os.path.join(meta, "local_db_metadata.json"), "w"))
'''


@pytest.fixture
def env(tmp_path):
    root = tmp_path / "pkg"
    (root / "cornerstone").mkdir(parents=True)
    (root / "cornerstone" / "__init__.py").write_text(STUB)
    dbdir = tmp_path / "dbdir"
    dbdir.mkdir()
    s = Settings(cornerstone_root=str(root), cornerstone_python=sys.executable,
                 db_path=str(dbdir / "cornerstone.sqlite"), workspace=str(tmp_path / "ws"), mode="demo")
    return s, dbdir


def wait_done(s, timeout=30):
    t0 = time.time()
    while dbsync.status(s)["running"] and time.time() - t0 < timeout:
        time.sleep(0.1)
    return dbsync.status(s)


def test_download_completes_and_records_sync(env, monkeypatch):
    s, dbdir = env
    (dbdir / "behaviour").write_text("ok")
    monkeypatch.setattr(dbsync, "FULL_DOWNLOAD_GB", 0.0)
    assert dbsync.status(s)["exists"] is False
    job = dbsync.launch(s)
    st = wait_done(s)
    assert st["job"]["status"] == "completed", Path(job["log"]).read_text()
    assert st["exists"] and st["last_sync"] == "2025-11-05"
    assert "stub db_update ok" in Path(job["log"]).read_text()


def test_download_failure_is_recorded(env, monkeypatch):
    s, dbdir = env
    (dbdir / "behaviour").write_text("fail")
    monkeypatch.setattr(dbsync, "FULL_DOWNLOAD_GB", 0.0)
    dbsync.launch(s)
    st = wait_done(s)
    assert st["job"]["status"] == "failed" and "remote folder not reachable" in st["job"]["error"]
    assert not st["exists"]


def test_cancel(env, monkeypatch):
    s, dbdir = env
    (dbdir / "behaviour").write_text("hang")
    monkeypatch.setattr(dbsync, "FULL_DOWNLOAD_GB", 0.0)
    dbsync.launch(s)
    t0 = time.time()
    while "stub db_update" not in Path(dbsync.status(s)["job"]["log"]).read_text() and time.time() - t0 < 30:
        time.sleep(0.1)
    assert "A database download is already running." in dbsync.preflight(s)
    dbsync.cancel(s)
    st = wait_done(s)
    assert st["job"]["status"] == "cancelled" and not st["running"]


def test_dead_process_is_reported_failed(env):
    s, _ = env
    (Path(s.workspace) / "jobs").mkdir(parents=True)
    (Path(s.workspace) / "jobs" / "db_update.json").write_text(json.dumps({"pid": 999999, "status": "running"}))
    st = dbsync.status(s)
    assert not st["running"] and st["job"]["status"] == "failed"


def test_preflight_disk_space_and_writable(env, monkeypatch, tmp_path):
    s, dbdir = env
    monkeypatch.setattr(dbsync, "FULL_DOWNLOAD_GB", 1e9)
    assert any("Not enough disk space" in p for p in dbsync.preflight(s))
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o500)
    try:
        s.db_path = str(ro / "cornerstone.sqlite")
        assert any("not writable" in p for p in dbsync.preflight(s))
    finally:
        ro.chmod(0o700)


def test_real_db_path_relative_to_package(tmp_path):
    cfg = tmp_path / "cornerstone" / "configs"
    cfg.mkdir(parents=True)
    (cfg / "db_config.json").write_text(json.dumps({"local_db_path": "db/cornerstone.sqlite"}))
    assert _db_path_from_cornerstone(str(tmp_path)) == str(tmp_path / "db" / "cornerstone.sqlite")
    s = Settings(cornerstone_root=str(tmp_path), mode="demo")
    assert s.real_db_path == str(tmp_path / "db" / "cornerstone.sqlite")
    assert s.active_db_path != s.real_db_path
