"""Data update jobs (cs_gui.datasync) with a stub ``cornerstone.data`` command line: no network."""
import json
import sys
import time
from pathlib import Path

import pytest

from cs_gui import datasync
from cs_gui.settings import Settings

STUB = '''
import json, os, sys, time
args = sys.argv[1:]
root = args[args.index("--root") + 1]
behaviour = open(os.path.join(os.path.dirname(root), "behaviour")).read().strip()
print("stub cornerstone.data", " ".join(args[2:]), behaviour, flush=True)
if behaviour == "fail":
    sys.exit("JSOC not reachable")
if behaviour == "hang":
    time.sleep(60)
os.makedirs(root, exist_ok=True)
json.dump({"datasets": {"sharp/DEF": {"updated_at": "2026-11-12T10:00:00Z", "source": {"kind": "jsoc"},
           "files": {"harp_000000.parquet": {"rows": 10, "bytes": 2000, "max_t_obs": "2026-11-11T23:48:00Z"}}}}},
          open(os.path.join(root, "manifest.json"), "w"))
'''


@pytest.fixture
def env(tmp_path):
    root = tmp_path / "pkg"
    (root / "cornerstone" / "data").mkdir(parents=True)
    (root / "cornerstone" / "__init__.py").write_text("")
    (root / "cornerstone" / "data" / "__init__.py").write_text("")
    (root / "cornerstone" / "data" / "__main__.py").write_text(STUB)
    data = tmp_path / "data"
    data.mkdir()
    s = Settings(cornerstone_root=str(root), cornerstone_python=sys.executable,
                 data_root=str(data / "datastore"), workspace=str(tmp_path / "ws"), mode="real")
    return s, data


def wait_done(s, timeout=30):
    t0 = time.time()
    while datasync.status(s)["running"] and time.time() - t0 < timeout:
        time.sleep(0.1)
    return datasync.status(s)


def test_commands(env):
    s, _ = env
    assert datasync.command(s, "update", sources=["goes"], full=True)[-4:] == ["update", "--sources", "goes", "--full"]
    assert datasync.command(s, "snapshot", url="U")[-3:] == ["bootstrap-snapshot", "--url", "U"]
    assert datasync.command(s, "import", sqlite="/x/db.sqlite")[-2:] == ["bootstrap-sqlite", "/x/db.sqlite"]
    with pytest.raises(ValueError):
        datasync.command(s, "import")


def test_update_completes_and_store_is_read(env):
    from cs_gui import datastore
    s, data = env
    (data / "behaviour").write_text("ok")
    assert datasync.status(s)["exists"] is False and not s.has_data()
    datasync.launch(s, "update", sources=["sharp_DEF"])
    st = wait_done(s)
    assert st["job"]["status"] == "completed" and st["exists"] and s.has_data()
    assert "stub cornerstone.data update --sources sharp_DEF ok" in Path(st["job"]["log"]).read_text()
    t = datastore.status(s.real_data_root)
    assert t.iloc[0]["dataset"] == "SHARP DEF" and t.iloc[0]["rows"] == 10 and t.iloc[0]["data until"] == "2026-11-11T23:48"


def test_failure_is_recorded(env):
    s, data = env
    (data / "behaviour").write_text("fail")
    datasync.launch(s, "snapshot")
    st = wait_done(s)
    assert st["job"]["status"] == "failed" and "error" in st["job"]
    assert "JSOC not reachable" in Path(st["job"]["log"]).read_text()


def test_cancel_and_single_run(env):
    s, data = env
    (data / "behaviour").write_text("hang")
    datasync.launch(s, "update")
    t0 = time.time()
    while "stub cornerstone.data" not in Path(datasync.status(s)["job"]["log"]).read_text() and time.time() - t0 < 30:
        time.sleep(0.1)
    assert "A data update is already running." in datasync.preflight(s, "update")
    datasync.cancel(s)
    st = datasync.status(s)
    assert not st["running"] and st["job"]["status"] == "cancelled"
    assert json.loads((Path(s.workspace) / "jobs" / "data_update.json").read_text())["status"] == "cancelled"


def test_preflight_problems(env, monkeypatch):
    s, _ = env
    monkeypatch.setitem(datasync.NEED_GB, "update", 1e9)
    assert any("Not enough disk space" in p for p in datasync.preflight(s, "update"))
    s.cornerstone_root = "/nonexistent"
    assert any("data layer" in p for p in datasync.preflight(s, "update"))
