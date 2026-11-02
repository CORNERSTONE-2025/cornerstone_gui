"""Variants, reproduction, resume, diff/export and job bookkeeping (no package needed)."""
import json
import subprocess
import sys

import pytest

from cs_gui.jobs import JobManager
from cs_gui.pipeline import Pipeline
from cs_gui.store import StoreError


def chain(store):
    a = store.create("labels", "L", {"source": "existing"}, status="completed")
    b = store.create("dataset", "D", {"window_hours": 24}, parents={"labels": a["id"]}, status="completed")
    c = store.create("split", "S", {"N_FOLDS": 3}, parents={"dataset": b["id"]})
    return a, b, c


def test_reproduce_chain_wires_new_artifacts(store, settings):
    a, b, c = chain(store)
    pl = Pipeline(settings, store)
    new = pl.reproduce_chain(c["id"])
    assert len(new) == 3
    n_lab, n_ds, n_sp = (store.get(x) for x in new)
    assert n_ds["parents"]["labels"] == n_lab["id"] and n_sp["parents"]["dataset"] == n_ds["id"]
    assert n_sp["derived_from"] == c["id"] and n_sp["config"] == c["config"]
    assert store.get(c["id"])["parents"]["dataset"] == b["id"]        # originals untouched


def test_resume_plan_skips_completed(store, settings):
    a, b, c = chain(store)
    assert Pipeline(settings, store).resume_plan(c["id"]) == [c["id"]]


def test_diff_and_export(store, settings):
    a, b, c = chain(store)
    d = store.create("split", "S2", {"N_FOLDS": 5}, parents={"dataset": b["id"]})
    pl = Pipeline(settings, store)
    diff = pl.diff(c["id"], d["id"])
    assert diff == [{"key": "config.N_FOLDS", c["id"]: 3, d["id"]: 5}]
    exp = pl.export(c["id"])
    assert [x["id"] for x in exp["chain"]] == [a["id"], b["id"], c["id"]]
    json.dumps(exp)
    assert "digraph" in pl.lineage_dot(c["id"])


def test_reconcile_marks_dead_jobs_failed(store, settings):
    a, b, c = chain(store)
    jm = JobManager(settings, store)
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    (jm.dir / "job-x.json").write_text(json.dumps({"id": "job-x", "pid": p.pid, "artifacts": [c["id"]],
                                                    "started_at": "", "log": ""}))
    store.set_status(c["id"], "running", job_id="job-x")
    assert jm.reconcile() == [c["id"]]
    m = store.get(c["id"])
    assert m["status"] == "failed" and "unexpectedly" in m["run"]["error"]


def test_preflight_refuses_completed_and_bad_settings(store, settings, tmp_path):
    a, b, c = chain(store)
    settings.cornerstone_python = str(tmp_path / "nope" / "python")
    jm = JobManager(settings, store)
    problems = jm.preflight([b["id"]])
    assert any("already completed" in p for p in problems)
    assert any("interpreter" in p for p in problems)
    with pytest.raises(StoreError):
        jm.launch([c["id"]])
    assert store.get(c["id"])["status"] == "pending"                 # nothing marked as started


def test_relative_settings_paths_are_absolute(tmp_path, monkeypatch):
    from cs_gui.settings import PROJECT_ROOT, load_settings
    f = tmp_path / "gui_settings.json"
    f.write_text('{"workspace": "workspace", "demo_workspace": "workspace_demo", "db_path": "data/x.sqlite"}')
    for env in ("CS_GUI_WORKSPACE", "CS_GUI_DEMO_WORKSPACE", "CS_GUI_DB_PATH"):
        monkeypatch.delenv(env, raising=False)
    s = load_settings(f)
    assert s.workspace == str(PROJECT_ROOT / "workspace")
    assert s.db_path == str(PROJECT_ROOT / "data" / "x.sqlite")
