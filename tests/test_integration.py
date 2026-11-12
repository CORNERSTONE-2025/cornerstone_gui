"""End-to-end tests running the REAL CORNERSTONE package (through the runner) on the synthetic
demo data store. Slow (a few minutes): ``pytest -m integration``."""
import copy
import json
import os
import subprocess
import time

import pandas as pd
import pytest

from conftest import CORNERSTONE_ROOT, cornerstone_available, dump, run_and_wait
from cs_gui.demo import build_demo_store
from cs_gui.jobs import JobManager
from cs_gui.pipeline import DEFAULT_TRAINING_CONFIG, Pipeline, default_label_config, default_split_config
from cs_gui.settings import Settings
from cs_gui.store import ArtifactStore

pytestmark = [pytest.mark.integration,
              pytest.mark.skipif(not cornerstone_available(), reason="CORNERSTONE checkout/virtualenv not found")]


@pytest.fixture(scope="session")
def demo(tmp_path_factory):
    ws = tmp_path_factory.mktemp("demo_ws")
    s = Settings(cornerstone_root=CORNERSTONE_ROOT,
                 demo_workspace=str(ws), mode="demo", min_free_disk_gb=0.1)
    build_demo_store(s.demo_data_root, n_harps=30, seed=3)
    store = ArtifactStore(s.active_workspace)
    jobs, pl = JobManager(s, store), Pipeline(s, store)
    out = {"settings": s, "store": store, "jobs": jobs, "pl": pl}
    lab = run_and_wait(jobs, store, pl.create_labels("L", default_label_config("DEF", "C", strat_window_hours=24.0))["id"])
    assert lab["status"] == "completed", dump(lab)
    ds = run_and_wait(jobs, store, pl.create_dataset("D", lab["id"], {
        "features": "clean", "quality_source": "feature_set", "window_hours": 24, "sampling_minutes": 12,
        "subsample": False, "subsample_size": 0, "random_seed": 42})["id"])
    assert ds["status"] == "completed", dump(ds)
    cfg = default_split_config(s)
    cfg.update(DIRECT_FEATURES=ds["summary"]["feature_names"], TEMPORAL_FILTER_MODE=None, TEMPORAL_WINDOW=None,
               N_FOLDS=3, SAVE_PLOTS=False)
    sp = run_and_wait(jobs, store, pl.create_split("S", ds["id"], cfg)["id"])
    assert sp["status"] == "completed", dump(sp)
    mc = pl.create_model_config("XGB", {"model": "XGBOOST", "model_params": {"n_estimators": 30},
                                        "training_config": copy.deepcopy(DEFAULT_TRAINING_CONFIG),
                                        "regularization_params": [1e2, 1e3], "seed": 42})
    tr = run_and_wait(jobs, store, pl.create_training("T", sp["id"], mc["id"], {
        "windows": ["3h", "24h"], "window_map": {"3h": 16, "24h": 121}, "n_folds": 3, "device": "cpu"})["id"])
    assert tr["status"] == "completed", dump(tr)
    inf = run_and_wait(jobs, store, pl.create_inference("I", tr["id"], sp["id"], {
        "windows": ["3h", "24h"], "threshold_mode": "cv", "model_choice": "best", "source_file": "test",
        "exclude_overlap": True, "bootstrap": {"n": 100, "cluster": True}})["id"])
    assert inf["status"] == "completed", dump(inf)
    out.update(lab=lab, ds=ds, sp=sp, mc=mc, tr=tr, inf=inf)
    return out


def test_pipeline_outputs(demo):
    store = demo["store"]
    assert demo["lab"]["summary"]["n_rows"] > 0 and "YES" in demo["lab"]["summary"]["label_counts"]
    ds = demo["ds"]["summary"]
    assert ds["n_samples"] > 0 and ds["n_features"] == 17 and ds["timesteps"] == 121
    lk = demo["sp"]["summary"]["leakage"]
    assert lk["test_vs_trainval_shared_harps"] == 0 and lk["n_folds"] == 3
    rows = demo["tr"]["summary"]["rows"]
    assert {r["window"] for r in rows} == {"3h", "24h"} and sum(r["selected"] for r in rows) == 2
    w = demo["inf"]["summary"]["windows"]["24h"]
    assert w["threshold_source"].startswith("model.threshold_") and 0 <= w["threshold"] <= 1
    pred = pd.read_csv(store.path(demo["inf"]["id"]) / "predictions.csv")
    assert {"HARPNUM", "t_issue", "y_true", "score", "threshold", "y_pred", "in_trainval_harp"} <= set(pred.columns)
    assert not pred["in_trainval_harp"].any()          # own test set: no AR seen in training
    for a in [demo[k]["id"] for k in ("lab", "ds", "sp", "tr", "inf")]:
        assert store.verify(a) == [], a
    prov = demo["tr"]["provenance"]
    assert prov["cornerstone"]["commit"] and "torch" in prov["packages"]


def test_data_store_is_never_modified(demo):
    s = demo["settings"]
    files = sorted(p.relative_to(s.demo_data_root) for p in s.demo_data_root.rglob("*") if p.is_file())
    assert all(str(f).startswith(("raw/", "manifest.json", "DEMO_INFO")) for f in files), files
    lab = demo["store"].path(demo["lab"]["id"])
    assert (lab / "labels.parquet").exists() and (lab / "flares.parquet").exists()   # outputs in the artifact
    ds = demo["ds"]["summary"]
    assert ds["label_key"] == "Cplus_delta24" and set(ds["strat_counts"]) <= {"C", "M", "X", "NO1", "NO2", "NO3", "NO4"}


def test_failure_is_recorded_and_never_completed(demo):
    store, jobs = demo["store"], demo["jobs"]
    # unknown model: bypass GUI validation on purpose to exercise the error path
    mc = store.create("model_config", "R", {"model": "NOT_A_MODEL", "model_params": {},
                                            "training_config": DEFAULT_TRAINING_CONFIG,
                                            "regularization_params": [1e-4], "seed": 1}, status="completed", mode="demo")
    tr = store.create("training", "R fail", {"windows": ["3h"], "window_map": {"3h": 16}, "n_folds": 3,
                                             "device": "cpu"},
                      parents={"split": demo["sp"]["id"], "model_config": mc["id"]}, mode="demo")
    m = run_and_wait(jobs, store, tr["id"])
    assert m["status"] == "failed" and m["run"]["error"] and m["run"]["traceback"]
    assert "completed" not in [h["status"] for h in m["history"]]


def test_cancel_running_job(demo):
    store, jobs, pl = demo["store"], demo["jobs"], demo["pl"]
    tc = copy.deepcopy(DEFAULT_TRAINING_CONFIG)
    tc.update(num_epochs=500, patience=500)
    tc["early_stopping"]["patience"] = 500
    mc = pl.create_model_config("LSTM long", {"model": "LSTM", "model_params": {}, "training_config": tc, "seed": 1})
    tr = pl.create_training("long", demo["sp"]["id"], mc["id"], {"windows": ["24h"], "window_map": {"24h": 121},
                                                                  "n_folds": 3, "device": "cpu"})
    job = jobs.launch([tr["id"]])
    t0 = time.time()
    while store.get(tr["id"])["status"] != "running" and time.time() - t0 < 120:
        time.sleep(0.5)
    time.sleep(5)
    jobs.cancel(job["id"])
    m = store.get(tr["id"])
    assert m["status"] == "cancelled"
    t0 = time.time()
    while jobs.active() and time.time() - t0 < 20:      # the killed process may take a moment to disappear
        time.sleep(0.5)
    still = [(j["id"], subprocess.run(["ps", "-o", "pid,ppid,stat,wchan:20,cmd", "-g", str(j["pid"])],
                                      capture_output=True, text=True).stdout) for j in jobs.active()]
    assert not still, still
    assert store.get(tr["id"])["status"] == "cancelled"   # never overwritten by the dying runner


def test_reproduce_chain_gives_same_metrics(demo):
    store, jobs, pl = demo["store"], demo["jobs"], demo["pl"]
    new = pl.reproduce_chain(demo["inf"]["id"])
    assert [store.get(a)["type"] for a in new] == ["labels", "dataset", "split", "training", "inference"]
    jobs.launch(new)
    t0 = time.time()
    while store.get(new[-1])["status"] in ("pending", "running") and time.time() - t0 < 1800:
        time.sleep(2)
        jobs.reconcile()
    m = store.get(new[-1])
    assert m["status"] == "completed", dump(m)
    for w in ("3h", "24h"):
        a = demo["inf"]["summary"]["windows"][w]["metrics"]
        b = m["summary"]["windows"][w]["metrics"]
        assert a["TSS"] == pytest.approx(b["TSS"]) and a["ROC_AUC"] == pytest.approx(b["ROC_AUC"])


def test_bundle_moves_the_chain_to_another_installation(demo, tmp_path):
    """The whole chain exported and imported in a new workspace: a new inference runs on the imported
    model and split without re-running anything upstream, with the same metrics."""
    from cs_gui import bundle
    src = demo["settings"]
    out = tmp_path / "chain.zip"
    bundle.export_bundle(demo["store"], [demo["inf"]["id"]], out)
    s2 = Settings(cornerstone_root=CORNERSTONE_ROOT, demo_workspace=str(tmp_path / "other"), mode="demo",
                  min_free_disk_gb=0.1)
    s2_store = ArtifactStore(s2.active_workspace)
    r = bundle.import_bundle(s2_store, out, mode="demo")
    assert demo["tr"]["id"] in r["imported"] and all(s2_store.verify(a) == [] for a in r["imported"])
    jobs2, pl2 = JobManager(s2, s2_store), Pipeline(s2, s2_store)
    # an installation needs a data store to start jobs (inference itself reads only the imported split)
    import shutil
    shutil.copytree(src.demo_data_root, s2.demo_data_root)
    inf = run_and_wait(jobs2, s2_store, pl2.create_inference("I2", demo["tr"]["id"], demo["sp"]["id"],
                                                             demo["inf"]["config"])["id"])
    assert inf["status"] == "completed", dump(inf)
    for w in demo["inf"]["summary"]["windows"]:
        assert inf["summary"]["windows"][w]["metrics"]["TSS"] == \
            pytest.approx(demo["inf"]["summary"]["windows"][w]["metrics"]["TSS"])


def test_all_pages_render(demo, monkeypatch):
    from streamlit.testing.v1 import AppTest
    s = demo["settings"]
    monkeypatch.setenv("CS_GUI_MODE", "demo")
    monkeypatch.setenv("CS_GUI_DEMO_WORKSPACE", s.demo_workspace)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pages = sorted(os.listdir(os.path.join(root, "app", "views")))
    for p in pages + ["../streamlit_app.py"]:
        at = AppTest.from_file(os.path.join(root, "app", "views", p), default_timeout=180).run()
        assert not at.exception, (p, [e.value for e in at.exception])
    # open the detail view of each artifact list
    for page, key in (("labeling.py", "open_labels"), ("dataset.py", "open_dsl"), ("training.py", "open_trl"),
                      ("inference.py", "open_infl")):
        at = AppTest.from_file(os.path.join(root, "app", "views", page), default_timeout=180).run()
        at.selectbox(key=key).select_index(1).run()
        assert not at.exception, (page, [e.value for e in at.exception])
    at = AppTest.from_file(os.path.join(root, "app", "views", "active_regions.py"), default_timeout=180).run()
    at.multiselect[1].select(at.multiselect[1].options[0]).run()
    assert not at.exception
    assert json.dumps(demo["pl"].export(demo["inf"]["id"]))
