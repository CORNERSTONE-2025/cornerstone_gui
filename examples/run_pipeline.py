"""Reproducible example pipeline without the GUI (same backend the GUI uses).

    python examples/run_pipeline.py                       # real mode, configs in examples/configs
    python examples/run_pipeline.py --mode demo           # synthetic demo database (no scientific value)
    python examples/run_pipeline.py --model model_lstm.json --training training.json

Every step creates a new artifact in the workspace (nothing is overwritten); the final lines print
the inference metrics and the id to open in the GUI (page "Experiments and pipeline").
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cs_gui.demo import build_demo_db  # noqa: E402
from cs_gui.jobs import JobManager  # noqa: E402
from cs_gui.pipeline import Pipeline  # noqa: E402
from cs_gui.settings import load_settings  # noqa: E402
from cs_gui.store import ArtifactStore  # noqa: E402

CFG = ROOT / "examples" / "configs"


def load(name):
    return json.loads((CFG / name).read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["real", "demo"], default=None)
    ap.add_argument("--labels", default="labels_existing_Cplus_DEF.json")
    ap.add_argument("--dataset", default="dataset_clean_w24h_subsample.json")
    ap.add_argument("--split", default="split_random.json")
    ap.add_argument("--model", default="model_xgboost.json")
    ap.add_argument("--training", default="training.json")
    ap.add_argument("--inference", default="inference_own_test.json")
    a = ap.parse_args()

    s = load_settings()
    if a.mode:
        s.mode = a.mode
    if s.mode == "demo" and not s.demo_db_path.exists():
        print("Creating the demo DB (synthetic):", build_demo_db(s.demo_db_path))
    if s.problems():
        sys.exit("Incomplete configuration:\n- " + "\n- ".join(s.problems()))
    store = ArtifactStore(s.active_workspace)
    jobs, pl = JobManager(s, store), Pipeline(s, store)

    def run(m):
        jobs.launch([m["id"]])
        while store.get(m["id"])["status"] in ("pending", "running"):
            time.sleep(2)
            jobs.reconcile()
        m = store.get(m["id"])
        print(f"{m['type']:10s} {m['id']}  {m['status']}  ({(m.get('run') or {}).get('duration_s')} s)")
        if m["status"] != "completed":
            sys.exit(f"Step failed: {m['run'].get('error')}\nLog: {store.log_path(m['id'])}")
        return m

    lab_cfg = load(a.labels)
    if s.mode == "demo":
        lab_cfg["source"] = "generate"           # the demo DB has no pre-computed label tables
    lab = run(pl.create_labels("Example: labels", lab_cfg))
    ds_cfg = load(a.dataset)
    if ds_cfg.get("subsample") and ds_cfg["subsample_size"] > lab["summary"]["n_rows"]:
        ds_cfg.update(subsample=False, subsample_size=0)    # fewer labelled events than requested
    ds = run(pl.create_dataset("Example: dataset", lab["id"], ds_cfg))
    sp_cfg = load(a.split)
    sp_cfg["DIRECT_FEATURES"] = [f for f in sp_cfg.get("DIRECT_FEATURES", []) if f in ds["summary"]["feature_names"]] \
        or ds["summary"]["feature_names"]
    if s.mode == "demo":
        sp_cfg.update(TEMPORAL_FILTER_MODE=None, TEMPORAL_WINDOW=None, TEST_CUTOFF_DATE=None)
    sp = run(pl.create_split("Example: split", ds["id"], sp_cfg))
    mc = pl.create_model_config("Example: model", load(a.model))
    tr_cfg = load(a.training)
    tr_cfg["n_folds"] = min(tr_cfg["n_folds"], sp["summary"]["leakage"]["n_folds"])
    tr = run(pl.create_training("Example: training", sp["id"], mc["id"], tr_cfg))
    inf = run(pl.create_inference("Example: inference", tr["id"], sp["id"], load(a.inference)))
    print("\nMetrics on the test set (samples never seen in training):")
    for w, e in inf["summary"]["windows"].items():
        m = e.get("metrics") or {}
        ci = (e.get("ci") or {}).get("TSS") or {}
        print(f"  {w:>4}: TSS={m.get('TSS', float('nan')):.3f} "
              f"[{ci.get('low', float('nan')):.3f}, {ci.get('high', float('nan')):.3f}]  "
              f"threshold={e['threshold']:.3f} ({e['threshold_source']})  n={e['n_evaluated']}")
    print(f"\nOpen the GUI and look for artifact {inf['id']} in the 'Experiments and pipeline' page.")


if __name__ == "__main__":
    main()
