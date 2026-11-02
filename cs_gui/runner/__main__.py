"""Runner entry point (executed with the CORNERSTONE virtualenv interpreter).

    python -m cs_gui.runner --workspace W --db DB --mode real --job JOB_ID ART_ID [ART_ID ...]
    python -m cs_gui.runner --introspect OUT.json --cornerstone-root ROOT

Artifacts are executed in the given order; completed ones are skipped (resume), the chain stops
at the first failure. stdout/stderr of each step go to ``<artifact>/logs/run.log``.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("CLEARML_OFFLINE_MODE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

from ..store import ArtifactStore, now_iso  # noqa: E402
from . import provenance  # noqa: E402

_current: dict = {}


def _redirect_output(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout.flush()
    sys.stderr.flush()
    fh = open(log_path, "ab", buffering=0)
    os.dup2(fh.fileno(), 1)
    os.dup2(fh.fileno(), 2)
    return fh


def _on_sigterm(signum, frame):
    # multiprocessing workers forked by the package inherit this handler and get SIGTERM when
    # their Pool is closed: only the runner process itself may mark the artifact as cancelled
    if os.getpid() != _current.get("pid"):
        os._exit(0)
    store, aid = _current.get("store"), _current.get("aid")
    if store and aid:
        try:
            store.set_status(aid, "cancelled", finished_at=now_iso(), error="Cancelled by the user (SIGTERM)")
        except Exception:
            pass
    os._exit(143)


def run_artifacts(workspace: str, db: str, mode: str, job_id: str, ids: list[str], cornerstone_root: str) -> int:
    from .steps import STEPS, Ctx
    store = ArtifactStore(workspace)
    _current["store"] = store
    _current["pid"] = os.getpid()
    signal.signal(signal.SIGTERM, _on_sigterm)
    prov = provenance.collect(cornerstone_root)
    for aid in ids:
        m = store.get(aid)
        if m["status"] == "completed":
            print(f"[cs-gui] {aid} already completed: skipped (chain resume)", flush=True)
            continue
        bad = [f"{r}={p} ({store.get(p)['status'] if store.exists(p) else 'missing'})"
               for r, p in m["parents"].items() if not store.exists(p) or store.get(p)["status"] != "completed"]
        if bad:
            store.set_status(aid, "failed", finished_at=now_iso(),
                             error="Dependencies not completed: " + ", ".join(bad))
            return 2
        if m["mode"] != mode:
            store.set_status(aid, "failed", finished_at=now_iso(),
                             error=f"Artifact in '{m['mode']}' mode run in '{mode}' mode")
            return 2
        _current["aid"] = aid
        t0 = time.time()
        store.set_status(aid, "running", pid=os.getpid(), job_id=job_id, started_at=now_iso(),
                         error=None, traceback=None)
        fh = _redirect_output(store.log_path(aid))
        print(f"\n===== [cs-gui] {now_iso()} step '{m['type']}' artifact {aid} (job {job_id}) =====", flush=True)
        os.chdir(store.path(aid))
        try:
            summary = STEPS[m["type"]](Ctx(store=store, m=m, db=db, mode=mode, cornerstone_root=cornerstone_root))
            store.register_outputs(aid)
            store.update(aid, summary=summary, provenance=prov)
            store.set_status(aid, "completed", finished_at=now_iso(), duration_s=round(time.time() - t0, 1),
                             exit_code=0)
            print(f"===== [cs-gui] completed in {time.time() - t0:.1f}s =====", flush=True)
        except BaseException as e:  # noqa: BLE001 - every failure must be recorded
            tb = traceback.format_exc()
            print(tb, flush=True)
            store.update(aid, provenance=prov)
            store.set_status(aid, "failed", finished_at=now_iso(), duration_s=round(time.time() - t0, 1),
                             error=f"{type(e).__name__}: {e}", traceback=tb, exit_code=1)
            return 1
        finally:
            sys.stdout.flush()
            fh.close()
    return 0


def introspect(out: str, cornerstone_root: str) -> None:
    """Describe what the installed package really offers (used to build the GUI forms)."""
    import inspect
    sys.path.insert(0, cornerstone_root)
    from cornerstone import Cornerstone
    from cornerstone.ml.model_configs import MODEL_CONFIGS
    from cornerstone.ml.regularization_configs import REGULARIZATION_PARAMS
    from cornerstone.ml.utils.model_manager import ModelManager
    from cornerstone.ml.utils.training import PT_MODELS, SK_MODELS
    from cornerstone.sharp_data import FeatureSelector
    import torch

    sigs = {}
    for meth in ("splitting_labeling", "create_hdf5_from_db", "create_dataset_splits", "train_models", "test_model"):
        sig = inspect.signature(getattr(Cornerstone, meth))
        sigs[meth] = {k: (None if v.default is inspect.Parameter.empty else repr(v.default))
                      for k, v in sig.parameters.items() if k != "self"}
    defaults = {}
    tc = Path(cornerstone_root) / "train_config.json"
    if tc.exists():
        try:
            defaults = json.loads(tc.read_text())
        except json.JSONDecodeError:
            pass
    trainable = [m for m in PT_MODELS + SK_MODELS if m in ModelManager.MODEL_CLASSES]
    info = {
        "generated_at": now_iso(),
        "cornerstone": provenance.git_state(cornerstone_root),
        "packages": provenance.collect(cornerstone_root)["packages"],
        "cuda_available": bool(torch.cuda.is_available()),
        "pt_models": PT_MODELS, "sk_models": SK_MODELS, "trainable_models": trainable,
        "model_configs": json.loads(json.dumps(MODEL_CONFIGS, default=str)),
        "regularization_params": REGULARIZATION_PARAMS,
        "feature_groups": FeatureSelector.FEATURE_GROUPS,
        "signatures": sigs,
        "default_training_config": defaults.get("TRAINING_CONFIG", {}),
        "default_window_map": (defaults.get("DATA_CONFIG") or {}).get("window_map",
                                                                    {"3h": 16, "6h": 31, "12h": 61, "24h": 121}),
    }
    Path(out).write_text(json.dumps(info, indent=2, default=str))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="cs_gui.runner")
    ap.add_argument("--workspace")
    ap.add_argument("--db")
    ap.add_argument("--mode", default="real")
    ap.add_argument("--job", default="manual")
    ap.add_argument("--introspect")
    ap.add_argument("--cornerstone-root", default=None)
    ap.add_argument("ids", nargs="*")
    a = ap.parse_args(argv)
    root = a.cornerstone_root or next((p for p in sys.path if (Path(p) / "cornerstone" / "__init__.py").exists()), "")
    if a.introspect:
        introspect(a.introspect, root)
        return 0
    if not (a.workspace and a.db and a.ids):
        ap.error("--workspace, --db and at least one artifact are required")
    return run_artifacts(a.workspace, a.db, a.mode, a.job, a.ids, root)


if __name__ == "__main__":
    sys.exit(main())
