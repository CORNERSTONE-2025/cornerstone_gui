"""Portable artifact bundles (cs_gui.bundle): export with ancestors, verified import, no overwrite."""
import json
import zipfile

import pytest

from cs_gui import bundle
from cs_gui.store import ArtifactStore, StoreError


def _done(store, kind, name, parents=None, files=None, mode="real"):
    m = store.create(kind, name, {"k": name}, parents=parents or {}, mode=mode)
    for rel, data in (files or {}).items():
        p = store.path(m["id"]) / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    store.register_outputs(m["id"])
    store.update(m["id"], summary={"n": len(files or {})})
    store.set_status(m["id"], "completed")
    return m["id"]


@pytest.fixture
def chain(tmp_path):
    s = ArtifactStore(tmp_path / "src")
    lab = _done(s, "labels", "L", files={"labels.parquet": b"P" * 1000, "labels_meta.json": b"{}"})
    ds = _done(s, "dataset", "D", {"labels": lab}, {"dataset.h5": b"H" * 5000})
    sp = _done(s, "split", "S", {"dataset": ds}, {"split/splits/split_test.h5": b"T" * 300})
    mc = _done(s, "model_config", "M")
    tr = _done(s, "training", "T", {"split": sp, "model_config": mc}, {"results/model.pkl": b"W" * 700})
    return s, {"lab": lab, "ds": ds, "sp": sp, "mc": mc, "tr": tr}


def test_round_trip_with_ancestors(chain, tmp_path):
    s, ids = chain
    out = tmp_path / "b.zip"
    desc = bundle.export_bundle(s, [ids["tr"]], out, note="for a colleague")
    assert [a["id"] for a in desc["artifacts"]] == [ids["lab"], ids["ds"], ids["sp"], ids["mc"], ids["tr"]]
    dst = ArtifactStore(tmp_path / "dst")
    r = bundle.import_bundle(dst, out, mode="real")
    assert sorted(r["imported"]) == sorted(ids.values()) and r["skipped"] == [] and r["note"] == "for a colleague"
    for aid in ids.values():
        assert dst.verify(aid) == []
        assert dst.get(aid)["parents"] == s.get(aid)["parents"]
        assert dst.get(aid)["history"][-1]["event"] == "imported"
    assert (dst.path(ids["tr"]) / "results" / "model.pkl").read_bytes() == b"W" * 700
    # importing again: everything already present, nothing changes
    assert bundle.import_bundle(dst, out)["skipped"] == list(desc and [a["id"] for a in desc["artifacts"]])


def test_conflict_corruption_and_unsafe_entries(chain, tmp_path):
    s, ids = chain
    out = tmp_path / "b.zip"
    bundle.export_bundle(s, [ids["ds"]], out)
    # same id, different contents in the destination: refused, nothing written
    dst = ArtifactStore(tmp_path / "dst")
    bundle.import_bundle(dst, out)
    (dst.path(ids["ds"]) / "dataset.h5").write_bytes(b"X" * 5000)
    dst.register_outputs(ids["ds"])
    with pytest.raises(StoreError, match="different contents"):
        bundle.import_bundle(dst, out)
    # corrupted file inside the bundle
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(out) as zi, zipfile.ZipFile(bad, "w") as zo:
        for i in zi.infolist():
            data = zi.read(i)
            zo.writestr(i, b"Z" * len(data) if i.filename.endswith("dataset.h5") else data)
    fresh = ArtifactStore(tmp_path / "fresh")
    with pytest.raises(StoreError, match="corrupted"):
        bundle.import_bundle(fresh, bad)
    assert not fresh.root.exists() or not any(fresh.root.iterdir())
    # path traversal
    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(out) as zi, zipfile.ZipFile(evil, "w") as zo:
        for i in zi.infolist():
            zo.writestr(i, zi.read(i))
        zo.writestr(f"artifacts/{ids['ds']}/../../escape.txt", b"x")
    with pytest.raises(StoreError, match="Unexpected entry"):
        bundle.import_bundle(ArtifactStore(tmp_path / "e"), evil)
    assert not (tmp_path / "escape.txt").exists()


def test_refusals(chain, tmp_path):
    s, ids = chain
    running = s.create("inference", "I", {}, parents={"training": ids["tr"], "source": ids["sp"]})
    with pytest.raises(StoreError, match="only completed"):
        bundle.export_bundle(s, [running["id"]], tmp_path / "x.zip")
    out = tmp_path / "b.zip"
    bundle.export_bundle(s, [ids["lab"]], out)
    with pytest.raises(StoreError, match="different mode"):
        bundle.import_bundle(ArtifactStore(tmp_path / "d"), out, mode="demo")
    notb = tmp_path / "n.zip"
    with zipfile.ZipFile(notb, "w") as z:
        z.writestr("x.txt", "x")
    with pytest.raises(StoreError, match="not an artifact bundle|Not an artifact bundle"):
        bundle.read_bundle(notb)
    future = tmp_path / "f.zip"
    with zipfile.ZipFile(future, "w") as z:
        z.writestr("bundle.json", json.dumps({"format": bundle.FORMAT, "version": 99, "artifacts": []}))
    with pytest.raises(StoreError, match="newer"):
        bundle.read_bundle(future)


def test_command_line(chain, tmp_path):
    s, ids = chain
    out = tmp_path / "c.zip"
    assert bundle.main(["export", "--workspace", str(s.workspace), "--out", str(out), ids["sp"]]) == 0
    assert bundle.main(["import", "--workspace", str(tmp_path / "w2"), str(out)]) == 0
    assert ArtifactStore(tmp_path / "w2").exists(ids["sp"])
