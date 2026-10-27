import pytest

from cs_gui.store import StoreError


def test_create_and_get(store):
    m = store.create("labels", "L1", {"a": 1})
    assert m["id"].startswith("lbl-")
    assert store.get(m["id"])["config"] == {"a": 1}
    assert [x["id"] for x in store.list("labels")] == [m["id"]]


def test_missing_parent_role_and_unknown_parent(store):
    with pytest.raises(StoreError, match="Dipendenze mancanti"):
        store.create("dataset", "D", {})
    with pytest.raises(StoreError, match="non trovato"):
        store.create("dataset", "D", {}, parents={"labels": "lbl-nope"})


def test_config_frozen_after_execution_and_completed_immutable(store):
    m = store.create("labels", "L", {"a": 1})
    store.update(m["id"], config={"a": 2})                 # still pending: editable
    store.set_status(m["id"], "running")
    with pytest.raises(StoreError):
        store.update(m["id"], config={"a": 3})
    store.set_status(m["id"], "completed")
    with pytest.raises(StoreError):
        store.set_status(m["id"], "pending")               # never re-run in place
    assert store.get(m["id"])["config"] == {"a": 2}


def test_outputs_hash_and_integrity(store):
    m = store.create("labels", "L", {})
    f = store.path(m["id"]) / "out.txt"
    f.write_text("hello")
    store.register_outputs(m["id"])
    store.set_status(m["id"], "completed")
    assert store.verify(m["id"]) == []
    f.write_text("hellO")                                  # same size, different content
    assert any("Checksum" in p for p in store.verify(m["id"]))
    f.unlink()
    assert any("mancante" in p for p in store.verify(m["id"]))


def test_lineage_and_protected_delete(store):
    a = store.create("labels", "L", {}, status="completed")
    b = store.create("dataset", "D", {}, parents={"labels": a["id"]})
    c = store.create("split", "S", {}, parents={"dataset": b["id"]})
    assert store.ancestors(c["id"]) == [a["id"], b["id"]]
    assert set(store.descendants(a["id"])) == {b["id"], c["id"]}
    with pytest.raises(StoreError, match="non confermata"):
        store.delete(a["id"])
    with pytest.raises(StoreError, match="dipendono"):
        store.delete(a["id"], confirm=True)
    gone = store.delete(a["id"], confirm=True, cascade=True)
    assert set(gone) == {a["id"], b["id"], c["id"]} and store.list() == []


def test_running_artifact_cannot_be_deleted(store):
    a = store.create("labels", "L", {})
    store.set_status(a["id"], "running")
    with pytest.raises(StoreError, match="esecuzione"):
        store.delete(a["id"], confirm=True)


def test_corrupted_manifest_is_listed(store):
    a = store.create("labels", "L", {})
    store.manifest_path(a["id"]).write_text("{not json")
    assert store.list()[0]["status"] == "corrupted"
    assert store.verify(a["id"])
