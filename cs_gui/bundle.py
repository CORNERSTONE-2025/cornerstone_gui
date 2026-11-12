"""Portable artifact bundles: move completed artifacts from one installation to another.

A bundle is a zip file::

    bundle.json                      format, version, exported artifacts (dependencies first), origin
    artifacts/<id>/manifest.json     the artifact as it is in the source workspace
    artifacts/<id>/...               every file of the artifact (outputs, logs)

``export_bundle`` packs the chosen artifacts **with all their ancestors**, so the bundle is
self-contained: on the other installation the artifacts are usable right away (results, plots,
Active Region history) and can be the input of new steps (e.g. inference with an imported model)
without re-running the chain. Artifact ids are kept, so the dependencies stay wired.

``import_bundle`` checks every file against the sha256 of its manifest before anything enters
the workspace, never overwrites: an artifact already present with the same outputs is skipped,
one with the same id but different outputs is a conflict and the import is refused.
Only the standard library is used (the module is importable from both environments).

    python -m cs_gui.bundle export --workspace W --out FILE.zip ID [ID ...]
    python -m cs_gui.bundle import --workspace W FILE.zip
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from .store import FINAL_OK, ArtifactStore, StoreError, _atomic_write_json, now_iso, sha256_file

FORMAT = "cornerstone-gui-artifacts"
VERSION = 1
# already compressed: storing them again with deflate only costs time
_STORED = (".h5", ".parquet", ".pt", ".pkl", ".joblib", ".png", ".zip", ".gz")


def bundle_ids(store: ArtifactStore, ids: list[str]) -> list[str]:
    """``ids`` with all their ancestors, dependencies first, without duplicates."""
    out: dict[str, None] = {}
    for aid in ids:
        for x in store.ancestors(aid) + [aid]:
            out.setdefault(x, None)
    return list(out)


def export_bundle(store: ArtifactStore, ids: list[str], out: str | Path, note: str = "") -> dict:
    """Write the bundle of ``ids`` (plus ancestors) to ``out``; returns its description."""
    if not ids:
        raise StoreError("Select at least one artifact to export")
    todo = bundle_ids(store, ids)
    for aid in todo:
        m = store.get(aid)
        if m["status"] != FINAL_OK:
            raise StoreError(f"{aid} is '{m['status']}': only completed artifacts (and chains) can be exported")
        problems = store.verify(aid, full_hash=False)
        if problems:
            raise StoreError(f"{aid} is not intact: {problems[:3]}")
    desc = {
        "format": FORMAT, "version": VERSION, "exported_at": now_iso(), "note": note,
        "roots": list(ids), "artifacts": [
            {"id": a, "type": store.get(a)["type"], "name": store.get(a)["name"], "mode": store.get(a)["mode"],
             "bytes": store.size_bytes(a)} for a in todo],
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".partial")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        z.writestr("bundle.json", json.dumps(desc, indent=2))
        for aid in todo:
            base = store.path(aid)
            for p in sorted(base.rglob("*")):
                if p.is_file() and ".tmp" not in p.name and not p.name.endswith(".partial"):
                    arc = f"artifacts/{aid}/{p.relative_to(base).as_posix()}"
                    z.write(p, arc, compress_type=zipfile.ZIP_STORED if p.suffix in _STORED else zipfile.ZIP_DEFLATED)
    tmp.replace(out)
    desc["file"] = str(out)
    desc["size"] = out.stat().st_size
    return desc


def read_bundle(path: str | Path) -> dict:
    """``bundle.json`` of a bundle, after checking its format."""
    with zipfile.ZipFile(path) as z:
        try:
            desc = json.loads(z.read("bundle.json"))
        except KeyError:
            raise StoreError("Not an artifact bundle (bundle.json missing)") from None
    if desc.get("format") != FORMAT:
        raise StoreError(f"Not an artifact bundle (format {desc.get('format')!r})")
    if int(desc.get("version", 0)) > VERSION:
        raise StoreError(f"Bundle version {desc['version']} is newer than this GUI ({VERSION}): update the GUI")
    return desc


def _outputs_signature(m: dict) -> dict:
    return {k: v.get("sha256") or v.get("size") for k, v in (m.get("outputs") or {}).get("files", {}).items()}


def import_bundle(store: ArtifactStore, path: str | Path, mode: str | None = None) -> dict:
    """Import a bundle into ``store``. Returns ``{"imported": [...], "skipped": [...]}``.

    ``mode`` (``real``/``demo``): artifacts of the other mode are refused, the two never mix.
    """
    desc = read_bundle(path)
    ids = [a["id"] for a in desc["artifacts"]]
    if mode and any(a.get("mode") != mode for a in desc["artifacts"]):
        raise StoreError(f"The bundle contains artifacts of a different mode than the current one ({mode})")
    staging = Path(tempfile.mkdtemp(prefix=".import-", dir=store.workspace))
    try:
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():
                name = PurePosixPath(info.filename)
                if info.is_dir() or name.as_posix() == "bundle.json":
                    continue
                # only artifacts/<listed id>/<relative path>: no absolute paths, no "..", no other ids
                if name.is_absolute() or ".." in name.parts or len(name.parts) < 3 \
                        or name.parts[0] != "artifacts" or name.parts[1] not in ids:
                    raise StoreError(f"Unexpected entry in the bundle: {info.filename}")
                dst = staging.joinpath(*name.parts[1:])
                dst.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as src, open(dst, "wb") as fh:
                    shutil.copyfileobj(src, fh, 1 << 20)
        imported, skipped = [], []
        for aid in ids:
            d = staging / aid
            try:
                m = json.loads((d / "manifest.json").read_text())
            except (OSError, json.JSONDecodeError):
                raise StoreError(f"{aid}: manifest missing or unreadable in the bundle") from None
            if m.get("id") != aid or m.get("status") != FINAL_OK:
                raise StoreError(f"{aid}: invalid manifest in the bundle")
            for rel, info in (m.get("outputs") or {}).get("files", {}).items():
                p = d / rel
                if not p.exists() or p.stat().st_size != info.get("size") \
                        or (info.get("sha256") and sha256_file(p) != info["sha256"]):
                    raise StoreError(f"{aid}: file {rel} missing or corrupted in the bundle")
            for role, pid in (m.get("parents") or {}).items():
                if pid not in ids and not store.exists(pid):
                    raise StoreError(f"{aid}: dependency {role} → {pid} is neither in the bundle nor in this workspace")
            if store.exists(aid):
                if _outputs_signature(store.get(aid)) != _outputs_signature(m):
                    raise StoreError(f"{aid} already exists in this workspace with different contents: "
                                     "import refused (nothing was changed)")
                skipped.append(aid)
        for aid in ids:
            if aid in skipped:
                continue
            m = json.loads((staging / aid / "manifest.json").read_text())
            m.setdefault("history", []).append({"at": now_iso(), "event": "imported", "status": m["status"],
                                                "bundle": Path(path).name, "exported_at": desc.get("exported_at")})
            m["imported"] = {"at": now_iso(), "bundle": Path(path).name, "exported_at": desc.get("exported_at")}
            _atomic_write_json(staging / aid / "manifest.json", m)
            store.root.mkdir(parents=True, exist_ok=True)
            (staging / aid).replace(store.path(aid))
            imported.append(aid)
        return {"imported": imported, "skipped": skipped, "roots": desc.get("roots", []), "note": desc.get("note", "")}
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="cs_gui.bundle")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export", help="export artifacts (with their ancestors) to a zip bundle")
    e.add_argument("--workspace", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--note", default="")
    e.add_argument("ids", nargs="+")
    i = sub.add_parser("import", help="import a bundle into a workspace")
    i.add_argument("--workspace", required=True)
    i.add_argument("bundle")
    a = ap.parse_args(argv)
    store = ArtifactStore(a.workspace)
    try:
        res = export_bundle(store, a.ids, a.out, a.note) if a.cmd == "export" else import_bundle(store, a.bundle)
    except StoreError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
