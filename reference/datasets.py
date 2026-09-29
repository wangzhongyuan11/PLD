"""Frozen named image collections; completion is relative to an exact ID set."""
import hashlib
import json
from pathlib import Path

from reference.tinyunet import BASELINE_SPEC_SHA256

CONTRACT_PATH = Path(__file__).with_name("dataset_contracts.json")


def collection(name):
    contracts = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if name not in contracts["collections"]:
        raise ValueError("unknown dataset collection: " + name)
    result = dict(contracts["collections"][name])
    result["name"] = name
    if len({r["id"] for r in result["cases"]}) != result["count"]:
        raise ValueError("dataset collection contains duplicate IDs")
    if ids_digest(result["cases"]) != result["ids_sha256"]:
        raise ValueError("dataset collection ID digest mismatch")
    return result


def ids_digest(records):
    data = "\n".join(sorted(r["source"]+":"+r["id"] for r in records))+"\n"
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def validate_manifest(manifest, model=None):
    """Return collection completeness, rejecting fabricated complete flags."""
    records = manifest["records"]
    if not manifest.get("passed") or not records or len({r["id"] for r in records}) != len(records):
        raise ValueError("failed, empty or duplicate reference dataset")
    if model is not None and manifest["model_spec_sha256"] != model.spec_hash:
        raise ValueError("dataset belongs to another model")
    if manifest.get("schema_version", 1) == 1:
        if manifest["model_spec_sha256"] != BASELINE_SPEC_SHA256:
            raise ValueError("legacy dataset schema cannot describe the final model")
        expected = collection("isic2016_test379")
        actual = {r["id"] for r in records}
        complete = actual == {r["id"] for r in expected["cases"]}
    elif manifest["schema_version"] == 2:
        expected = collection(manifest["collection"]["name"])
        identity = {k: expected[k] for k in ("name", "role", "primary", "count", "ids_sha256")}
        if manifest["collection"] != identity:
            raise ValueError("dataset collection identity changed")
        wanted = {(r["source"], r["id"]) for r in expected["cases"]}
        actual = {(r["source"], r["id"]) for r in records}
        if not actual <= wanted:
            raise ValueError("unexpected dataset IDs or sources")
        complete = actual == wanted
        if model is not None and manifest["threshold"] != model.threshold:
            raise ValueError("dataset threshold differs from selected model")
    else:
        raise ValueError("unsupported reference dataset schema")
    if bool(manifest.get("complete_test_set")) != complete:
        raise ValueError("dataset completion flag differs from frozen ID coverage")
    return complete
