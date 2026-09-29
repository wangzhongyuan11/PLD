"""Verify the frozen team handoff; optionally restore model fixtures for a clone.

Uses only the Python standard library. Never overwrites a different local model.
Source comparison normalizes CRLF/LF; frozen archive hashes remain byte-exact.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def archive_files(path):
    result = {}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            name = member.name
            parts = PurePosixPath(name).parts
            if not parts or name.startswith("/") or ".." in parts or "\\" in name or ":" in name:
                raise ValueError(f"unsafe archive path: {name}")
            if member.isdir():
                continue
            if not member.isfile() or name in result:
                raise ValueError(f"nonregular or duplicate archive entry: {name}")
            result[name] = archive.extractfile(member).read()
    return result


def require_hash(data, expected, name):
    if digest(data) != expected:
        raise ValueError(f"SHA256 mismatch: {name}")


def source_equal(name, data):
    current = (ROOT / name).read_bytes()
    if current.replace(b"\r\n", b"\n") != data.replace(b"\r\n", b"\n"):
        raise ValueError(f"checkout differs from frozen source: {name}")


def verify():
    base = ROOT / "delivery"
    manifest = json.loads((base / "MANIFEST.json").read_text(encoding="utf-8"))
    for name, record in manifest["files"].items():
        path = (base / name).resolve()
        if not path.is_relative_to(base.resolve()):
            raise ValueError("delivery manifest escapes its directory")
        data = path.read_bytes()
        require_hash(data, record["sha256"], name)
        if len(data) != record["bytes"]:
            raise ValueError(f"size mismatch: {name}")

    runtime = archive_files(base / manifest["runtime"])
    deployment = json.loads(runtime["deployment_manifest.json"])
    if set(runtime) != set(deployment["files"]) | {"deployment_manifest.json"}:
        raise ValueError("unexpected runtime archive contents")
    for name, sha in deployment["files"].items():
        require_hash(runtime[name], sha, name)
        if name.startswith(("board/", "reference/", "scripts/")):
            source_equal(name, runtime[name])

    hardware = base / "c_v3/hardware"
    build = json.loads((hardware / "build_manifest.json").read_text(encoding="utf-8"))
    sources = archive_files(hardware / "build_sources.tar.gz")
    if set(sources) != set(build["sources"]):
        raise ValueError("unexpected build source archive contents")
    for name, sha in build["sources"].items():
        require_hash(sources[name], sha, name)
        source_equal(name, sources[name])
    for name, sha in build["artifacts"].items():
        data = runtime["overlay/" + name] if name.endswith((".bit", ".hwh")) else (hardware / name).read_bytes()
        require_hash(data, sha, name)

    models = archive_files(base / manifest["model_fixtures"])
    final = json.loads(models["role_a_final/MANIFEST.json"])
    expected = {"role_a_final/MANIFEST.json"}
    for record in final["files"]:
        name = "role_a_final/" + record["path"]
        expected.add(name)
        require_hash(models[name], record["sha256"], name)
        if len(models[name]) != record["bytes"]:
            raise ValueError(f"model size mismatch: {name}")
    for filename, prefix in (("final_manifest.json", "role_a_final/"),
                             ("baseline_manifest.json", "role_a_handoff/role_a/")):
        frozen = json.loads((ROOT / "reference" / filename).read_text(encoding="utf-8"))
        for name, sha in frozen["files"].items():
            expected.add(prefix + name)
            require_hash(models[prefix + name], sha, prefix + name)
    if set(models) != expected:
        raise ValueError("unexpected model fixture archive contents")
    for name in runtime:
        if name.startswith("role_a_final/") and runtime[name] != models[name]:
            raise ValueError(f"runtime and model fixtures differ: {name}")

    reports = archive_files(base / "c_v3/validation/reports.tar.gz")
    summary = json.loads((base / "c_v3/validation/summary.json").read_text(encoding="utf-8"))
    require_hash(reports["fastq_stability1000.json"], summary["report_sha256"], "board report")
    require_hash((base / manifest["runtime"]).read_bytes(), summary["deployment_sha256"], "measured runtime")
    print(f"DELIVERY_PASS artifacts={len(manifest['files'])} runtime_files={len(deployment['files'])} models={len(models)}")
    return models


def prepare_models(models):
    pending = []
    for name, data in models.items():
        target = (ROOT / name).resolve()
        if not target.is_relative_to(ROOT.resolve()):
            raise ValueError("model target escapes checkout")
        if target.exists():
            if target.read_bytes() != data:
                raise ValueError(f"local model differs; preserve it and use a fresh checkout: {name}")
        else:
            pending.append((target, data))
    # All destinations have been checked before writing the first file.
    for target, data in pending:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(data)
    print(f"MODEL_PREPARE_PASS restored={len(pending)} existing={len(models)-len(pending)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-models", action="store_true")
    args = parser.parse_args()
    models = verify()
    if args.prepare_models:
        prepare_models(models)
