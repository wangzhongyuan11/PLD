"""Archive prepared test tensors only after checking every artifact hash."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reference.datasets import validate_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--allow-subset", action="store_true")
    parser.add_argument("--images", type=Path, action="append", help="include original JPEG files; repeat for multiple source directories")
    args = parser.parse_args()
    manifest_path = args.dataset/"manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not validate_manifest(data) and not args.allow_subset:
        raise ValueError("reference evaluation incomplete")
    paths = [manifest_path]
    for record in data["records"]:
        path = args.dataset/record["file"]
        if path.resolve().parent != args.dataset.resolve() or path.name != record["id"]+".npz":
            raise ValueError("invalid artifact path")
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("artifact hash mismatch")
        paths.append(path)
    if len(set(paths)) != len(paths):
        raise ValueError("duplicate cases")
    images = {}
    if args.images:
        available = [path for root in args.images for path in root.rglob("*.jpg")]
        image_map = {f.stem: f for f in available}
        if len(available) != len(image_map):
            raise ValueError("duplicate source image IDs")
        for record in data["records"]:
            path = image_map[record["id"]]
            if hashlib.sha256(path.read_bytes()).hexdigest() != record["image_sha256"]:
                raise ValueError("source JPEG hash differs from reference")
            images[record["id"]+".jpg"] = path
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output, "w:gz") as archive:
        for path in paths:
            archive.add(path, arcname="dataset/"+path.name, recursive=False)
        for name, path in images.items():
            archive.add(path, arcname="dataset/images/"+name, recursive=False)
    print(f"EVALUATION_PACKAGE_PASS cases={len(paths)-1} bytes={args.output.stat().st_size}")


if __name__ == "__main__":
    main()
