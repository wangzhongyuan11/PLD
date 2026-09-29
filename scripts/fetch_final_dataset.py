"""Fetch only the final dev638's missing ISIC2017 images/GT from A's sources.

No training or ISIC2018 data is requested. Downloads retain per-file SHA256
and URLs; final numerical equivalence is checked by prepare_evaluation.py.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reference.datasets import collection

BASE = "https://isic-archive.s3.amazonaws.com/"
ARCHIVES = {
    "isic2017_validation": "ISIC-2017_Validation_Part1_GroundTruth.zip",
    "isic2017_exposed_validation": "ISIC-2017_Test_v2_Part1_GroundTruth.zip",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(url, path):
    if not path.exists():
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=40) as response:
                    data = response.read()
                    length = response.headers.get("Content-Length")
                    if length and len(data) != int(length):
                        raise ValueError("incomplete HTTP response")
                path.write_bytes(data)
                break
            except Exception:
                if attempt == 2: raise
                time.sleep(attempt+1)
    return {"url": url, "sha256": digest(path), "bytes": path.stat().st_size}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    for name in ("downloads", "images", "masks"):
        (args.output/name).mkdir(parents=True, exist_ok=True)
    provenance_path = args.output/"provenance.json"
    report = json.loads(provenance_path.read_text()) if provenance_path.exists() else {"files": {}}
    report.update(passed=False, collection="final_dev638", scope="262 ISIC2017 development images and GT; no 2018 data")
    # Existing files are reusable only with their recorded identity.
    for name, record in report["files"].items():
        if not (args.output/name).is_file() or digest(args.output/name) != record["sha256"]:
            raise ValueError("download identity changed: "+name)
    cases = [r for r in collection("final_dev638")["cases"] if r["source"] in ARCHIVES]
    try:
        for source, filename in ARCHIVES.items():
            path = args.output/"downloads"/filename
            record = download(BASE+"challenges/2017/"+filename, path)
            report["files"][path.relative_to(args.output).as_posix()] = record
            with zipfile.ZipFile(path) as archive:
                names = {Path(n).name: n for n in archive.namelist() if n.lower().endswith(".png")}
                for case in (r for r in cases if r["source"]==source):
                    filename = case["id"]+"_segmentation.png"
                    data = archive.read(names[filename])
                    target = args.output/"masks"/filename
                    target.write_bytes(data)
                    report["files"][target.relative_to(args.output).as_posix()] = {
                        "archive_sha256": record["sha256"], "member": names[filename],
                        "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "source": source}
            print("MASKS_PASS", source, flush=True)
        with ThreadPoolExecutor(max_workers=4) as pool:
            jobs = {pool.submit(download, BASE+"images/"+r["id"]+".jpg", args.output/"images"/(r["id"]+".jpg")): r for r in cases}
            for count, future in enumerate(as_completed(jobs), 1):
                case = jobs[future]
                record = future.result()
                report["files"]["images/"+case["id"]+".jpg"] = {**record, "source": case["source"]}
                provenance_path.write_text(json.dumps(report, indent=2)+"\n")
                if count % 20 == 0: print(f"DOWNLOAD_PROGRESS {count}/{len(cases)}", flush=True)
        report["passed"] = True
    finally:
        provenance_path.write_text(json.dumps(report, indent=2)+"\n")
    print("FINAL_DATA_FETCH_PASS images=262 masks=262", flush=True)


if __name__ == "__main__": main()
