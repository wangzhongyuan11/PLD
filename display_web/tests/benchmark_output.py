"""Repeatable local output-pipeline benchmark; no FPGA/HDMI access.

The fixture is deterministic: the real 1022x767 sample mask is nearest-resized
for the large case, and a fixed noisy RGB image is encoded once to JPEG. Each
run goes through Console.submit(mode="mask"), so measurements include upload
decoding but no board inference. Results are host-only and are not board FPS.
"""
import argparse
import base64
import gc
import hashlib
import io
import json
import os
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import PIL
from PIL import Image


def parse_size(value):
    width, height = value.lower().split("x")
    return int(width), int(height)


def fixture(source, size):
    from pld_result import load_result

    sample = load_result(source / "samples/frame_000001.json")
    mask1022 = sample.original_mask()
    width, height = size
    if mask1022.shape == (height, width):
        mask = mask1022
    else:
        mask = np.asarray(Image.fromarray(mask1022.astype(np.uint8) * 255).resize(
            (width, height), Image.Resampling.NEAREST)) != 0

    with Image.open(source / "samples/synthetic_original.png") as original:
        rgb = np.asarray(original.convert("RGB").resize((width, height), Image.Resampling.BICUBIC)).copy()
    y, x = np.mgrid[:height, :width]
    rng = np.random.default_rng(20261012)
    noise = rng.integers(-10, 11, (height, width, 1), dtype=np.int16)
    pattern = ((x // 29 + y // 17) % 7 - 3)[:, :, None]
    rgb = np.clip(rgb.astype(np.int16) + noise + pattern, 0, 255).astype(np.uint8)
    rgb[mask] = np.clip(rgb[mask].astype(np.int16) - 18, 0, 255).astype(np.uint8)

    image = io.BytesIO()
    Image.fromarray(rgb).save(image, format="JPEG", quality=95, subsampling=0)
    mask_png = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8) * 255).save(mask_png, format="PNG", compress_level=6)
    return image.getvalue(), mask_png.getvalue()


def decoded_sha256(path):
    with Image.open(path) as image:
        return hashlib.sha256(np.asarray(image).tobytes()).hexdigest()


def wait(app, ident, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = app.job(ident)
        if job["state"] != "running":
            return job
        time.sleep(0.002)
    raise TimeoutError("benchmark job did not finish")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--sizes", nargs="+", default=["1022x767", "2125x1411"])
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.runs < 3:
        parser.error("use at least 3 runs for a stable median")
    source = args.source.resolve()
    sys.path.insert(0, str(source))
    import server

    all_results = []
    with tempfile.TemporaryDirectory() as data_name:
        for size_text in args.sizes:
            size = parse_size(size_text)
            image_raw, mask_raw = fixture(source, size)
            body = {"mode": "mask", "name": "benchmark-{}".format(size_text),
                    "image": base64.b64encode(image_raw).decode(),
                    "mask": base64.b64encode(mask_raw).decode(), "thickness": 2}
            runs = []
            for index in range(args.runs):
                app = server.Console(data_name, token="benchmark")
                app.board = False
                gc.collect()
                started = time.perf_counter()
                submitted = app.submit(dict(body))
                job = wait(app, submitted["id"])
                elapsed = time.perf_counter() - started
                if job["state"] != "done":
                    raise RuntimeError(job.get("error", "benchmark job failed"))
                folder = Path(data_name) / job["id"]
                files = {path.name: path.stat().st_size for path in folder.iterdir() if path.is_file()}
                pngs = {path.name: decoded_sha256(path) for path in folder.glob("*.png")}
                runs.append({"external_wall_seconds": elapsed,
                             "processing_seconds": job["statistics"].get("processing_seconds"),
                             "stage_seconds": job["statistics"].get("final_stage_seconds",
                                                                    job["statistics"].get("stage_seconds")),
                             "files": files, "png_decoded_sha256": pngs,
                             "statistics": job["statistics"]})
                print("{:9} run {:2d}: external {:.6f}s processing {:.6f}s zip {:.0f} KiB".format(
                    size_text, index + 1, elapsed, job["statistics"]["processing_seconds"],
                    files.get("results.zip", 0) / 1024), flush=True)
            external = [run["external_wall_seconds"] for run in runs]
            png_stage = [sum(run["stage_seconds"]["png_save"].values()) for run in runs]
            png_bytes = [sum(size for name, size in run["files"].items()
                             if name.endswith(".png")) for run in runs]
            zip_bytes = [run["files"].get("results.zip", 0) for run in runs]
            result = {"size": list(size), "runs": runs,
                      "median_external_wall_seconds": statistics.median(external),
                      "min_external_wall_seconds": min(external),
                      "max_external_wall_seconds": max(external),
                      "median_png_save_seconds": statistics.median(png_stage),
                      "median_png_bytes": statistics.median(png_bytes),
                      "median_zip_bytes": statistics.median(zip_bytes)}
            all_results.append(result)

    payload = {"source": str(source), "python": sys.version, "platform": platform.platform(),
               "cpu_count": os.cpu_count(), "pillow": PIL.__version__,
               "png_compress_level": server.PNG_COMPRESS_LEVEL,
               "png_save_workers": server.PNG_SAVE_WORKERS,
               "numpy": np.__version__, "runs_per_size": args.runs,
               "results": all_results}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("WROTE {}".format(args.output_json))


if __name__ == "__main__":
    main()
