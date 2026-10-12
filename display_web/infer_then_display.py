"""PYNQ-Z2 static-image workflow: B inference exits, then C loads HDMI.

The two full bitstreams run sequentially. No concurrent inference/display.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import signal
import subprocess
import sys

from pld_result import load_result, save_result_images
from show_result import show_on_board


def verify_runtime(runtime):
    manifest = json.loads((runtime / "deployment_manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        target = (runtime / name).resolve()
        if runtime not in target.parents:
            raise ValueError("Runtime manifest path escapes deployment")
        if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise ValueError("Runtime checksum mismatch: " + name)


def inference_command(runtime, image, destination, portable=False):
    command = [sys.executable, str(runtime / "board/infer_images.py"), str(image),
               "--bit", str(runtime / "overlay/dma_top.bit"), "--output", str(destination),
               "--model", str(runtime / "role_a_final/int8"),
               "--fuse-maps", "--fuse-decoder", "--fast-quantize"]
    if not portable:
        for option, name in (("--native-copy", "tensor_copy.so"), ("--native-control", "tensor_control.so"),
                             ("--native-resize", "rgb_resize.so")):
            path = runtime / "out/native" / name
            if not path.is_file():
                raise FileNotFoundError("Build B's native libraries on the board first; missing " + str(path))
            command += [option, str(path)]
        command.append("--tx-range-sync")
    return command


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime", type=Path, required=True, help="Extracted B runtime directory")
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="New B result directory")
    p.add_argument("--buttons", action="store_true")
    p.add_argument("--preview-only", action="store_true", help="Run PL inference then write PNGs without loading HDMI")
    p.add_argument("--portable", action="store_true", help="Explicit slower B Python/Pillow path; no native .so required")
    p.add_argument("--view", choices=("cutout", "crop", "mask", "original", "dashboard"), default="cutout")
    args = p.parse_args(argv)
    if platform.system() != "Linux" or not platform.machine().lower().startswith("arm"):
        p.error("Run with the board's PYNQ Python environment on PYNQ-Z2")
    runtime, source, output = args.runtime.resolve(), args.image.resolve(), args.output.resolve()
    if output.exists():
        p.error("--output must not already exist")
    if not source.is_file():
        p.error("Input image does not exist")
    verify_runtime(runtime)
    command = inference_command(runtime, source, output, args.portable)
    print("Stop all other HDMI/DMA programs first. Running B inference...", flush=True)
    # check=True prevents display after a failed run, including a cleanup failure.
    # B's ExitStack closes/reset DMA before a successful process exit.
    subprocess.run(command, cwd=str(runtime), check=True)
    result = load_result(output / "frame_000001.json", source)
    save_result_images(result, output / "display", args.view)
    print("INFERENCE_AND_POSTPROCESS_PASS:", output / "display", flush=True)
    if not args.preview_only:
        show_on_board(result, args.view, args.buttons)


if __name__ == "__main__":
    def terminate(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped. Do not resume hardware work until the inference process has stopped.")
