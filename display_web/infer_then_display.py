"""PYNQ-Z2 static-image workflow: B inference exits, then C loads HDMI.

The two full bitstreams run sequentially. No concurrent inference/display.
"""
import argparse
import hashlib
import json
import os
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


class RuntimeVerifier:
    """Hash once; watch stat signatures per request on the local deployment FS.

    This is a performance cache, not a defense against privileged tampering.
    The deployment must remain immutable while a worker is alive.
    """
    def __init__(self, runtime):
        self.runtime = Path(runtime).resolve()
        self.identity = None
        self.snapshot = None
        self.full_checks = 0

    def _snapshot(self, files):
        result = {}
        # Reuse parent resolution only within this snapshot, never across
        # requests: directory symlink changes must still be detected.
        parents = {}
        root = str(self.runtime)
        prefix = os.path.join(root, '')
        for name in ['deployment_manifest.json'] + list(files):
            parent, leaf = os.path.split(os.path.join(root, name))
            if parent not in parents:
                parents[parent] = os.path.realpath(parent)
            target = os.path.join(parents[parent], leaf)
            if os.path.islink(target):
                target = os.path.realpath(target)
            if not target.startswith(prefix):
                raise ValueError('Runtime path escapes deployment: '+name)
            stat = os.stat(target)
            result[name] = (target, stat.st_dev, stat.st_ino, stat.st_size,
                            stat.st_mtime_ns, stat.st_ctime_ns)
        return result

    def check(self):
        manifest_path = self.runtime/'deployment_manifest.json'
        raw = manifest_path.read_bytes()
        identity = hashlib.sha256(raw).hexdigest()
        if self.identity is not None and identity != self.identity:
            raise ValueError('运行包清单已变更，请重启服务后重新校验')
        files = json.loads(raw.decode('utf-8'))['files']
        before = self._snapshot(files)
        if self.snapshot == before and platform.system() == 'Linux':
            return 'metadata_unchanged'
        # Any detected change forces full content validation. Publish no cache
        # until the manifest and all signatures remain stable across that read.
        verify_runtime(self.runtime)
        after = self._snapshot(files)
        if before != after or manifest_path.read_bytes() != raw:
            raise ValueError('校验期间运行包发生变化，请重启服务')
        self.snapshot = after
        self.identity = identity
        self.full_checks += 1
        return 'full_sha256'


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
