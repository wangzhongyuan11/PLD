"""Build and verify the optional ARM gather kernel on the PYNQ board."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("out/native/tensor_copy.so"))
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.machine() not in ("armv7l", "armv8l"):
        parser.error("build on the 32-bit ARM PYNQ target")
    root = Path(__file__).resolve().parents[1]
    source = root/"board/native/tensor_copy.c"
    wrapper = root/"board/native_copy.py"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    flags = ["-O3", "-std=c11", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
             "-mcpu=cortex-a9", "-mfpu=neon", "-mfloat-abi=hard"]
    manifest = {"source_sha256": digest(source), "wrapper_sha256": digest(wrapper),
                "flags": flags, "machine": platform.machine(),
                "compiler": subprocess.check_output(["gcc", "--version"], text=True).splitlines()[0]}
    subprocess.run(["gcc", *flags, str(source), "-o", str(args.output)], check=True)
    subprocess.run([sys.executable, str(root/"scripts/verify_native_copy.py"), str(args.output)], check=True)
    if digest(source) != manifest["source_sha256"] or digest(wrapper) != manifest["wrapper_sha256"]:
        raise RuntimeError("native source changed during compilation")
    manifest["library_sha256"] = digest(args.output)
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2)+"\n")
    print("NATIVE_BUILD_PASS "+str(args.output), flush=True)


if __name__ == "__main__":
    main()
