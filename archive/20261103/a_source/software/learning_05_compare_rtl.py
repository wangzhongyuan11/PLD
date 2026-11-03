"""Read a P2 image, simulate the RTL threshold, and compare every output label.

This checks Python/RTL agreement, not agreement with a medical annotation.
Run from any directory: py <project>/software/learning_05_compare_rtl.py
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from baseline import read_pnm, write_pnm


PROJECT = Path(__file__).resolve().parent.parent


def checked_run(command, log_path):
    result = subprocess.run(command, cwd=PROJECT, capture_output=True,
                            text=True, encoding="utf-8", errors="replace", timeout=120)
    log_path.write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"Tool failed; see {log_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path,
                        default=PROJECT / "data/learning/tiny_gray.pgm")
    parser.add_argument("--thresholds", type=int, nargs="+", default=[100, 170])
    parser.add_argument("--iverilog", default="C:/iverilog/bin/iverilog.exe")
    parser.add_argument("--vvp", default="C:/iverilog/bin/vvp.exe")
    args = parser.parse_args()
    width, height, values = read_pnm(args.image, "P2")
    count = width * height
    if not 1 <= count <= 65536 or len(values) != count:
        raise ValueError("Image must contain 1..65536 pixels with matching dimensions")
    if any(not 0 <= t <= 255 for t in args.thresholds):
        raise ValueError("Threshold must be in 0..255")
    output_dir = PROJECT / "runs/learning/file_compare"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    report = {"status": "running", "image": str(args.image.resolve()),
              "sha256": hashlib.sha256(args.image.read_bytes()).hexdigest(),
              "width": width, "height": height, "pixels": count, "cases": []}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        # Each line is one unsigned byte in hexadecimal, in row-major order.
        input_path = output_dir / "pixels.hex"
        input_path.write_text("".join(f"{value:02X}\n" for value in values), encoding="ascii")
        program = output_dir / "image_file.vvp"
        checked_run([args.iverilog, "-g2012", "-s", "tb_learning_image_file",
                     f"-Ptb_learning_image_file.N={count}", "-o", str(program),
                     str(PROJECT / "hardware/rtl/learning_frame_threshold_config.v"),
                     str(PROJECT / "hardware/sim/tb_learning_image_file.sv")],
                    output_dir / "compile.log")
        for threshold in args.thresholds:
            expected = [1 if value < threshold else 0 for value in values]
            actual_path = output_dir / f"rtl_t{threshold}.txt"
            # Truncate before launching, so stale successful output cannot pass.
            actual_path.write_text("", encoding="ascii")
            # Icarus $readmemh can reject non-ASCII absolute path arguments.
            # The subprocess cwd is PROJECT; use its ASCII relative run paths.
            checked_run([args.vvp, str(program),
                         f"+INPUT={input_path.relative_to(PROJECT).as_posix()}",
                         f"+OUTPUT={actual_path.relative_to(PROJECT).as_posix()}",
                         f"+THRESHOLD={threshold}",
                         f"+WAVE={(output_dir / f't{threshold}.vcd').relative_to(PROJECT).as_posix()}"],
                        output_dir / f"sim_t{threshold}.log")
            tokens = actual_path.read_text(encoding="ascii").split()
            if len(tokens) != count or any(value not in ("0", "1") for value in tokens):
                raise ValueError(f"Invalid RTL output length/labels for T={threshold}")
            actual = [int(value) for value in tokens]
            for i, (python_label, rtl_label) in enumerate(zip(expected, actual)):
                if python_label != rtl_label:
                    row, col = divmod(i, width)
                    raise AssertionError(
                        f"T={threshold}: pixel ({row+1},{col+1}), "
                        f"Python={python_label}, RTL={rtl_label}")
            for name, labels in (("python", expected), ("rtl", actual)):
                write_pnm(output_dir / f"{name}_mask_t{threshold}.pgm",
                          width, height, [label * 255 for label in labels])
            report["cases"].append({"threshold": threshold, "status": "PASS",
                                     "python": expected, "rtl": actual,
                                     "target_pixels": sum(actual), "mismatches": 0})
            print(f"PASS T={threshold}: pixels={count}, targets={sum(actual)}, mismatches=0")
        report["status"] = "PASS"
    except Exception as error:
        report["status"] = "FAIL"
        report["error"] = str(error)
        raise
    finally:
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
