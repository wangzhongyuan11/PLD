#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${VIVADO_BIN:-/d/vivado/2026.1/Vivado/bin}"
PY="${ACDC_PYTHON:-$ROOT/acdc/.venv/Scripts/python.exe}"
export ACDC_CLOCK_MHZ="${1:-125}"
case "$ACDC_CLOCK_MHZ" in 100|125) ;; *) echo "clock must be 100 or 125 MHz"; exit 1;; esac
OUT="$ROOT/out/acdc/vivado_fast${ACDC_CLOCK_MHZ}"
mkdir -p "$OUT"
cd "$ROOT"
rm -f "$OUT/build_manifest.json"
"$BIN/vivado.bat" -mode batch -notrace -nolog -nojournal -source "$ROOT/acdc/vivado/package_fast.tcl" > "$OUT/package.log" 2>&1
"$BIN/vivado.bat" -mode batch -notrace -nolog -nojournal -source "$ROOT/acdc/vivado/build_fast.tcl" -tclargs cnn "$OUT" > "$OUT/build.log" 2>&1
"$PY" -B "$ROOT/scripts/extract_xsa.py" "$OUT/dma_top.xsa" "$OUT/dma_top"
"$PY" -B "$ROOT/scripts/check_build.py" "$OUT"
"$PY" -B -m acdc.package "$OUT" --fast
echo ACDC_FAST_BUILD_DONE
