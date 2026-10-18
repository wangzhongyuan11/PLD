#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${VIVADO_BIN:-/d/vivado/2026.1/Vivado/bin}"
PY="${ACDC_PYTHON:-$ROOT/acdc/.venv/Scripts/python.exe}"
OUT="$ROOT/out/acdc/vivado"
mkdir -p "$OUT"
cd "$ROOT"
rm -f "$OUT/build_manifest.json"
"$BIN/vivado.bat" -mode batch -notrace -nolog -nojournal -source "$ROOT/acdc/vivado/package.tcl" > "$ROOT/out/acdc/package.log" 2>&1
"$BIN/vivado.bat" -mode batch -notrace -nolog -nojournal -source "$ROOT/acdc/vivado/build.tcl" -tclargs cnn "$OUT" > "$ROOT/out/acdc/build.log" 2>&1
"$PY" -B "$ROOT/scripts/extract_xsa.py" "$OUT/dma_top.xsa" "$OUT/dma_top"
"$PY" -B "$ROOT/scripts/check_build.py" "$OUT"
"$PY" -B -m acdc.package "$OUT"
echo ACDC_BUILD_DONE
