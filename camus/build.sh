#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${VIVADO_BIN:-/d/vivado/2026.1/Vivado/bin}"
PY="${CAMUS_PYTHON:-/d/conda/python.exe}"
OUT="$ROOT/out/camus/vivado"
mkdir -p "$OUT"
cd "$ROOT"
# Only this candidate's certificate is invalidated; old builds stay intact.
rm -f "$OUT/build_manifest.json"
"$BIN/vivado.bat" -mode batch -notrace -nolog -nojournal -source "$ROOT/camus/vivado/package.tcl" > "$ROOT/out/camus/package.log" 2>&1
"$BIN/vivado.bat" -mode batch -notrace -nolog -nojournal -source "$ROOT/camus/vivado/build.tcl" -tclargs cnn "$OUT" > "$ROOT/out/camus/build.log" 2>&1
"$PY" -B "$ROOT/scripts/extract_xsa.py" "$OUT/dma_top.xsa" "$OUT/dma_top"
"$PY" -B "$ROOT/scripts/check_build.py" "$OUT"
"$PY" -B -m camus.package --certify-build
echo CAMUS_BUILD_DONE
