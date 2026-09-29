#!/usr/bin/env bash
# One-shot Vivado build (stage 1/2): BD -> synth -> impl -> bitstream -> XSA
# -> extracted dma_top.bit / dma_top.hwh under out/vivado_<mode>/.
#
# Usage: scripts/build.sh fifo|incr|cnn
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${VIVADO_BIN:-/d/vivado/2026.1/Vivado/bin}"
MODE="${1:?usage: build.sh fifo|incr|cnn}"
case "$MODE" in fifo|incr|cnn) ;; *) echo "unsupported mode: $MODE" >&2; exit 2 ;; esac
OUT="$ROOT/out/vivado_$MODE"
LOG="$ROOT/out/vivado_$MODE.log"
mkdir -p "$ROOT/out"

if [ "$MODE" = "incr" ]; then
  "$BIN/vivado.bat" -mode batch -notrace -nolog -nojournal \
    -source "$ROOT/vivado/package_axis_incr.tcl" \
    -log "$ROOT/out/package_axis_incr.log"
fi
if [ "$MODE" = "cnn" ]; then
  mkdir -p "$OUT"
  rm -f "$OUT/build_manifest.json"
  "$BIN/vivado.bat" -mode batch -notrace -nolog -nojournal \
    -source "$ROOT/vivado/package_axis_tensor_accel.tcl" \
    -log "$ROOT/out/package_axis_tensor_accel.log"
fi

"$BIN/vivado.bat" -mode batch -notrace -nolog -nojournal \
  -source "$ROOT/vivado/build_dma_bd.tcl" -log "$LOG" -tclargs "$MODE" "$OUT"

python "$ROOT/scripts/extract_xsa.py" "$OUT/dma_top.xsa" "$OUT/dma_top"
python "$ROOT/scripts/check_build.py" "$OUT"
if [ "$MODE" = "cnn" ]; then python "$ROOT/scripts/write_build_manifest.py" "$OUT"; fi
echo "BUILD_DONE mode=$MODE"
ls -l "$OUT"/dma_top.bit "$OUT"/dma_top.hwh "$OUT"/dma_top.xsa
