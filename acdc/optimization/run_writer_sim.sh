#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="${VIVADO_BIN:-/d/vivado/2026.1/Vivado/bin}"
PY="${ACDC_PYTHON:-$ROOT/acdc/.venv/Scripts/python.exe}"
P="${1:-4}"
case "$P" in 4|6) ;; *) echo "P must be 4 or 6" >&2; exit 2;; esac
OUT="$ROOT/out/acdc/optimization/writer_p${P}_sim"
mkdir -p "$OUT"
cd "$ROOT"
"$PY" -m acdc.optimization.gen_tile_vectors "$OUT" --p "$P"
cd "$OUT"
"$BIN/xvlog.bat" -sv -nolog "$ROOT/acdc/optimization/rtl/conv_array.sv" "$ROOT/rtl/requant_q24.sv" "$ROOT/acdc/optimization/rtl/conv_compute.sv" "$ROOT/acdc/optimization/rtl/tile_banks.sv" "$ROOT/acdc/optimization/rtl/tile_conv.sv" "$ROOT/acdc/optimization/rtl/stream_fork.sv" "$ROOT/acdc/optimization/rtl/pool_stream.sv" "$ROOT/acdc/optimization/rtl/dma_transfer.sv" "$ROOT/acdc/optimization/rtl/tile_writer.sv" "$ROOT/acdc/optimization/sim/datamover_sink.sv" "$ROOT/acdc/optimization/sim/tb_tile_conv.sv" > compile.log 2>&1
"$BIN/xelab.bat" -nolog "tb_tile_writer_p$P" -s writer -timescale 1ns/1ps >> compile.log 2>&1
"$BIN/xsim.bat" writer -R > simulation.log 2>&1
rg 'TB_PASS tile_writer' simulation.log
