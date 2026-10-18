#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="${VIVADO_BIN:-/d/vivado/2026.1/Vivado/bin}"
PY="${ACDC_PYTHON:-$ROOT/acdc/.venv/Scripts/python.exe}"
OUT="$ROOT/out/acdc/optimization/arbiter_sim"
mkdir -p "$OUT"
cd "$ROOT"
"$PY" -m acdc.optimization.gen_tile_vectors "$OUT" --p 4
cd "$OUT"
"$BIN/xvlog.bat" -sv -nolog "$ROOT/acdc/optimization/rtl/dma_read_arbiter.sv" "$ROOT/acdc/optimization/sim/datamover_source.sv" "$ROOT/acdc/optimization/sim/tb_dma_read_arbiter.sv" > compile.log 2>&1
"$BIN/xelab.bat" -nolog tb_dma_read_arbiter -s arbiter -timescale 1ns/1ps >> compile.log 2>&1
"$BIN/xsim.bat" arbiter -R > simulation.log 2>&1
rg 'TB_PASS dma_read_arbiter' simulation.log
