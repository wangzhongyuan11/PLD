#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="${VIVADO_BIN:-/d/vivado/2026.1/Vivado/bin}"
PY="${ACDC_PYTHON:-$ROOT/acdc/.venv/Scripts/python.exe}"
P="${1:-4}"
case "$P" in 4|6) ;; *) echo "P must be 4 or 6" >&2; exit 2;; esac
OUT="${ACDC_SIM_OUTPUT:-$ROOT/out/acdc/optimization/input_p${P}_sim}"
mkdir -p "$OUT"
cd "$ROOT"
"$PY" -m acdc.optimization.gen_input_vectors "$OUT" --p "$P"
cd "$OUT"
"$BIN/xvlog.bat" -sv -nolog "$ROOT/acdc/optimization/rtl/input_transform.sv" "$ROOT/acdc/optimization/rtl/dma_transfer.sv" "$ROOT/acdc/optimization/rtl/input_loader.sv" "$ROOT/acdc/optimization/rtl/dma_read_arbiter.sv" "$ROOT/acdc/optimization/rtl/lut_loader.sv" "$ROOT/acdc/optimization/sim/datamover_source.sv" "$ROOT/acdc/optimization/sim/tb_input_loader.sv" > compile.log 2>&1
"$BIN/xelab.bat" -nolog "tb_input_loader_p$P" -s input_loader -timescale 1ns/1ps >> compile.log 2>&1
"$BIN/xsim.bat" input_loader -R > simulation.log 2>&1
rg 'TB_PASS input_loader' simulation.log
