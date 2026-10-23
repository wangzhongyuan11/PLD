#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="${VIVADO_BIN:-/d/vivado/2026.1/Vivado/bin}"
OUT="$ROOT/out/acdc/optimization/dma_sim"
mkdir -p "$OUT"
cd "$OUT"
"$BIN/xvlog.bat" -sv -nolog "$ROOT/acdc/optimization/rtl/dma_transfer.sv" "$ROOT/acdc/optimization/sim/tb_dma_transfer.sv" > compile.log 2>&1
"$BIN/xelab.bat" -nolog tb_dma_transfer -s dma -timescale 1ns/1ps >> compile.log 2>&1
"$BIN/xsim.bat" dma -R > simulation.log 2>&1
rg 'TB_PASS dma_transfer' simulation.log
