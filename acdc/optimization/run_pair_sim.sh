#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="${VIVADO_BIN:-/d/vivado/2026.1/Vivado/bin}"
OUT="$ROOT/out/acdc/optimization/pair_sim"
mkdir -p "$OUT"
cd "$OUT"
"$BIN/xvlog.bat" -sv -nolog "$ROOT/acdc/optimization/rtl/int8_pair_mul.sv" "$ROOT/acdc/optimization/sim/tb_int8_pair_mul.sv" > compile.log 2>&1
"$BIN/xelab.bat" -nolog tb_int8_pair_mul -s pair -timescale 1ns/1ps >> compile.log 2>&1
"$BIN/xsim.bat" pair -R > simulation.log 2>&1
rg 'TB_PASS int8_pair_mul' simulation.log
