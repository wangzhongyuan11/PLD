#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="${VIVADO_BIN:-/d/vivado/2026.1/Vivado/bin}"
OUT="$ROOT/out/acdc/optimization/control_sim"
mkdir -p "$OUT"
cd "$OUT"
"$BIN/xvlog.bat" -sv -nolog "$ROOT/acdc/optimization/rtl/frame_control.sv" "$ROOT/acdc/optimization/sim/tb_frame_control.sv" > compile.log 2>&1
"$BIN/xelab.bat" -nolog tb_frame_control -s control -timescale 1ns/1ps >> compile.log 2>&1
"$BIN/xsim.bat" control -R > simulation.log 2>&1
rg 'TB_PASS frame_control' simulation.log
