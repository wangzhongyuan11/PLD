#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="${VIVADO_BIN:-/d/vivado/2026.1/Vivado/bin}"
OUT="$ROOT/out/acdc/optimization/queue_sim"
mkdir -p "$OUT"
cd "$OUT"
"$BIN/xvlog.bat" -sv -nolog "$ROOT/acdc/optimization/rtl/tile_job_queue.sv" "$ROOT/acdc/optimization/sim/tb_tile_job_queue.sv" > compile.log 2>&1
"$BIN/xelab.bat" -nolog tb_tile_job_queue -s queue -timescale 1ns/1ps >> compile.log 2>&1
"$BIN/xsim.bat" queue -R > simulation.log 2>&1
rg 'TB_PASS tile_job_queue' simulation.log
