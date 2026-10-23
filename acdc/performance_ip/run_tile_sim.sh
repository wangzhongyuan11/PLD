#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="${VIVADO_BIN:-/d/vivado/2026.1/Vivado/bin}"
PY="${ACDC_PYTHON:-$ROOT/acdc/.venv/Scripts/python.exe}"
cd "$ROOT"
"$PY" -B -c 'from acdc.gen_vectors import main; main("out/acdc/fast/vectors",0x41434434)'
OUT="$ROOT/out/acdc/fast/sim_accel"
mkdir -p "$OUT"
(
 cd "$OUT"
 "$BIN/xvlog.bat" -sv -nolog "$ROOT/rtl/requant_q24.sv" \
  "$ROOT/rtl/axi_lite_endpoint.sv" "$ROOT/rtl/axis_byte_pack.sv" "$ROOT/rtl/axis_byte_unpack.sv" \
  "$ROOT/acdc/rtl/tensor_tile_engine_acdc.sv" "$ROOT/acdc/performance_ip/conv_mac_128.sv" \
  "$ROOT/acdc/performance_ip/conv_tile_engine_acdc_128.sv" \
  "$ROOT/acdc/performance_ip/axis_tensor_accel_acdc_fast.sv" \
  "$ROOT/acdc/performance_ip/tb_axis_tensor_accel_acdc_fast.sv"
 "$BIN/xelab.bat" -nolog -debug typical tb_axis_tensor_accel_acdc_fast -s acdc_fast -timescale 1ns/1ps
 "$BIN/xsim.bat" acdc_fast -R
) > "$ROOT/out/acdc/fast/simulation.log" 2>&1
rg 'TB_PASS tb_axis_tensor_accel_acdc_fast' "$ROOT/out/acdc/fast/simulation.log"
echo ACDC_FAST_ALL_SIMS_PASS
