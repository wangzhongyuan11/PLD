#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${VIVADO_BIN:-/d/vivado/2026.1/Vivado/bin}"
PY="${ACDC_PYTHON:-$ROOT/acdc/.venv/Scripts/python.exe}"
cd "$ROOT"
"$PY" -B -m acdc.gen_vectors
OUT="$ROOT/out/acdc/sim_accel"
mkdir -p "$OUT"
(
 cd "$OUT"
 "$BIN/xvlog.bat" -sv -nolog "$ROOT/rtl/conv_mac_array.sv" "$ROOT/rtl/requant_q24.sv" \
  "$ROOT/rtl/axi_lite_endpoint.sv" "$ROOT/rtl/axis_byte_pack.sv" "$ROOT/rtl/axis_byte_unpack.sv" \
  "$ROOT/acdc/rtl/conv_tile_engine_acdc.sv" "$ROOT/acdc/rtl/tensor_tile_engine_acdc.sv" \
  "$ROOT/acdc/rtl/axis_tensor_accel_acdc.sv" "$ROOT/acdc/sim/tb_axis_tensor_accel_acdc.sv"
 "$BIN/xelab.bat" -nolog -debug typical tb_axis_tensor_accel_acdc -s acdc -timescale 1ns/1ps
 "$BIN/xsim.bat" acdc -R
) > "$ROOT/out/acdc/simulation.log" 2>&1
rg 'TB_PASS tb_axis_tensor_accel_acdc' "$ROOT/out/acdc/simulation.log"
echo ACDC_ALL_SIMS_PASS
