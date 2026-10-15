#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${VIVADO_BIN:-/d/vivado/2026.1/Vivado/bin}"
PY="${CAMUS_PYTHON:-/d/conda/python.exe}"
cd "$ROOT"
"$PY" -B -m camus.gen_vectors
OUT="$ROOT/out/camus/sim_accel"
mkdir -p "$OUT"
(
 cd "$OUT"
 "$BIN/xvlog.bat" -sv -nolog "$ROOT/rtl/conv_mac_array.sv" "$ROOT/rtl/requant_q24.sv" \
  "$ROOT/rtl/axi_lite_endpoint.sv" "$ROOT/rtl/axis_byte_pack.sv" "$ROOT/rtl/axis_byte_unpack.sv" \
  "$ROOT/camus/rtl/conv_tile_engine_camus.sv" "$ROOT/camus/rtl/tensor_tile_engine_camus.sv" \
  "$ROOT/camus/rtl/axis_tensor_accel_camus.sv" "$ROOT/camus/sim/tb_axis_tensor_accel_camus.sv"
 "$BIN/xelab.bat" -nolog -debug typical tb_axis_tensor_accel_camus -s camus -timescale 1ns/1ps
 "$BIN/xsim.bat" camus -R
) > "$ROOT/out/camus/simulation.log" 2>&1
rg 'TB_PASS tb_axis_tensor_accel_camus' "$ROOT/out/camus/simulation.log"
echo CAMUS_ALL_SIMS_PASS
