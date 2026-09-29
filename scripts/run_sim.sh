#!/usr/bin/env bash
# Command-line simulation loop (stage 0): xvlog -> xelab -> xsim -R.
# Verdict from the TB's own TB_PASS/TB_FAIL marker plus process exit code.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${VIVADO_BIN:-/d/vivado/2026.1/Vivado/bin}"
LOG_DIR="$ROOT/out/sim_logs"
mkdir -p "$LOG_DIR"

run_tb() {
  local tb="$1"; shift
  local files=()
  local f
  for f in "$@"; do files+=("$(cd "$ROOT" && realpath "$f")"); done
  local dir="$ROOT/out/sim_$tb"
  echo "=== $tb ==="
  rm -rf "$dir"; mkdir -p "$dir"
  (
    cd "$dir"
    "$BIN/xvlog.bat" -sv -nolog "${files[@]}"
    "$BIN/xelab.bat" -nolog -debug typical "$tb" -s "$tb" -timescale 1ns/1ps
    "$BIN/xsim.bat" "$tb" -R
  ) > "$LOG_DIR/$tb.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ] || ! grep -q "TB_PASS $tb" "$LOG_DIR/$tb.log"; then
    echo "FAIL $tb (rc=$rc), see $LOG_DIR/$tb.log"
    tail -40 "$LOG_DIR/$tb.log"
    exit 1
  fi
  grep -E "TB_PASS|phase .* OK|phase .* FAIL" "$LOG_DIR/$tb.log" | tail -30
  echo "PASS $tb"
}

cd "$ROOT"
case "${1:-all}" in
  all|axis)
    run_tb tb_axis_reg_slice rtl/axis_reg_slice.sv sim/tb_axis_reg_slice.sv
    run_tb tb_axis_incr rtl/axis_reg_slice.sv rtl/axis_incr.sv sim/tb_axis_incr.sv
    ;;
  numeric|tile|tile-full|tile-fused|tensor|control|bytes|accel) ;;
  *) echo "usage: run_sim.sh [all|axis|numeric|tile|tile-full|tile-fused|tensor|control|bytes|accel]" >&2; exit 2 ;;
esac
if [ "${1:-all}" = "all" ] || [ "${1:-all}" = "numeric" ]; then
  python scripts/gen_requant_vectors.py
  run_tb tb_requant_q24 rtl/requant_q24.sv sim/tb_requant_q24.sv
  python scripts/gen_mac_vectors.py
  run_tb tb_conv_mac_array rtl/conv_mac_array.sv sim/tb_conv_mac_array.sv
fi
if [ "${1:-all}" = "all" ] || [ "${1:-all}" = "tile" ] || [ "${1:-all}" = "tile-full" ] || [ "${1:-all}" = "tile-fused" ]; then
  if [ "${1:-all}" = "tile-full" ]; then python scripts/gen_tile_vectors.py --full
  elif [ "${1:-all}" = "tile-fused" ]; then python scripts/gen_tile_vectors.py --full-fused-only
  else python scripts/gen_tile_vectors.py; fi
  run_tb tb_conv_tile_engine rtl/conv_mac_array.sv rtl/requant_q24.sv rtl/conv_tile_engine.sv sim/tb_conv_tile_engine.sv
fi
if [ "${1:-all}" = "all" ] || [ "${1:-all}" = "tensor" ]; then
  python scripts/gen_tensor_vectors.py
  run_tb tb_tensor_tile_engine rtl/conv_mac_array.sv rtl/requant_q24.sv rtl/conv_tile_engine.sv rtl/tensor_tile_engine.sv sim/tb_tensor_tile_engine.sv
fi
if [ "${1:-all}" = "all" ] || [ "${1:-all}" = "control" ]; then
  run_tb tb_axi_lite_endpoint rtl/axi_lite_endpoint.sv sim/tb_axi_lite_endpoint.sv
fi
if [ "${1:-all}" = "all" ] || [ "${1:-all}" = "bytes" ]; then
  run_tb tb_axis_bytes rtl/axis_byte_pack.sv rtl/axis_byte_unpack.sv sim/tb_axis_bytes.sv
fi
if [ "${1:-all}" = "all" ] || [ "${1:-all}" = "accel" ]; then
  python scripts/gen_accel_vectors.py
  run_tb tb_axis_tensor_accel rtl/conv_mac_array.sv rtl/requant_q24.sv rtl/conv_tile_engine.sv rtl/tensor_tile_engine.sv rtl/axis_byte_pack.sv rtl/axis_byte_unpack.sv rtl/axi_lite_endpoint.sv rtl/axis_tensor_accel.sv sim/tb_axis_tensor_accel.sv
fi
echo "ALL_SIMS_PASS"
