#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="${VIVADO_BIN:-/d/vivado/2026.1/Vivado/bin}"
PY="${ACDC_PYTHON:-$ROOT/acdc/.venv/Scripts/python.exe}"
OUT="${ACDC_SIM_OUTPUT:-$ROOT/out/acdc/optimization/executor_netlist_sim}"
NETLIST="${ACDC_NETLIST:-$ROOT/out/acdc/optimization/executor_netlist_probe/netlist.v}"
test -f "$NETLIST"
mkdir -p "$OUT"
cd "$ROOT"
MODEL_ARGS=()
if [[ -n "${ACDC_MODEL:-}" ]]; then
  MODEL_ARGS=(--model "$ACDC_MODEL" --model-spec-sha "${ACDC_MODEL_SHA:?ACDC_MODEL_SHA is required}")
fi
# This bounded netlist TB uses a reduced grid; full-grid correctness is checked
# separately by RTL and physical-board runs. Preserve the fixture's identity.
"$PY" -m acdc.optimization.gen_executor_vectors "$OUT" --p 4 --divisor 8 "${MODEL_ARGS[@]}"
cd "$OUT"
"$BIN/xvlog.bat" -sv -nolog "$NETLIST" "$ROOT/acdc/optimization/sim/datamover_memory.sv" "$ROOT/acdc/optimization/sim/tb_executor_netlist.sv" > compile.log 2>&1
"$BIN/xvlog.bat" -nolog "$BIN/../data/verilog/src/glbl.v" >> compile.log 2>&1
"$BIN/xelab.bat" -nolog -L unisims_ver tb_executor_netlist glbl -s netlist -timescale 1ns/1ps >> compile.log 2>&1
"$BIN/xsim.bat" netlist -R > simulation.log 2>&1
rg 'TB_PASS executor_functional_netlist' simulation.log
