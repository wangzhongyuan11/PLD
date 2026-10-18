#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="${VIVADO_BIN:-/d/vivado/2026.1/Vivado/bin}"
PY="${ACDC_PYTHON:-$ROOT/acdc/.venv/Scripts/python.exe}"
OUT="$ROOT/out/acdc/optimization/streams_sim"
mkdir -p "$OUT"
cd "$ROOT"
"$PY" -m acdc.optimization.gen_stream_vectors "$OUT"
cd "$OUT"
"$BIN/xvlog.bat" -sv -nolog "$ROOT/acdc/optimization/rtl/pool_stream.sv" "$ROOT/acdc/optimization/rtl/stream_fork.sv" "$ROOT/acdc/optimization/rtl/input_transform.sv" "$ROOT/acdc/optimization/sim/tb_pool_stream.sv" "$ROOT/acdc/optimization/sim/tb_input_transform.sv" > compile.log 2>&1
for TOP in pool_stream input_transform; do
  "$BIN/xelab.bat" -nolog "tb_$TOP" -s "$TOP" -timescale 1ns/1ps >> compile.log 2>&1
  "$BIN/xsim.bat" "$TOP" -R > "${TOP}.log" 2>&1
  rg "TB_PASS $TOP" "${TOP}.log"
done
