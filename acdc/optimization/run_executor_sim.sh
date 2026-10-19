#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="${VIVADO_BIN:-/d/vivado/2026.1/Vivado/bin}"
PY="${ACDC_PYTHON:-$ROOT/acdc/.venv/Scripts/python.exe}"
P="${1:-4}"
DIVISOR="${2:-8}"
case "$P" in 4|6) ;; *) echo "P must be 4 or 6" >&2; exit 2;; esac
OUT="${ACDC_SIM_OUTPUT:-$ROOT/out/acdc/optimization/executor_p${P}_d${DIVISOR}_sim}"
mkdir -p "$OUT"
cd "$ROOT"
ARGS=()
if [[ -n "${ACDC_MODEL:-}" ]]; then ARGS+=(--model "$ACDC_MODEL" --model-spec-sha "$ACDC_MODEL_SHA"); fi
"$PY" -m acdc.optimization.gen_executor_vectors "$OUT" --p "$P" --divisor "$DIVISOR" "${ARGS[@]}"
cd "$OUT"
FILES=(conv_array conv_compute tile_banks tile_conv stream_fork pool_stream dma_transfer tile_writer parameter_loader input_transform input_loader lut_loader dma_read_arbiter tile_job_queue tile_executor)
SOURCES=()
for FILE in "${FILES[@]}"; do SOURCES+=("$ROOT/acdc/optimization/rtl/$FILE.sv"); done
"$BIN/xvlog.bat" -sv -nolog "${SOURCES[@]}" "$ROOT/rtl/requant_q24.sv" "$ROOT/acdc/optimization/sim/datamover_memory.sv" "$ROOT/acdc/optimization/sim/tb_tile_executor.sv" > compile.log 2>&1
"$BIN/xelab.bat" -nolog "tb_tile_executor_p$P" -s executor -timescale 1ns/1ps >> compile.log 2>&1
"$BIN/xsim.bat" executor -R > simulation.log 2>&1
rg 'TB_PASS tile_executor' simulation.log
