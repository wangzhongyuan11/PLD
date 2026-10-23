#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="${VIVADO_BIN:-/d/vivado/2026.1/Vivado/bin}"
P="${1:-4}"
case "$P" in 4|6) ;; *) echo "P must be 4 or 6" >&2; exit 2;; esac
OUT="$ROOT/out/acdc/optimization/array_p${P}_sim"
mkdir -p "$OUT"
cd "$OUT"
"$BIN/xvlog.bat" -sv -nolog "$ROOT/acdc/optimization/rtl/conv_array.sv" "$ROOT/acdc/optimization/sim/tb_conv_array.sv" > compile.log 2>&1
"$BIN/xelab.bat" -nolog "tb_conv_array_p$P" -s array -timescale 1ns/1ps >> compile.log 2>&1
"$BIN/xsim.bat" array -R > simulation.log 2>&1
rg 'TB_PASS conv_array' simulation.log
