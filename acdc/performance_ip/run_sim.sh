#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="${VIVADO_BIN:-/d/vivado/2026.1/Vivado/bin}"
OUT="$ROOT/out/acdc/mac128_sim"
mkdir -p "$OUT"
cd "$OUT"
"$BIN/xvlog.bat" -sv -nolog "$ROOT/acdc/performance_ip/conv_mac_128.sv" "$ROOT/acdc/performance_ip/tb_conv_mac_128.sv" > compile.log 2>&1
"$BIN/xelab.bat" -nolog tb_conv_mac_128 -s mac128 -timescale 1ns/1ps >> compile.log 2>&1
"$BIN/xsim.bat" mac128 -R > simulation.log 2>&1
rg 'TB_PASS conv_mac_128' simulation.log
