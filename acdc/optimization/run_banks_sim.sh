#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="${VIVADO_BIN:-/d/vivado/2026.1/Vivado/bin}"
P="${1:-4}"
case "$P" in 4|6) ;; *) echo "P must be 4 or 6" >&2; exit 2;; esac
OUT="$ROOT/out/acdc/optimization/banks_p${P}_sim"
mkdir -p "$OUT"
cd "$OUT"
"$BIN/xvlog.bat" -sv -nolog "$ROOT/acdc/optimization/rtl/tile_banks.sv" "$ROOT/acdc/optimization/sim/tb_tile_banks.sv" > compile.log 2>&1
"$BIN/xelab.bat" -nolog "tb_tile_banks_p$P" -s banks -timescale 1ns/1ps >> compile.log 2>&1
"$BIN/xsim.bat" banks -R > simulation.log 2>&1
rg 'TB_PASS tile_banks' simulation.log
