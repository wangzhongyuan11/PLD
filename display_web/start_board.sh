#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
runtime="${1:-/home/xilinx/jupyter_notebooks/pld/final_mode6_fastq_20260929_v3}"
exec sudo -E /usr/local/share/pynq-venv/bin/python3 server.py --host 0.0.0.0 --runtime "$runtime"
