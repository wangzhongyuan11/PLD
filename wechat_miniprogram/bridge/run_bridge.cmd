@echo off
rem Runs the bridge directly (no extra window via start) with output to log.
rem Used by the PLDBridge scheduled task; do not close unexpectedly.
cd /d %~dp0
if not exist config.json exit /b 1
if not exist logs mkdir logs
"D:\conda\python.exe" bridge.py --config config.json >>"logs\bridge_stdout.log" 2>&1
