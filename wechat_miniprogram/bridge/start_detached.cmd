@echo off
rem Start the bridge in a detached minimized window (survives parent session).
rem Stop: close the minimized "PLD-bridge" window or kill python.exe in Task Manager.
cd /d %~dp0
if not exist config.json (
  echo [ERROR] config.json not found
  pause
  exit /b 1
)
if not exist logs mkdir logs
start "PLD-bridge" /min cmd /c ""D:\conda\python.exe" bridge.py --config config.json >>"logs\bridge_stdout.log" 2>&1"
echo Bridge started in detached window. Log: logs\bridge_stdout.log
