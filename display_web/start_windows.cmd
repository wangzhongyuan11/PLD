@echo off
cd /d "%~dp0"
set "PLD_WEB_PY=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if exist "%PLD_WEB_PY%" (
  "%PLD_WEB_PY%" server.py --open
) else (
  python server.py --open
)
pause
