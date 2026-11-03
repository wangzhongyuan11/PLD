@echo off
cd /d "%~dp0\..\.."
".venv\Scripts\python.exe" -B -m software.interactive_segmentation.server --model skin --port 8766
pause
