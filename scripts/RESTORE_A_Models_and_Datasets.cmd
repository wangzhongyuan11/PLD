@echo off
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
  python "RESTORE_A_Models_and_Datasets.py"
) else (
  py -3 "RESTORE_A_Models_and_Datasets.py"
)
set "RESTORE_EXIT=%ERRORLEVEL%"
pause
exit /b %RESTORE_EXIT%
