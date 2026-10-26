@echo off
rem PLD 小程序桥接启动脚本。先复制 config.example.json 为 config.json 并填写
rem app_secret 与 hub_token（见 ..\docs\DEPLOY.md）。
setlocal
cd /d %~dp0
set PY=D:\conda\python.exe
if not exist "%PY%" set PY=python
if not exist config.json (
  echo [错误] 缺少 config.json，请先复制 config.example.json 并填写密钥。
  pause
  exit /b 1
)
"%PY%" bridge.py --config config.json
pause
