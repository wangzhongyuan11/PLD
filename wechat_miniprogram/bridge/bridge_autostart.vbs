' PLD 桥接静默自启（当前用户登录时运行，无需管理员）。
' 停用方式：删除「启动」文件夹里的本文件副本，或结束 pythonw.exe 进程。
' 手动重启：双击 bridge/start_detached.cmd。
CreateObject("WScript.Shell").Run _
  """D:\conda\pythonw.exe"" ""D:\PLD小程序\bridge\bridge.py"" --config ""D:\PLD小程序\bridge\config.json""", _
  0, False
