# 2026-10-26 最新仓库源码

本目录标记此次 PR 的应用源码、服务配置与微信工程，和 `current_board_20261025/` 的历史板端采集记录分开。
`SOURCE_MAP.json` 记录仓库文件路径、LF 规范化 SHA256 与字节数，不是板端安装清单；本轮未重启服务、加载位流或重新采集板端文件。

新增内容包括 `wechat_miniprogram/` 的完整可编辑工程、真实服务结果合同与云回传修复，以及共享常驻 HDMI、模型交接和导出优化。
ACDC 的 `X-Auto-Display: false` 需要配套 web/engine 源码，省略时保留原行为。模型、位流、数据集和唯一正式 Release 指针均沿用现有交付。

微信工程从本机独立开发副本的当前源码导入；独立副本的提交记录在清单中标记。没有复制其 `.git`、AppSecret/访问码配置、日志、演示数据产物或微信工具私有配置。
`wechat_miniprogram/bridge/config.example.json` 是模板，不能直接用于运行。

仓库根目录校验源码：

```powershell
python scripts/verify_current_board_sources.py --manifest deployment/repository_20261026/SOURCE_MAP.json
```

在 `wechat_miniprogram/bridge/` 运行 `python -m unittest discover -s tests -p 'test_*.py'`；
在 `wechat_miniprogram/` 运行 `python tools/check_miniprogram.py`、`node tools/test_api_js.cjs` 和 `node tools/test_regressions.cjs`。
主程序定向回归覆盖网页导出、入口交接、显示子进程和任务级显示开关。Windows 上两个 Unix-socket broker 用例按平台跳过，不能据此称 Linux 共享服务已验收。

原板端记录中的物理显示、触控和时延结论只适用于各自记录的版本。小程序回传提速的旧模拟结果也不作为真实微信网络的性能承诺。
后续实际部署时应生成同次安装的文件身份与回退记录；本 PR 不覆盖历史采集 hash 或自动变更正在运行的应用。
