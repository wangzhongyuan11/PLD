# 团队协作

`main`保存可供团队使用的版本。修改前`git pull --ff-only`并查看`git status`；各自在任务分支提交，通过Pull Request合并，例如`git switch -c feat/c-display-integration`。C从[交接文档](docs/C_HANDOFF.md)开始，硬件接口由B维护；对寄存器、时钟、DMA或输出语义的修改需同步合同。

先运行`python scripts/release_assets.py out/pld_dual_20261017_assets.zip --prepare`准备最终与旧回归模型，再按根README安装主机依赖。源码按LF保存；冻结包里的原始字节和SHA256不修改。SSH凭据只从本机环境或交互输入取得，不写进代码、文档、提交或共享的测试输出。

| 改动 | 相关检查 |
|---|---|
| 文档/目录 | 链接、产物hash、`git diff --check`；无需重新上板或重建 |
| PS/参考逻辑 | 对应Python用例；量化/布局变化核对参考；影响硬件路径时做针对板测 |
| RTL/IP/BD/时钟 | 自校验xsim、布线后setup/hold/脉宽、bit/hwh配对、受影响数值和恢复板测 |
| 最终交付版本 | 用新目录保存产物、部署manifest和相应证据；不覆盖`delivery/pld_dual_20261017/`后继续引用其1000帧结论 |

Windows构建使用Git Bash和Vivado 2026.1，入口见[环境说明](docs/validation/VIVADO_ENVIRONMENT.md)。示例：`D:\Git\bin\bash.exe scripts/run_sim.sh tile-fused`、`D:\Git\bin\bash.exe scripts/build.sh cnn`；调整工具位置时设置`VIVADO_BIN`。既有`vivado/ps7_config.tcl`足以重建当前BD；只有重新提取配置时才需本机教程。

`local/`、`out/`、`ip_repo/`、展开模型、原始 ISIC 数据和系统镜像**不进 Git**。经用户授权的公开数据集（ISIC 2016/2017 Task 1 为 CC0；CAMUS 为 CC BY-NC-SA 4.0，仅非商业科研）在唯一 Release 资产中按其原许可发布，不改变第三方条款。`delivery/`只跟踪小型清单、说明、部署模板与证据；模型/运行包不直接加入 Git，通过唯一 GitHub Release 附件发布，并更新`delivery/pld_dual_20261017/MANIFEST.json`。不把实验输出全目录加入Git。队友可自行准备JPEG测试图；整套dev638原图/GT未放入Git，全集评估方法见[评估说明](docs/validation/EVALUATION.md)。

初次发布采用干净快照；早期本机历史含误收的Windows缓存，已保留在本地归档分支，不推送该分支或使用`git push --all/--mirror`。交付包中的旧commit ID仅是原实验的本机来源标识；跨仓库验证以源码和产物SHA256为准。之后正常保留团队提交历史，不做共享分支强推。

## A 组协作（pld-medical/）

A 组的算法、接口契约与产物索引位于 pld-medical/；A 的本地细则只在 pld-medical/AGENTS.md 维护，根 AGENTS.md 不重复导入。本仓库对 A 只跟踪协作说明、接口文档与产物索引：git clone 不含 A 的训练源码、数据集、.venv 或模型权重。A 的完整电脑端接口与三个模型运行闭包在唯一 Release 资产 pld-dual-20261017 的附件中，下载后独立解压运行，入口见 docs/A_WORKSPACE.md。不因 A 组改动替换 delivery/current.json 或板端正式交付。
