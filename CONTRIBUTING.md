# 团队协作

`main`保存可供团队使用的版本。修改前`git pull --ff-only`并查看`git status`；各自在任务分支提交，通过Pull Request合并，例如`git switch -c feat/c-display-integration`。C从[交接文档](docs/C_HANDOFF.md)开始，硬件接口由B维护；对寄存器、时钟、DMA或输出语义的修改需同步合同。

先运行`python scripts/verify_delivery.py --prepare-models`准备最终与旧回归模型，再按根README安装主机依赖。源码按LF保存；冻结包里的原始字节和SHA256不修改。SSH凭据只从本机环境或交互输入取得，不写进代码、文档、提交或共享的测试输出。

| 改动 | 相关检查 |
|---|---|
| 文档/目录 | 链接、产物hash、`git diff --check`；无需重新上板或重建 |
| PS/参考逻辑 | 对应Python用例；量化/布局变化核对参考；影响硬件路径时做针对板测 |
| RTL/IP/BD/时钟 | 自校验xsim、布线后setup/hold/脉宽、bit/hwh配对、受影响数值和恢复板测 |
| 最终交付版本 | 用新目录保存产物、部署manifest和相应证据；不覆盖`delivery/c_v3/`后继续引用其1000帧结论 |

Windows构建使用Git Bash和Vivado 2026.1，入口见[环境说明](docs/validation/VIVADO_ENVIRONMENT.md)。示例：`D:\Git\bin\bash.exe scripts/run_sim.sh tile-fused`、`D:\Git\bin\bash.exe scripts/build.sh cnn`；调整工具位置时设置`VIVADO_BIN`。既有`vivado/ps7_config.tcl`足以重建当前BD；只有重新提取配置时才需本机教程。

`local/`、`out/`、`ip_repo/`、展开模型、原始ISIC数据和系统镜像不上传。`delivery/`是用户指定的团队冻结交付例外：只加入经过检查的小型运行包、模型回归材料和证据，并更新`delivery/MANIFEST.json`。不把实验输出全目录加入Git。队友可自行准备JPEG测试图；整套dev638原图/GT未放入Git，全集评估方法见[评估说明](docs/validation/EVALUATION.md)。

初次发布采用干净快照；早期本机历史含误收的Windows缓存，已保留在本地归档分支，不推送该分支或使用`git push --all/--mirror`。交付包中的旧commit ID仅是原实验的本机来源标识；跨仓库验证以源码和产物SHA256为准。之后正常保留团队提交历史，不做共享分支强推。
