# pld_dual_20261017

同一系统包含两个冻结模型与对应位流：皮肤镜 125 MHz、CAMUS 100 MHz；统一工作台使用已验证的 interactive v4。严格文件清单与附件整体 SHA256 见 [MANIFEST.json](MANIFEST.json)。不依据目录日期推断版本。

从 [Release](https://github.com/wangzhongyuan11/PLD/releases/tag/pld-dual-20261017) 下载 `pld_dual_20261017_assets.zip` 至仓库 `out/`，在仓库根执行 `python scripts/release_assets.py out/pld_dual_20261017_assets.zip --prepare`。已存在但不同的本地模型会被拒绝覆盖；先保留并核查冲突。

附件：`skin/` 和 `camus/` 是逐字节冻结运行包；Skin 额外附带三套已有 ARM 原生库和源码绑定证书，不能在 x86 上加载。`models/` 是完整模型及旧回归 fixture；`hardware/` 是配对 bit/HWH/XSA、构建源清单、时序、资源及 DRC；`hub/`、`web/`、`units/` 是统一应用；`fixtures/camus/` 是公开开发小样本与隔离 nibabel wheel；`tools/` 是部署/板测脚本快照，以仓库当前工具为维护入口。

PYNQ 3.1.1 运行目录不得让同名 XSA 与 bit 相邻；CAMUS 的 XSA 在 `overlay/evidence/`，硬件完整副本在独立 `hardware/`。不修改冻结 manifest 的源字节或历史 provenance 字段。

部署与回退见 [部署文档](../../docs/DEPLOYMENT.md)。结构与接口见 [架构](../../docs/IMPLEMENTATION.md)。本次测试范围、耗时和未验项目见 [验证记录](../../docs/validation/CLEANUP_20261017.md)。原始训练/开发全集、访问码及网页历史均不在附件。
