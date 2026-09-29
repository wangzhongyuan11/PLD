# 团队冻结交付

当前交付位于[c_v3](c_v3/README.md)。`MANIFEST.json`固定14个二进制/证据文件；运行包保持真实板测原SHA256，没有重新混装。

`test_fixtures/models.tar.gz`包含A最终包142文件及旧模型124个回归文件，供新克隆运行测试；旧模型只作回归，不用于当前C部署。展开文件由`python scripts/verify_delivery.py --prepare-models`恢复到被Git忽略的标准路径。原始数据集、完整第三方仓库和系统镜像不在交付中。

在仓库根目录运行`python scripts/verify_delivery.py`，同时核验产物、运行包内部、模型、构建来源与当前源码。源码只容忍CRLF/LF差异，任何其他变更须作为新版本重新验证。已有运行包的构建/板测来源标识保留，不因整理目录重写。
