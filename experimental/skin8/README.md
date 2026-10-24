# Skin8 已部署源码快照

此目录保存 2026-10-23 已验收并部署的 Skin8 源码和必要证据，2026-10-24 恢复 HDMI 生命周期。RTL 已逐文件匹配已部署 bitstream 的 build_receipt.json；不包含后来时序失败的 reader/signature pipeline 实验改动。

已有固定输入板测：皮肤镜 1000/1000 head 一致，1000 帧 93.947 秒，10.644 FPS；CAMUS 100/100 head 一致，3.659 FPS。不是数据集准确率或浏览器端到端时间。本次不重复修改或构建 RTL。

HDMI 新功能、部署依赖、验证范围与回退见 ../../deployment/hdmi_touch_20261024/README.md。二进制位流和编译参数在同次交接压缩包中；Git 仅保存源码和小型证据。
