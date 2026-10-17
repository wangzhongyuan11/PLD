# B → C 当前交接

唯一正式版本见 [交付说明](../delivery/pld_dual_20261017/README.md)。双模型共享单 DMA 所有者，网页与独立板测互斥；普通网页产物与交互细化产物的范围见 [架构](IMPLEMENTATION.md)。部署见 [DEPLOYMENT](DEPLOYMENT.md)，最新实测见 [验证](validation/CLEANUP_20261017.md)。

皮肤镜 head 是有符号 INT8 `[1,1,256,256]`，尺度 0.1995087721216397，最终阈值 q≥−1，去除补边后按元数据还原；CAMUS head 是 INT8 `[1,4,256,256]`，必须使用其独立 spec、尺度与几何合同。不能混用两者模型、HWH、位流或后处理。

寄存器、AXIS、TLAST/TKEEP、有界超时和恢复合同见 [加速器寄存器](design/ACCELERATOR_REGISTERS.md)、[PS 集成](design/PS_INTEGRATION.md)。原 1000 帧/dev638 属于既有皮肤镜核心证据，不代表本次整个双模型工作台完成长稳或交互质量评估。历史 c_v3 交接可从 Git 提交 50d07721e6349dfec101732f8673e077a00d7ce9 查阅。
