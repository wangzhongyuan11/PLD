# B计算子系统交付合同

本文件维护PS集成细节，当前冻结版本入口见[C交接说明](../C_HANDOFF.md)和[团队运行包](../../delivery/c_v3/README.md)。最终模型已完成638图与1000帧板测；在线P95目标仍有缺口，具体范围见[最终状态](../FINAL_MODEL_STATUS.md)。C按此合同完成后处理/HDMI/UI集成。

## 固定输入与输出

模型为A最终 `soup_v5_a050`，输入 `[1,3,256,256]` NCHW INT8，输出 `[1,1,256,256]` NCHW INT8。所有Conv、ReLU/尺度映射、Pool、nearest Upsample、Concat均在PL；PS负责原图预处理、LUT参数、halo拷贝、控制/DMA和拼接。不可用CPU网络结果代替PL执行，也不可逐层注入golden代替单一首层输入。

原图按EXIF方向→RGB→双线性letterbox→float32除255→首层input_scale最近偶数舍入/饱和。最终head保持有符号原始logit量化值，不是uint8概率或二值掩膜。默认概率0.45对应 `q >= -1`，head尺度0.1995087721216397。有效区外必须按背景处理；面积分母为有效区像素数，未经标定不转为物理面积。

## 一次安装与运行

使用 `scripts/package_cnn_board.py` 产生的配套部署包，在新的板端目录解包并核对 `deployment_manifest.json` 的逐文件SHA256。bit/hwh同一构建，manifest通过源码、配对和布线时序门禁；不要组合来自不同构建的文件。板端Python为 `/usr/local/share/pynq-venv/bin/python3`，登录shell初始化XRT，按 `board/README.md` 使用sudo。

可选本机加速库由部署内源码在板端构建：

```bash
/usr/local/share/pynq-venv/bin/python3 scripts/build_native_copy.py
/usr/local/share/pynq-venv/bin/python3 scripts/build_native_control.py
/usr/local/share/pynq-venv/bin/python3 scripts/build_native_resize.py
```

第三项要求板端已验证的Pillow 9.0.1，记录并校验Pillow核心二进制、C源码、包装器与库hash。采用标准Pillow或两线程Pillow缩放时不需要该库；原生resize与 `--parallel-resize` 互斥，不能静默切换已测量的后端。原生行缩放及快速RGB8量化的当前组合已通过最终v3整网和1000帧板测。

常规推理无需外部GT，golden不参与网络输入：

```bash
/usr/local/share/pynq-venv/bin/python3 board/infer_images.py image1.jpg image2.jpg --bit overlay/dma_top.bit --output results/run01 --native-copy out/native/tensor_copy.so --native-control out/native/tensor_control.so --tx-range-sync --fuse-maps --fuse-decoder --fast-quantize --native-resize out/native/rgb_resize.so
```

`--output`必须是新目录。模式5融合要求ABI≥1.1，模式6的`--fuse-decoder`要求ABI≥1.2；驱动在旧硬件上明确拒绝。驱动从配套HWH取得预期FCLK0并与实际值比较，不能只改变PS时钟而继续使用旧时序报告。命令、寄存器和ready失效规则见[寄存器合同](ACCELERATOR_REGISTERS.md)。

## C消费结果

每个成功帧保存 `frame_000001.bin` 与同名JSON。bin固定65536字节；按有符号int8、NCHW解释。JSON包含：

| 字段 | C使用方式 |
|---|---|
| `frame_id`、`source_name`、`status` | 匹配输入与显示帧；只有success可提交正常结果 |
| `shape`、`dtype`、`layout` | `[1,1,256,256]`、int8、NCHW |
| `model_spec_sha256`、`model_manifest_sha256` | 防止参数/解释口径混版 |
| `output_sha256` | 核验二进制结果身份 |
| `threshold_probability`、`threshold_q`、`comparison` | 默认0.45、-1、`>=`；不要再按旧q≥0处理 |
| `head_output_scale` | 量化logit反量化尺度 |
| `valid_rect` | `[left,top,width,height]`，右/下边界不包含；补边不计面积/指标 |
| `input_metadata` | 源图/量化输入hash、原始及缩放尺寸、有效像素数 |
| `online_b_seconds` | 解码后RGB预处理至B原始head；不含C处理/显示、JPEG解码和文件IO |

同进程集成可复用 `Model`、`TileAccelerator` 和 `PLGraph`，参考 `board/infer_images.py`。`PLGraph.run`返回独立的head数组及报告，C可以保留此数组，不会被下次DMA覆盖。底层TX/RX staging仅归B驱动使用，不能保留其视图，不能同时使用 `ol.axi_dma` 或启动另一驱动争用同一DMA。

失败会非零退出或抛出 `AcceleratorFault`，附核心/DMA状态和恢复结果。先确认DMA复位/重配置停止DDR访问，再释放缓冲区。已完成帧可保留；失败帧不能伪装成功。长时运行的调用者应记录失败并在明确恢复后重新加载参数/LUT/输入，禁止无限重试。

## 复现与验收

主机参考、RTL仿真、构建、逐层板测、全集评估分别执行并保存证据。最终精度集合为开发选模用dev638，五阈值与来源/面积组另报，不能称独立测试集。实际板测使用原JPEG在线预处理，仅向PL注入首层输入，逐字节比较原始head。

最终B验收还要求在线≥1 FPS、P95≤1秒，dev638全覆盖、≥1000帧且≥600秒、故障恢复、内存趋势和精确部署审查；当前未完成项见实施状态。CPU加速比使用同模型板载NumPy整数参考，不能泛化为相对优化ARM库的优势。C后处理/交互/HDMI不计入本轮B验收，也不能在B报告中宣称其已通过。

参考实现依据：Pillow 9.0.1的 [RGB内存布局](https://github.com/python-pillow/Pillow/blob/9.0.1/src/libImaging/Imaging.h)、[行指针导出](https://github.com/python-pillow/Pillow/blob/9.0.1/src/_imaging.c)及[Q22重采样规则](https://github.com/python-pillow/Pillow/blob/9.0.1/src/libImaging/Resample.c)。原生候选保留横向8bit中间值与逐级舍入，版本/hash门禁和逐像素实测共同约束兼容性。
