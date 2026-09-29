# B→C交付：A最终模型的板上推理接口

交付版本：2026-09-29冻结的125 MHz、ABI 1.2 `soup_v5_a050`，B最小PS7/DMA/CNN overlay。用户决定先向C交付已跑通版本，暂不继续追求更高性能。C的PL后处理、显示BD及HDMI/UI仍由C集成，不能把本页的离线示意图当成C显示验收。

## 取用哪个包

| 项目 | 冻结值 |
|---|---|
| 源码、模型、bit/hwh交付包 | [delivery/c_v3/runtime.tar.gz](../delivery/c_v3/runtime.tar.gz)；SHA256 `7651de35eade4af88d6ea79407e06d6ee4c87e8847a0bd64b732698f51b42873` |
| 板上已部署目录 | `/home/xilinx/jupyter_notebooks/pld/final_mode6_fastq_20260929_v3/`；部署清单161文件逐项hash通过 |
| 同次构建bit/hwh/XSA | 团队[hardware目录](../delivery/c_v3/hardware/)保存XSA/报告；bit/hwh在runtime包内；bit `0df9c718172d2d496f28a93719a2858d3a9b366ea9145d58bf0a3788687f0baf`；hwh `7adfbe2d757dd09f5dab77562b2fb4db2ef86435db2fb60bf3ca85a02281da13`；xsa `834270c2016b6b8a5739b0fdd44b1026fffc45cba980f84fc5380861c27e037d` |
| 模型 | `role_a_final/int8`，spec SHA256 `fe137ca5b128fc1983fabad9bee39b3208cda4256e0d04c4e27cef5c07666aef`；124文件由`reference/final_manifest.json`固定 |
| 硬件条件 | `user:pld:axis_tensor_accel:1.0`，ABI寄存器`0x00010002`，FCLK0 125 MHz；AXI-Lite `0x43C00000`、DMA `0x40400000`；同次构建WNS +0.013 ns、WHS +0.049 ns、WPWS +2.75 ns；7108 Slice LUT、67 BRAM36等效、53 DSP，低于已协商的B侧预算32,000 LUT/90 BRAM36/96 DSP |

包中有`board/`、`reference/`、`role_a_final/int8/`、`overlay/dma_top.bit`及同名`.hwh`、原生库C源码和构建脚本。板上原生`.so`在该部署目录按`/usr/local/share/pynq-venv/bin/python3 scripts/build_native_copy.py`、`build_native_control.py`、`build_native_resize.py`生成，加载时会检查源码、库、Pillow 9.0.1和构建清单。不要把另一版的bit、hwh、库或模型拼入这个包。若C重建含显示模块的BD，须重新检查资源、布线时序和板上数值；本bit仅证明B最小BD。

## C直接消费的结果

无需GT的入口是`board/infer_images.py`。每幅原始JPEG输出独立的`frame_000001.bin`和同名JSON；`.bin`恰为65536字节、有符号INT8、NCHW `[1,1,256,256]`的**原始logit**，不是二值mask、RGB或诊断类别。JSON包含`frame_id`、`status`、`output_sha256`、模型/spec/manifest hash、`shape`、`dtype`、`layout`、`head_output_scale`、概率和整数阈值、`valid_rect`、原图/letterbox几何、输入hash以及计时。`PLGraph.run`返回的head为独立数组；C异步持有它或再次送入PL时仍须保持单一DMA/overlay拥有者，不能持有B内部staging buffer视图。

最终模型主阈值为概率0.45、`head_output_scale=0.1995087721216397`、有符号整数比较`q>=-1`。C生成mask时必须先按**signed** INT8比较，再与`valid_rect=[left,top,width,height]`相与；补边强制背景。面积比例分母为有效区域像素数，不是全图65536；没有像素尺度校准时不报告物理面积。要把256×256的mask叠到原图，应先去掉letterbox补边，再按JSON中的原图宽高回映射。详见[完整寄存器和包合同](design/ACCELERATOR_REGISTERS.md)。

同一进程集成时，C可复用`Model`、`NativeResize`、`TileAccelerator`和`PLGraph`；`board/infer_images.py`是已在板上执行的端到端调用示例。核心调用为`graph = PLGraph(model, device, fuse_maps=True, fuse_decoder=True)`，将`letterbox_int8`的NCHW输入和含模型hash、输入hash、有效区的`input_metadata`送入`head, timing = graph.run(data, frame=frame_id, input_metadata=metadata)`；`timing["result"]`就是上述结果合同。完整命令使用`--native-copy out/native/tensor_copy.so --native-control out/native/tensor_control.so --native-resize out/native/rgb_resize.so --tx-range-sync --fuse-maps --fuse-decoder --fast-quantize`。`--output`必须是尚不存在的目录，失败非零退出，不复用部分结果。板上通过`board_ssh.py run --sudo`或初始化XRT的交互`sudo -E`运行，不用系统Python。

在已部署目录中的最小实测命令（把输入换成C自己的JPEG，并选新的输出目录）：

```bash
cd /home/xilinx/jupyter_notebooks/pld/final_mode6_fastq_20260929_v3
sudo -E /usr/local/share/pynq-venv/bin/python3 board/infer_images.py \
  ../final_dataset_638/dataset/images/ISIC_0000003.jpg \
  --bit overlay/dma_top.bit --output out/c_smoke_new \
  --native-copy out/native/tensor_copy.so \
  --native-control out/native/tensor_control.so \
  --native-resize out/native/rgb_resize.so \
  --tx-range-sync --fuse-maps --fuse-decoder --fast-quantize
```

## 真实板测与可查看样例

同一v3部署从dev638的638张原始JPEG在线预处理并完成1000帧/1483.827秒；638张唯一图的输入与PL head均和冻结INT8参考一致，第二循环362张重复图也一致。独立审查比对同顺序旧版报告，1000/1000个head SHA256与指标相同；短包、缺包超时、计算中复位及下一帧恢复通过。宏平均Dice 0.8420508068333175、IoU 0.7583410835598974，dev638是开发集而非独立测试集。核心中位0.700375秒、P95 0.726514秒、1.422 FPS；从**已解码RGB**进入B预处理到独立head的在线B中位0.758002秒、P95 **1.265720秒**。所以数值、连续运行和恢复可交付，项目自定的在线P95≤1秒尚未达标；JPEG解码、文件IO、C后处理与HDMI不在此计时。报告为`out/final_model/fastq_stability1000.json`和`fastq_stability1000_audit.json`，版本边界见[最终模型状态](FINAL_MODEL_STATUS.md)。

此外，**无需GT的推理入口**在同一部署上独立运行了2016、2017 validation、2017 exposed validation的真实JPEG。下载`.bin`及JSON后逐字节hash与冻结参考相同，四图均返回65536字节。样例在`out/final_model/c_handoff_3images_20260929/`和`c_handoff_highres_20260929/`；对应打包备份的SHA256分别为`2ae8149167cda588798b0aedbc010c943348eb861845eee754c857d0451f3cbc`、`a09c2f17c26f2daa7431b98120c2eeb96ee9dba7ad62c00d38c034abe9fb11e7`。ISIC_0000003和高分辨率ISIC_0015037的板上结果离线展示图为`out/final_model/c_handoff_ISIC_0000003.png`和`c_handoff_ISIC_0015037.png`；对应GT下Dice分别为0.967345、0.959358。示意图只是下载后渲染，不是C的PL后处理或HDMI结果。不同图可能分割较差，不能把两张展示样例的精度当作全集精度。

## GitHub协作版本的文件位置

本页历史本机`out/final_model/`路径对应的团队副本已经放入[delivery/c_v3/](../delivery/c_v3/README.md)：`validation/reports.tar.gz`保存原1000帧JSON/JSONL及审查结果，`validation/summary.json`可直接查看；`examples/`保存上述两个样例压缩包及PNG。在仓库根目录执行`python scripts/verify_delivery.py`验证整个交付。运行包只复制、不重打包，SHA256与原板测相同。
