# C可运行交付：c_v3_20260929

本目录随Git完整克隆。模型为A最终`soup_v5_a050`，PYNQ-Z2、PYNQ 3.1.1、Python 3.10.4、Pillow 9.0.1，125MHz / ABI 1.2。无需GT即可运行自己的JPEG；输出signed INT8 head和结果JSON，C再做后处理、显示及集成。

| 内容 | 用途 |
|---|---|
| [runtime.tar.gz](runtime.tar.gz) | 配套bit/hwh、最终模型、PS运行代码、三块原生库源码/构建脚本；SHA256 `7651de35eade4af88d6ea79407e06d6ee4c87e8847a0bd64b732698f51b42873` |
| [hardware/](hardware/) | 同次XSA、setup/hold/脉宽和资源/DRC报告、build manifest；build_sources.tar.gz保留构建时精确源码字节 |
| [validation/summary.json](validation/summary.json) | 独立审查摘要；reports.tar.gz保存1000帧逐帧报告、审查结果和逐层/恢复证据 |
| [examples/](examples/) | 真实板上下载的head/JSON和离线叠加PNG；不代表HDMI已实现 |

## 三步运行

1. 在主机仓库根目录运行`python scripts/verify_delivery.py`，看到`DELIVERY_PASS`；把本目录`runtime.tar.gz`传到板上。
2. 在板上新建独立目录并解包，随后从包内源码构建原生库。不要覆盖旧部署：

```bash
mkdir -p ~/jupyter_notebooks/pld
cd ~/jupyter_notebooks/pld
mkdir c_v3_team
tar -xzf /path/to/runtime.tar.gz -C c_v3_team
cd c_v3_team
/usr/local/share/pynq-venv/bin/python3 scripts/build_native_copy.py
/usr/local/share/pynq-venv/bin/python3 scripts/build_native_control.py
/usr/local/share/pynq-venv/bin/python3 scripts/build_native_resize.py
```

3. 在已初始化XRT的板上交互终端运行以下命令（`/path/to/image.jpg`换成自己的原始JPEG，`out/demo_01`必须尚不存在）：

```bash
sudo -E /usr/local/share/pynq-venv/bin/python3 board/infer_images.py \
  /path/to/image.jpg --bit overlay/dma_top.bit --output out/demo_01 \
  --native-copy out/native/tensor_copy.so \
  --native-control out/native/tensor_control.so \
  --native-resize out/native/rgb_resize.so \
  --tx-range-sync --fuse-maps --fuse-decoder --fast-quantize
```

通过SSH自动执行时，使用仓库`scripts/board_ssh.py run --sudo`让登录shell初始化XRT。现有板卡上的相同冻结包已部署于`/home/xilinx/jupyter_notebooks/pld/final_mode6_fastq_20260929_v3/`，可直接在该目录使用，不必重复解包/编译。

成功标志为`INFERENCE_PASS`及退出码0，生成`frame_000001.bin`（65536字节）和同名JSON。C按`int8[1,1,256,256]`读取，默认`q>=-1`且位于`valid_rect`才为前景；保留帧号、模型身份和有效区。详细结果/错误/所有权约束见[C交接说明](../../docs/C_HANDOFF.md)和[PS集成](../../docs/design/PS_INTEGRATION.md)。如需先核对模型自带golden，可用包内`board/verify_cnn.py`；使用自己的JPEG不需要golden或GT。

## 已有板测和限制

同版本638张原JPEG、1000帧/1483.827秒全部与冻结INT8参考一致，短包、缺包、计算中复位恢复通过；dev638平均Dice 0.8421、IoU 0.7583。核心1.422 FPS，已解码RGB到head在线P95 1.266秒，未达到原定1秒目标。此处不包含JPEG解码、文件IO、C后处理或显示；dev638是开发集。C改变BD/时钟后必须重新检查时序与数值，不能沿用B最小BD的板测结论。
