# PL tile 计算引擎与内存布局

本文对应 `rtl/conv_tile_engine.sv`、`rtl/tensor_tile_engine.sv`。二者共享特征RAM，由`axis_tensor_accel`互斥调度；外部PS接口见[寄存器合同](ACCELERATOR_REGISTERS.md)，验证状态见[记录](../history/STAGE34_STATUS.md)。下面是PL内部地址，PS通过标准NCHW/OIHW数据包加载，不直接操作bank地址。

## 卷积数据流

`NCHW 输入写入 → 4 个输入 bank 同步读 → 32 个权重 bank 同步读 → 8×4 MAC → 8 个 INT32 累加结果 → 单路 Q24 流水 → 8 个输出 bank → NCHW 读回`。

输入 tile 已包含 halo；PL 固定 stride=1、local padding=0。对于全图的内部 tile，halo 是相邻 tile 的实际像素；只在全图边界外补零。3×3 的输出 H/W 比输入各小 2，1×1 则相同。PS 只聚集这些字节，卷积和量化均在 PL 执行。

| 存储 | 容量 | bank | bank 内地址 |
|---|---|---|---|
| 输入 INT8 | 4×32768 字节 | `c % 4` | `(c // 4)*Hin*Win + y*Win + x` |
| 权重 INT8 | 32×2048 字节 | `(o % 8)*4 + i % 4` | `(o // 8)*ceil(Cin/4)*K*K + (i // 4)*K*K + ky*K + kx` |
| bias / Q24 | 各 64×INT32 | 按输出通道索引 | `o`；由 `param_multiplier` 选择 bias 或 multiplier |
| 输出 INT8 | 8×8192 字节 | `o % 8` | `(o // 8)*Hout*Wout + y*Wout + x` |

内部写口是bank/address/data；顶层包加载器把外部NCHW/OIHW字节顺序转换为这张表，保持A的交付布局。

每个输出通道组包含最多 8 个通道。对一个空间坐标，地址发生器依次遍历 kx、ky、4 通道输入组：一拍读 4 个激活和 32 个权重，交给 MAC。首拍装载 bias，末拍输出累加值。未满 4/8 通道的 lane 强制为零，即使对应 RAM 未初始化也不能污染结果。Q24 依次处理有效输出 lane，将结果写回对应输出 bank；输出尚未经过 ReLU。

例如第一层一个 32×256 输出 tile：输入为 `[1,3,34,258]`，每个有效输入 bank 用 8772 字节，输出 bank 各用满 8192 字节。每个空间点需要 9 拍 MAC 输入，8 个通道结果经 Q24 写回；阶段3/4状态机曾测得26拍/空间点，该历史值不代表后续流水优化版。当前ABI 1.2使用紧凑deep halo融合decoder入口，容量/地址规则以[模式6合同](ACCELERATOR_REGISTERS.md)为准，完整调度175命令/帧。计算周期不能直接换算成在线分割FPS。

`start`在idle时锁存配置；后续CFG_PLANE/CFG_SIZE/CFG_CHECK流水计算并检查容量，避免配置乘法串联形成长路径。`busy`期间配置变化不影响当前计算。`done`是单拍脉冲，正常完成时所有输出已写入RAM。错误码1表示维度/容量无效，2表示MAC累加溢出，3表示内部状态无效。非法维度在运行前拒绝；顶层READY及标签保证bias、权重、输入完整且配置匹配。

复位清控制和流水有效位，不清 RAM 内容。复位后残留 RAM 不等于数据有效；顶层必须清 PARAM/INPUT/LUT_READY，重新加载。正常算完后可以只换输入复用本层参数。`busy` 时加载写口被忽略；顶层应向 PS 明确拒绝这种命令，不能把“忽略”当作成功。

## 非卷积数据流

`tensor_tile_engine`使用卷积引擎的`aux_input_*`与`aux_output_*`访问同一组特征RAM，不增加另一套大缓存。两个引擎不能同时运行；顶层互斥调度加载、计算与读回。阶段5已改为连续读流水：地址递增不在内循环做乘法，MAP/UPSAMPLE/CONCAT流水填满后每拍输出一个像素，POOL每个输出发四次读请求；BRAM选择/池化比较与LUT查表之间有一级寄存。完整命令另含配置、流水填充排空、装载和读回。优化后的完整BD时序与板测见`../history/STAGE5_STATUS.md`。

| cfg_op | 运算 | 输入与输出 |
|---|---|---|
| 0 | MAP | 每个原始 INT8 字节查 LUT0，形状不变；LUT 可以包含 ReLU 和尺度转换 |
| 1 | POOL | 有符号 MaxPool2×2 后查 LUT0；H/W 各减半，要求输入 H/W 偶数 |
| 2 | UPSAMPLE | nearest×2 后查 LUT0；H/W 各加倍 |
| 3 | CONCAT_UP | 编码器分支查 LUT0在前；深层分支 nearest×2、查 LUT1在后；输出通道为两支之和 |

LUT 共 512 字节：`lut_addr[8]` 选择分支，低 8 位是 INT8 的原始补码。不能把负数当作正数比较池化，也不能先经过任意 LUT 再取 max。RTL 先做 signed max，再查表；测试中额外使用非单调表来区分这两种顺序。

CONCAT_UP 的配置 H/W 是编码器分支的空间尺寸，`cfg_split` 是它的通道数，`cfg_channels` 是拼接后的总通道数。深层分支尺寸为 H/2×W/2。两支分别加载，bank 分布如下：

- 编码器按普通 NCHW 输入 bank 排布，从地址 0 开始。
- 深层分支使用自己的局部通道号分 bank，从每个 bank 的 `ceil(split/4)*H*W` 开始。
- 深层支路地址为上述 base 加 `(local_c//4)*(H/2)*(W/2) + (y//2)*(W/2) + x//2`。
- 输出由 PL 按总通道号依次写入标准输出 bank，因此拼接顺序、上采样和两支尺度转换均由 PL 完成。PS 只分别装载原始分支字节。

输入容量检查必须计入分支通道补齐；不能用两支总字节数除以 4 估算。输出可到 96 通道，受 `ceil(C/8)*Hout*Wout <= 8192` 约束。LUT、输入配置和容量错误必须在顶层启动前完整校验。

## RAM 推断与验证入口

共享端口的地址、使能和写数据必须先组合仲裁，再放入单一 RAM 读/写语句。将两个互斥读写分支直接写在 RAM process 中曾导致 Vivado 改用大量 LUT RAM；现已修正，并在 `vivado/check_tile.tcl` 加入卷积引擎必须推断为 64 BRAM36 等效容量的门禁。

```powershell
& D:\Git\bin\bash.exe scripts/run_sim.sh tile       # 小块、边界、复位、容量拒绝
& D:\Git\bin\bash.exe scripts/run_sim.sh tile-full  # 15层完整尺寸，共80个tile
& D:\Git\bin\bash.exe scripts/run_sim.sh tensor     # 14处层间转换及独立算子边界
& D:\Git\bin\bash.exe scripts/run_sim.sh control    # AXI-Lite AW/W解耦、WSTRB、背压、复位
& D:\Git\bin\bash.exe scripts/run_sim.sh bytes      # AXIS字节打包/解包、TKEEP、背压与复位
& D:\vivado\2026.1\Vivado\bin\vivado.bat -mode batch -notrace -nojournal -source vivado/check_tile.tcl -log out/tile_synth.log
```

完整单层测试分别装载 A 的该层输入，只能证明算子/地址映射正确。板上整网必须从同一首层输入连续运行所有算子，golden 只用于比较，不得逐层注入代替真实网络。上述`check_tile.tcl`仅检查独立引擎OOC；完整BD的AXIS、AXI-Lite、DMA、时钟复位和布线另由`build.sh cnn`验收，已有结果见`../history/STAGE34_STATUS.md`及`../history/STAGE5_STATUS.md`。
