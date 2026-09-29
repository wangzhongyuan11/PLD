# B 计算子系统：数值、存储与交付

此文件说明阶段3/4的数据合同。正式顶层为`axis_tensor_accel`，驱动为`board/tensor_accel.py`，整网调度为`board/pl_graph.py`；寄存器、命令、错误语义以[实现合同](ACCELERATOR_REGISTERS.md)为准。构建和板测状态见[验证记录](../history/STAGE34_STATUS.md)。阶段0/1/2的incr接口不变；用户于2026-09-27确认C接受当前接口及资源预算；显示工程尚未联调。

## 数值与模型换版

当前默认是 `role_a_final/soup_v5_a050`，由`reference/final_manifest.json`锚定124文件；spec SHA256 `fe137ca5b128fc1983fabad9bee39b3208cda4256e0d04c4e27cef5c07666aef`。最终模型的638图与1000帧板测已经完成，当前交付为ABI 1.2，详见[最终状态](../FINAL_MODEL_STATUS.md)。`reference/baseline_manifest.json`仅保留旧模型回归。再次换版需独立manifest和部署目录，重跑相关数值与硬件检查；不兼容的结构/容量明确拒绝，不截断张量。

外部DDR和AXIS张量均为NCHW、INT8[-127,127]；参数为OIHW INT8、INT32小端bias和INT32小端Q24乘数。现有15层名称/图结构作为ABI1的模型拓扑；通道、尺度、权重从规格加载，不把基线权重写死在RTL。

每层卷积结果在ReLU前作为调试/验收输出。层间映射用256字节LUT，按原始字节索引（负数使用其补码索引）。LUT生成顺序固定为：有符号INT8→ReLU（按算子需要）→float32乘源尺度→float32除目标尺度→ties-even→饱和。浮点只用于PS生成参数表，PL运行时查表。跳连和主分支使用各自的表，编码器分支在前。

## 计算与缓存结构

- `conv_mac_array`：8个输出通道×4个输入通道，每周期32项INT8乘积；分组累加INT32，报告中间溢出。偏置仅在一组点积的第一拍加载。
- `requant_q24`：四级可停顿流水，保留64位乘积，正负ties-even，饱和到[-127,127]；可串行处理8路累加结果。
- `conv_tile_engine`实现4个输入bank，每bank32768字节；输入通道按`c%4`分bank，bank内按`c//4`、行、列存储。顶层AXIS加载器把NCHW转换为bank地址，外部文件布局不变。
- 已实现8个输出bank，每bank8192字节；输出通道按`o%8`分bank。权重32个bank，每bank2048字节，支持最多64输出/96输入通道及3×3核；无效通道lane强制为0，不能读取未初始化的padding权重参与乘加。
- 层内复用权重，按输入/输出容量选择空间tile。PS只做数据聚集、halo拷贝、调度和比对；乘加、requant、ReLU/Pool/Upsample/Concat尺度映射均必须走PL算子，不能用NumPy结果代替板上运算。
- 输入tile带全图相邻像素的halo，全图外补0；内部tile边界不得误补0。tile在PL中可用local padding=0，完整小图算子测试另覆盖padding=1。

容量检查必须使用补齐后的bank占用：`ceil(Cin/4)*(tileH+K-1)*(tileW+K-1) <= 32768`，`ceil(Cout/8)*tileH*tileW <= 8192`。不能只看有效数据的总字节数。

基线675,282,944 MAC/帧；32 MAC/拍、100 MHz的理想计算下界约0.211秒，实际还含通道尾组、流水、传输、非卷积及PS调度。8 MAC/拍的理想下界已达0.844秒，因此采用32路，给搬运和显示预留时间。该估算不是实测帧率。

B子系统暂按不超过90 BRAM36、96 DSP、32,000 LUT设计，给C及集成至少保留50 BRAM36、124 DSP、21,200 LUT；用户已于2026-09-27确认C接受此预算。输入/输出/权重RAM初估约64 BRAM36，另计DMA、LUT表、FIFO及控制。以最终布线后利用率为准。

## 数据与命令生命周期

沿用simple AXI DMA和64 bit AXIS。核心AXI-Lite控制页为 `0x43C00000`，DMA为`0x40400000`。位宽、地址由HWH核对，实际时钟从PS读取，驱动拒绝不匹配的overlay。

| 操作 | 输入与输出 | 完成条件 |
|---|---|---|
| LOAD_PARAMS | MM2S发送该层weights+bias+Q24连续包；无S2MM结果 | 字节计数/TLAST正确，参数已写入RAM，PARAM_READY |
| LOAD_LUT | MM2S发送256或512字节映射表；无S2MM结果 | 表写入完成，LUT_READY |
| LOAD_INPUT | MM2S发送NCHW tile；无S2MM结果 | 张量长度正确，INPUT_READY |
| CONV | 使用已加载输入和参数，S2MM接收NCHW结果 | 运算完成且输出TLAST成功握手 |
| MAP / POOL / UPSAMPLE / CONCAT | 使用已加载输入与相应LUT，S2MM接收结果 | 运算与输出均完成，顺序/形状匹配 |
| RESET | 清除状态、有效位、在途数据；参数有效标记失效 | DMA已先停止/复位，核心回到IDLE |

控制页包含ID/ABI版本、START/RESET、BUSY/DONE/ERROR、错误码、COMMAND/MODE、输入/输出通道及H/W、kernel、收发字节数、frame/job及完成标识、命令/计算周期计数。卷积local padding固定0，halo由PS加载。准确偏移见[寄存器表](ACCELERATOR_REGISTERS.md)。

AXI-Lite实现必须分别接受AW与W，不假定二者同拍；处理WSTRB和响应背压。忙时禁止覆盖活动命令配置。完成标识与frame/job绑定，不能把上一包DONE当成本次成功。

AXIS以低地址字节对应低lane，TKEEP有效位从bit0连续；满拍0xFF，尾拍按实际字节数标记。TLAST属于当前命令的一整个包，不是每行结束。输出长度可能不同于输入，必须按输出长度重新生成TKEEP/TLAST。

`SimpleDMA`提供`send`、`receive`、独立`send_nbytes/receive_nbytes`及`submit/wait`，所有等待有界。原`transfer(...,receive_only=True)`仍为超时注入，不能与真正的`receive()`混用。错误时保存DMA和核心状态，先复位DMA再核心/重载overlay，确认停止后才能释放缓冲区；恢复后参数和LUT重新加载。

内部tile引擎、共享RAM和`CONCAT_UP`的双分支布局见[TILE_ENGINES.md](TILE_ENGINES.md)。`axi_lite_endpoint`实现AXI-Lite握手和响应保持；`axis_tensor_accel`实现寄存器后端、配置/长度校验、包加载和输出读回。

## DDR与C结果交付

PS持有连续输入/输出DMA staging buffer；整图特征、跳连、权重与LUT使用独立逻辑对象。输出完成及缓存同步前，C不得读/复用该帧。跳连1/3/5层的ReLU前参考输出与实际后续需要的数据生命周期分别记录，直到对应decoder消费完成才能复用。

最终结果为`[1,1,256,256]` signed INT8 logit，附带frame_id、模型/规格hash、head输出尺度、原图/letterbox尺寸、有效区域、状态与耗时；补边区域用于显示/面积时强制背景。C获取的输出需要所有权交接，不能异步引用已释放的DMA buffer。

最终模型采用概率阈值0.45、head尺度0.1995087721216397，整数掩膜条件为**有符号q≥−1且位于有效区域**。概率0.50/0.53/0.56/0.60对应q≥0/1/2/3。`PLGraph.run`从冻结模型合同输出概率阈值、整数阈值和模型身份，旧模型q≥0仅用于其独立0.50协议。ABI 1.2增加融合decoder模式6，既有寄存器偏移及模式0–5保持兼容；精确模式限制见[寄存器合同](ACCELERATOR_REGISTERS.md)。

验收必须从同一首层输入贯通所有算子；逐层golden仅用于比较，不得将每层golden input或PS计算的卷积/非卷积结果注入链路后宣称整网硬件完成。
