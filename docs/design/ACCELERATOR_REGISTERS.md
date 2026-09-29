# AXIS Tensor Accelerator：ABI 1.2 实现合同

对应`rtl/axis_tensor_accel.sv`与`board/tensor_accel.py`。ABI 1.0已有布线/板测见[验证记录](../history/STAGE34_STATUS.md)；1.1增加模式5输入映射卷积；1.2增加模式6双分支融合卷积。模式0–5和命令/寄存器偏移保持兼容。驱动接受1.0、1.1、1.2，并在旧硬件上拒绝对应的新模式。最终模型的数值、性能和板测范围见[最终模型状态](../FINAL_MODEL_STATUS.md)。B维护此控制与结果合同，C后续据此集成。

## 物理连接与所有权

IP为`user:pld:axis_tensor_accel:1.0`。当前正式候选`aclk`为125 MHz；`aresetn`低有效，内部控制采用同步复位。AXIS输入/输出均64位TDATA、8位TKEEP和TVALID/TREADY/TLAST；AXI-Lite为32位数据、12位地址。无中断，PS有界轮询。最小BD中控制页为`0x43C00000/4KiB`，DMA仍为`0x40400000`，MM2S经HP2读DDR，S2MM经HP0写DDR。历史100 MHz产物的性能与时序证据只适用于对应bit/hwh。

一次只有一个PS驱动拥有DMA、核心和staging buffer。加载、计算、读回互斥；忙时配置写入与重复START返回AXI SLVERR，当前工作继续，不能把被拒绝的写入当成已配置。DMA与核心均完成后，PS将结果复制到独立的NCHW INT8数组；C接收此数组的所有权，不得持有下一次DMA会覆盖的staging view。C若需要再次送入PL后处理，按结果长度另行DMA输入。

## 寄存器

偏移均相对于核心控制页，按4字节对齐。RW寄存器支持WSTRB逐字节更新；不支持的地址或对RO寄存器写入返回SLVERR。AXI AW与W可不同拍到达，读写响应在背压下保持稳定。

| 偏移 | 名称 | 类型 | 含义 |
|---|---|---|---|
| 0x00 | ID | RO | `0x504C4431` |
| 0x04 | ABI | RO | `0x00010002`；旧1.1/1.0产物分别为`0x00010001`/`0x00010000` |
| 0x08 | CONTROL | W1P | bit0 START，bit1 RESET；同时写入时RESET优先；读0 |
| 0x0C | STATUS | RO | bit0 BUSY，bit1 DONE，bit2 ERROR，bit3 PARAM_READY，bit4 INPUT_READY，bit5 LUT_READY，bit6 FAULT |
| 0x10 | ERROR_CODE | RO | 下表错误码；RESET清除 |
| 0x14 | COMMAND | RW | 1 LOAD_PARAMS、2 LOAD_INPUT、3 LOAD_LUT、4 RUN |
| 0x18 | MODE | RW | 0 CONV、1 MAP、2 POOL、3 UPSAMPLE、4 CONCAT_UP、5 MAPPED_CONV、6 FUSED_DECODER_CONV |
| 0x1C / 0x20 | H / W | RW | 输入空间尺寸；CONV/模式5/模式6含halo，CONCAT_UP指编码器分支尺寸 |
| 0x24 / 0x28 | CI / CO | RW | 输入/卷积输出通道；非卷积输出通道等于CI |
| 0x2C | KERNEL | RW | CONV为1或3；卷积stride1/local padding0 |
| 0x30 | SPLIT | RW | CONCAT_UP与模式6的编码器skip分支通道数；总输入通道为CI |
| 0x34 / 0x38 | FRAME / JOB | RW | 当前帧与命令编号；PS每条命令递增JOB |
| 0x3C / 0x40 | COMPLETED_FRAME / JOB | RO | 成功或失败工作对应的编号，防止使用上一次DONE |
| 0x44 | CYCLES | RO | START后整个命令周期数，含数据等待与背压；UINT32回绕 |
| 0x48 / 0x4C | EXPECTED_IN / OUT | RO | 硬件根据配置算出的命令收/发总字节数 |
| 0x50 / 0x54 | RECEIVED / SENT | RO | 已消费输入字节数 / 已完成输出AXIS握手的字节数 |
| 0x58 | COMPUTE_CYCLES | RO | 等待计算引擎的周期数，排除加载与结果串行读回 |

DONE为粘滞状态，下一次被接受的START或RESET清除。FAULT时BUSY=0，DONE=1，ERROR=1，三个READY均失效；仅RESET可恢复。RESET可在忙时接受，但PS必须先停止/复位DMA，防止DMA继续访问旧缓冲区。RESET保留配置寄存器，清计数、完成编号、流水和数据有效标记；RAM物理内容不清零。

PARAM_READY以CI/CO/KERNEL与已加载参数的标签匹配为准。INPUT_READY匹配MODE/H/W/CI/SPLIT。LUT_READY在CONCAT_UP与模式6时要求两个表均已加载。修改尺寸可以保留同层参数，但必须重新加载匹配尺寸的输入；尺度表对应哪个模型层由PS的冻结模型调度器负责。

模式5与CONV形状、参数包、输出格式一致。先加载参数和256字节LUT0，再LOAD_INPUT：每个原始补码字节在PL查表后写入输入RAM，随后RUN输出仍为requant后、ReLU前的原始卷积结果。未加载LUT就LOAD_INPUT返回错误4；更换LUT使已有模式5输入失效，必须重载输入，不能复用旧映射结果。模式5参数与模式0可按原标签复用，输入标签不能跨模式复用。PS图调度仅在8个纯MAP过渡使用该模式，并检查LUT[0]=0以保持halo补零语义；Pool和Concat仍独立在PL执行。

模式6用于最终模型第8、10、12层的decoder入口3×3卷积：skip分支先放完整分辨率halo patch，deep分支随后放半分辨率紧凑patch。两支分别由LUT0/LUT1完成ReLU与尺度转换；卷积引擎读取deep时按nearest×2取样，通道顺序始终skip在前。PS只拷贝两支原始INT8字节及补零halo，不在CPU计算插值、Concat或卷积。H/W必须为偶数且在`[4,258]`内，KERNEL固定3，`0<SPLIT<CI`且SPLIT与`CI-SPLIT`均为4的倍数；输入bank占用为`ceil(SPLIT/4)*H*W + ceil((CI-SPLIT)/4)*(H/2+1)*(W/2+1)`，不得超过32768字节。卷积输出bank仍不得超过8192字节。局部输出坐标`(y,x)`、卷积核坐标`(ky,kx)`读取deep紧凑patch的`(floor((y+ky+1)/2), floor((x+kx+1)/2))`；越界halo由PS置零，要求两张LUT的0号条目均为0。RUN返回requant后、ReLU前的原始输出。修改任一LUT使已映射的模式6输入失效，必须重载输入。

## 命令和包格式

每条加载命令只接收一个AXIS包。最低地址字节位于TDATA[7:0]；有效TKEEP必须从bit0连续，非末拍必须0xFF；只允许末拍部分有效。TLAST必须恰好落在硬件期望的最后一个有效字节，不能用行尾代替包尾。短包、长包、非法TKEEP均进入FAULT，已写入的部分数据不再有效。

| 命令 | DMA操作 | 精确包内容/长度 | 完成条件 |
|---|---|---|---|
| LOAD_PARAMS | MM2S单独发送 | OIHW INT8权重 + CO个小端INT32 bias + CO个小端INT32 Q24乘数；`CO*CI*K*K+8*CO`字节 | 最后参数写入RAM且长度/TLAST正确 |
| LOAD_INPUT | MM2S单独发送 | 普通模式：`[1,CI,H,W]` NCHW INT8；CONCAT_UP：skip在前、deep在后，长度`SPLIT*H*W+(CI-SPLIT)*(H/2)*(W/2)`；模式6：skip完整halo在前、deep紧凑halo在后，长度`SPLIT*H*W+(CI-SPLIT)*(H/2+1)*(W/2+1)` | 完整输入及内部bank转换完成 |
| LOAD_LUT | MM2S单独发送 | MAP/POOL/UPSAMPLE/模式5为256字节；CONCAT_UP/模式6为LUT0后接LUT1，共512字节 | 完整表写入 |
| RUN | S2MM单独接收 | CONV/模式5/模式6：`[1,CO,H-K+1,W-K+1]`；MAP同输入形状；POOL空间减半；UPSAMPLE空间加倍；CONCAT_UP：`[1,CI,H,W]` | 计算完成、输出串行打包完成，且最后一个TLAST真正被下游接收 |

权重、输入和LUT输出值不能为INT8的-128；bias可使用整个INT32范围；Q24必须为非负INT32。内部累加溢出明确失败。当前A冻结模型另有更严格的静态绝对值界，不依赖“实际样例没有溢出”判断安全。

例如首层3→8、3×3的参数包为216字节权重、32字节bias、32字节乘数，共280字节。32×256输出tile的输入为3×34×258=26316字节，输入末拍TKEEP=0x0F；输出65536字节，输出末拍TKEEP=0xFF。先发送参数，再发送输入，再RUN接收输出；输入TLAST不会被直接转发为输出TLAST。

所有后续算子也按此机制运行：PS把上一层PL输出分块送回，PL执行ReLU/尺度LUT、Pool，以及独立或融合的nearest Upsample/Concat。PS不执行卷积、池化、插值或尺度转换来替代硬件，只生成小型尺度参数表、拷贝halo与拼接传输包、收集结果。

## 错误与恢复

| 错误码 | 含义 |
|---|---|
| 1 | 命令、模式、尺寸、通道、kernel、split或bank容量无效 |
| 2 | TLAST位置与预期总字节数不符 |
| 3 | TKEEP为零/非连续，或非末拍不满 |
| 4 | 输入、参数或LUT尚未加载/标签不匹配 |
| 5 | 卷积INT32累加溢出 |
| 6 | 计算引擎或内部RAM返回状态异常 |
| 7 | 参数/输入/LUT数值范围不支持 |

输入一直不来、下游一直不接收时核心会保持BUSY，PS必须在期限内报超时并保存核心与DMA状态。驱动默认每次DMA/核心等待上限2秒，顺序为：DMA复位并确认两通道halted → 核心RESET → 确认状态清除 → DMA启动 → 重新加载参数/LUT/输入。若DMA复位失败，尝试重载同版overlay；确认DDR访问停止前不得free/reuse buffer。`SimpleDMA.transfer(...,receive_only=True)`仍只用于旧通路超时注入，真正单向接收使用`receive()`。

## 模型、帧与C交付

PS加载模型前检查冻结manifest、规格、参数和golden哈希；上板只接受配套bit/hwh、ID/ABI与HWH地址/位宽，并读取实际FCLK0。参数不固化在bitstream，可按同一图结构和容量合同显式换版，但必须生成新manifest并重新做数值/板测验收。

输出为65536个signed INT8 logit，NCHW `[1,1,256,256]`，不是类别标签或RGB图。交给C时同时提供FRAME、模型/规格hash、head输出尺度、原图与letterbox元数据、有效区域及成功状态。最终A模型概率阈值0.45对应`q>=-1`；补边强制背景，面积分母使用有效区域。旧模型阈值仍由其独立模型身份决定。没有有效区域元数据时不得自行假定整幅图都有效。C的后处理/HDMI/UI在B交付之后独立集成，不以本合同替代C侧联调。
