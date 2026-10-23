# ACD8 空间并行执行器 ABI 2.0

这是独立 ACDC 实验架构，不能和旧 ACD4、Skin/CAMUS overlay 或驱动混用。顶层为 `rtl/spatial_accel.sv`。`frame_control.sv` 管理 AXI-Lite，`frame_dispatch.sv` 从 DDR 读取执行表，`tile_job_queue.sv` 管理有界预取和缓存，`tile_executor.sv` 连接输入、参数、CAT、卷积和写回。

## 每帧数据流

1. 通过现有 `load_bundle` 验证 A 模型和哈希；实验模型显式标记 synthetic。`program.compile_program` 编译完整 15 Conv / 21 逻辑节点，不改变模型。
2. PS 分配并长期持有一块物理连续 CMA arena，按**实际物理地址**编译程序。默认 arena 容量16 MiB，P256实际使用6,005,760 B、P384使用6,004,736 B。不能把示例物理地址用于实际板卡。
3. `image.prepare_segments` 只生成输入、派生参数、CAT表和执行表；不初始化或注入中间特征和 head。输入从 NCHW 转为 C8HW8（3补8），参数按Co8分组，每组64字节bias/M头加W8x8权重。
4. flush CPU 改写的区域，配置程序地址、任务数量、epoch，然后一次 START。PS 不逐层注入输入，不逐tile提交MMIO。常驻模型参数和CAT表可只在换模型时写入；每帧更新输入和执行表epoch。
5. PL 自行取描述符，预取下个任务的输入/权重，与当前卷积及写回重叠。输入缓存和参数缓存各有两个slot；空间tile相同的Co组复用输入；已驻留的参数组可复用。
6. C02/C04/C06输出同时进入原尺寸特征写回和signed MaxPool；C09/C11/C13通过读取skip/deep、精确CAT LUT和全局nearest地址生成虚拟CAT，避免落盘整个CAT张量。
7. 上一层所有任务的卷积及S2MM状态确认完成，才允许下一层首次输入预取。完成整个执行表后置DONE。
8. PS invalidate物理head区域，`image.decode_head` 检查填充lane为0，恢复 `[1,4,256,256]` INT8。后续概率、原空间恢复、ED/ES、MPR、表面重建、体积/EF和导出沿用Axial应用链路；驱动和完整SoC构建入口已接入，时序和真实板测尚待验收。

CPU无需读取中间特征；不要每帧拷贝或同步整个arena。缓存同步、提交、等待、head转换必须计入真实PS-DMA-PL帧延迟。

## AXI-Lite 寄存器

32位，小端；地址按4字节对齐。AW/W可独立到达，B/R响应在背压下保持。未知、未对齐、只读寄存器写入，或运行时修改配置返回SLVERR。支持WSTRB。

| 地址 | 读/写 | 含义 |
|---|---|---|
| 0x00 | R | MAGIC `0x41434438`（ACD8） |
| 0x04 | R | ABI `0x00020000` |
| 0x08 | R | `[7:0]=P`，`[15:8]=8`输出分组，`[20:16]=0x1f`功能位 |
| 0x0C | W | 单次命令：1=START、2=空闲时清缓存、4=ACK_DONE；组合命令拒绝 |
| 0x10 | R | bit0忙/启动在途，bit1 DONE锁存，bit2终止故障，bit3启动在途 |
| 0x14 | RW | DDR执行表首地址，64字节对齐 |
| 0x18 | RW | 任务数，1..65535，单任务64字节 |
| 0x1C | RW | 本帧epoch，与所有描述符word15一致 |
| 0x20 | R | 16位故障码 |
| 0x24 | R | 故障任务tag |
| 0x28 | R | 自START到完成/故障的PL周期（不包含PS） |
| 0x2C | R | `[15:0]`已提交任务，`[31:16]`已确认完成任务 |
| 0x30 | R | 输入缓存命中数（累计，硬复位清零） |
| 0x34 | R | 参数缓存命中数（累计，硬复位清零） |
| 0x38 | R | 装载/运行重叠周期（累计，硬复位清零） |
| 0x3C | R | 构建时声明的时钟Hz，仍需与实际FCLK/时序核对 |

功能位16..20分别表示DDR描述符、双输入缓存、双参数缓存、Pool、虚拟CAT。IRQ为DONE锁存或FAULT；轮询仍须使用主机有界超时。

## 64字节任务

每行一个小端uint32。`program.Job`定义、校验并序列化；未定义位为0。

| word | 含义 |
|---|---|
| 0 / 1 | source0/source1的完整C8HW8张量首地址 |
| 2 | 当前Co8参数组首地址 |
| 3 | 当前CAT的512字节查表首地址；非CAT为0 |
| 4 / 5 | 当前Co8输出tile首像素地址 / 完整输出行stride（字节） |
| 6 / 7 | Pool输出tile首像素地址 / Pool完整行stride；禁用时0 |
| 8 | image_h[31:16]、image_w[15:0] |
| 9 | origin_y[31:16]、origin_x[15:0] |
| 10 | tile_h[31:16]、tile_w[15:0] |
| 11 | Ci[7:0]、有效Co[15:8]（1..8）、CAT split[23:16]、flags[31:24] |
| 12 | 从0递增的全帧任务tag，必须等于执行表序号 |
| 13 / 14 | source0/source1生产者逻辑节点ID |
| 15 | 本帧epoch |

flags bit0=3×3（否则1×1），bit1=ReLU，bit2=CAT，bit3=Pool，bit4=此前层的写回屏障。Pool的origin/尺寸必须为偶数。图像网格为2次幂且不超过256；输入、输出及参数地址按8字节对齐。输入halo由PL补零，不作为DDR中的扩展图像存储。精确INT8数值沿用生产规格。

输入缓存键包含epoch、生产者ID、两支地址、CAT表地址、完整网格、ROI、Ci、split及kernel/CAT类型；不含输出地址和Co组。参数键为epoch、参数地址、Ci/kernel。缓存valid仅在各自加载的**数据和状态均确认成功后**置位。READY/RUNNING/完成通知待消费的任务都保护所引用slot，缓存命中可以共享读取。

同一epoch中的源和参数内存不可被PS改写。新数据、新参数或地址复用必须更新epoch；32位epoch回绕前，在确认空闲后清除所有缓存。任何模型切换须重新核验哈希并生成派生参数；不得仅更换名称。

## 失败与验证边界

故障高字节：01描述符/缓存控制，02输入DMA，03参数DMA，04CAT表，05卷积，06写回，07程序读取/帧协议。低字节保留下层原因。首个终止故障和任务tag锁存，不会靠下一次START清除。

故障后，已经接受的DMA事务仍可能在途。禁止释放或改写相关CMA；必须确认vendor DataMover两方向及AXI master已停止/排空，再做共同复位或受控overlay恢复。ACK_DONE、CLEAR_CACHE或仅复位软件monitor均不构成安全恢复。完整SoC的halt/reset/恢复仍需实装和板测。

软停止按厂商 PG022 的 [MM2S Soft Shutdown](https://docs.amd.com/r/en-US/pg022_axi_datamover/MM2S-Soft-Shutdown) 与 [S2MM Soft Shutdown](https://docs.amd.com/r/en-US/pg022_axi_datamover/S2MM-Soft-Shutdown) 定义：HALT 后仍会完成已经提交的内存映射事务，期间数据无效；两个 `halt_cmplt` 保持到硬复位。GPIO同时拉高两方向HALT，PS等待两个完成位后才执行共同复位。此说明是设计依据，不能替代实际故障注入验证。

若有界HALT等待失败，真实Linux设备的关闭路径先记录unsafe并用SIGSTOP挂起拥有CMA的进程，保留XRT对象与lease；不允许CLI异常退出隐式释放仍可能被写入的内存。服务管理器通过PID/cgroup及退出状态拒绝切换，unit禁止超时SIGKILL。这是终止隔离状态，不是恢复成功；确认master停止或板卡复位前不得kill/继续该进程。

RTL回归中的`datamover_memory.sv`是带随机背压、延迟写状态、演化DDR及独立版本化数值oracle的行为模型；不等于vendor IP或真实DDR。空间缩小fixture保持15层和全部通道；`--divisor 1`另行覆盖原始256/128/64/32网格。综合后功能仿真必须继续对独立整数oracle逐写比较；OOC资源/时序和仿真周期均不能替代完整SoC布线及板测FPS。
