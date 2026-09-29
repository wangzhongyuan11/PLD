# 阶段 0/1/2 验收记录

主机日期：2026-09-26。结论：**本地验证 → 已有 Vivado 构建产物复核 → 部署 → PS/DDR/DMA/PL → 精确比较 → 故障恢复**已闭环。阶段编号与目标保持 [PLAN.md](../PLAN.md) 原样；完整命令见 [board/README.md](../../board/README.md)。

本轮基于 `fa6de30`，复用 `307da19` 已验证的 RTL/TKEEP 修复、PS7 配置、BD、IP 打包与 bitstream；没有改 RTL、PS7 参数或重新综合整个工程。新增有界 PS DMA 驱动、连续包/错误恢复验证和构建产物门禁。最终代码由本记录所在提交固定，板测按下表 SHA256 精确对应文件版本。

## 逐项验收

| 阶段 | 检查与证据 | 结论 |
|---|---|---|
| 0 工具链 | 2026.1 安装、Zynq 器件和综合/实现许可实测见 [环境报告](../validation/VIVADO_ENVIRONMENT.md)；本轮再次运行 xvlog→xelab→xsim，reg_slice 272 beats、incr 5349 beats/116 包，`ALL_SIMS_PASS`，退出 0 | 通过 |
| 0 构建 | 本轮 `check_build.py` 对 fifo/incr 均 `BUILD_CHECK_PASS`：bit/hwh 与 XSA 内容逐字节一致；核对实现日志末次 bitstream DRC 为 0 Errors。打包 IP 中两份 RTL 与当前源码 SHA256 相等，component.xml 含两侧 TKEEP | 通过，复用已有构建 |
| 0 时序门禁 | fifo：WNS +2.205、WHS +0.024、WPWS +3.750 ns；incr：+2.226/+0.020/+3.750 ns；TNS/THS/TPWS 与失败端点均 0。负向构造 WNS=-0.001 的报告、与XSA不符的bit均被门禁拒绝；无效构建模式退出2 | 通过 |
| 1 预编译通路 | tutorial：7 种尺寸×4轮 + 1000随机长度包 + 超时恢复后7包，`BOARD_PASS`，退出0 | 1035包正确 |
| 1 自建通路 | 2026.1 fifo：相同测试，`BOARD_PASS`，退出0；板端 bit/hwh 哈希与本机相等 | 1035包正确 |
| 2 自研 IP | 2026.1 incr：相同测试，全部输出 `(input+1) mod 256`，`BOARD_PASS`，退出0 | 1035包正确 |
| 2 PS 故障 | 三套 overlay 分别只开启 S2MM、不发送数据，2秒期限触发 `transfer timeout`；记录 CR/SR/地址/长度，复位 DMA、重载 PL 后7种尺寸全过 | 三套 `RECOVERY_PASS` |
| 2 失败返回 | incr `--inject-timeout`：先正常7包、注入超时、恢复后7包，再按负例设计退出1；JSON `recovery_passed=true`、`cleanup_recovered=true` | 通过 |
| 2 数值诊断 | 用 fifo 期望检查 incr bit：首差异 `index=0 expected=134 actual=135`，退出1、保存寄存器，清理恢复成功 | 通过 |
| 2 故障后再运行 | 独立新进程加载 incr，7/7尺寸与尾部保护区均正确，`BOARD_PASS`、退出0 | 通过 |
| 2 主机回归 | `unittest discover -s sim -p 'test_*.py'`：10项全过；覆盖旧 idle 误判、错误位优先、短包、非法地址/长度、start/reset期限、先复位或重配置再释放缓冲区 | 通过 |

四种输入为固定种子随机、全零、全255（覆盖溢出回绕）、递增字节。随机包长度均匀取1～65536字节，seed=20260926；标准尺寸1/7/8/64/1000/65536/1048576字节。每包比较全部有效字节、实际接收长度和64字节尾部保护区；不是抽样CRC比较。另重跑A交接核验，15/15小块acc/requant/参数切片通过，尚未扩展为整网数值验收。

## 性能与资源

| Overlay | 1 MiB吞吐中位数（4轮） | 范围 MB/s | LUT / FF / BRAM36 / DSP |
|---|---:|---:|---|
| tutorial | 434.36 MB/s | 431.34～450.23 | 本轮不重新综合第三方bit |
| fifo | 425.75 MB/s | 424.62～439.16 | 2303 / 3267 / 4.5 / 0 |
| incr | 434.21 MB/s | 430.76～437.04 | 2304 / 3288 / 3 / 0 |

计时含源/目的缓存flush、MMIO编程、完成轮询和invalidate；不含缓冲区分配、数据生成/比较、overlay加载。MB使用10^6字节，有效负载只计算一次。这是占位通路吞吐，不是CNN推理或端到端显示性能。

## 产物与原始证据

板端：`/home/xilinx/jupyter_notebooks/pld/stage012_20260926/`；本机原始JSON/日志：`out/stage012/`。文件名 `tutorial`、`fifo`、`incr`、`negative_timeout`、`negative_mismatch`、`final_smoke`，各有 `.json/.log`。主机与板端哈希已交叉核对，生成物不入Git。

| 文件 | SHA256 |
|---|---|
| tutorial bit | `c8ef4e84495597ec251aae53e88e43184118e760d430aebbf65dafbced78ceb4` |
| tutorial hwh | `78046fe4af0917bf8b6c6bed105f741c5b7af594e5b30e8b99b857ac46d1d476` |
| fifo bit | `e91b5f4ef9e42e4f4629d6ddc8432cd6fb6cbdcc18e830bb7c4049abe8e1ebee` |
| fifo hwh | `191d4906a0110de3ef822f945436d89d6699f7f86af87afb927eb669a46a7924` |
| fifo xsa | `8d9c3101bae82ba9c209c43d03f46f666a019978f85bce52c4ef7715df78211c` |
| incr bit | `ec9fa8a64059aada066e567e060c236a0272e495d90b1fa679ef7dbe1ceec70f` |
| incr hwh | `3f23ba2b0728b7fd9e1710a566acca89f04ef6bac113439910b4bfdab9ed13ed` |
| incr xsa | `d4f9dd5fbc8285082d520a1662c9ed22c5fa9324621d22a9233604ab40756f9f` |
| simple_dma.py（三轮与最终版本相同） | `f429a1bae9c0c6cd26a9b61409ec32e7906cee0927111ecc0aac270a284937ec` |
| verify_overlay.py（三套1035包测试版） | `cd167c915412677ec850e3f261999be6d0a94fcece3cbb0874ddd67ac8949b10` |
| verify_overlay.py（最终部署版） | `718d40666ad9fedecc44868201abc2bf29ee8a40477345b0c7c9baba71bc0f88` |

最终脚本在三轮板测后只补充了顶层清理异常的JSON记录和退出2分支；DMA传输/正常包/恢复算法未变。随后在此最终版本执行两项退出1负例和独立7包正常测试，保留两个版本的哈希，不将历史测试混记为同一文件。

## 保留的限制

- 本轮完成阶段0/1/2，不代表TinyUNet、分割精度、HDMI或医学辅助系统已经完成。正式B/C合同、量化算子、层间尺度、整网调度属于阶段3起的任务。
- PS7负DDR skew继承原始教程；现有DDR传输测试通过，不消除温度/电压/长期压力下的板级风险。综合告警保留在环境报告。
- 已实测缺失发送端导致的超时与数据不匹配恢复；DMA错误位、启动/复位卡住、短包在主机模型中验证，没有通过非法DDR地址制造真实总线错误。物理设备无法复位或重配置时不声称已恢复。
- 每套1000随机包是通路稳定性验证，不是阶段5要求的整机连续10分钟/1000帧验收，也未证明整机内存长期稳定或C显示工程集成。
