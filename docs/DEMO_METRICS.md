# CPU 软件基线与 FPGA 演示指标

网页入口“CPU / FPGA 指标对比”打开可关闭的对比表，保留原模型页面。HDMI 触摸顶部任务状态栏打开对比叠层，再触摸返回原图或放大画面；ACDC 的顶部 ED/ES 等按钮保留，触摸按钮下方的状态栏打开指标。两端读取同一份离线评测快照，不在演示时自动启动耗时的 CPU 参考。

HDMI 入口用于普通结果查看；ROI 交互时可从网页顶栏查阅配对报告，不把全图网络评测当作 ROI 或概率复用路径的响应时间。

包含处理帧率、单帧或单切片中位数/P95 延迟、完整循环吞吐率、有效 GMAC/s、功耗、LUT/FF/BRAM36/DSP、DDR 相关 DMA 流量估算、Dice/IoU 和精度百分点变化。原有 HUD 继续显示当前保存任务的计时；对比表明确是已记录结果，不能冒充实时遥测。

## 导入真实报告

现有 `perf_eval` 支持的皮肤镜/CAMUS 配对报告可直接导入。先按 [评测工具说明](../perf_eval/README.md)生成 CPU 和 FPGA 的 benchmark.json 及其 NPY 结果。ACDC 展示使用切片/s，但旧评测生产者不支持 ACDC；需另有符合相同完整报告合同的 ACDC 配对导出并验收，不能把皮肤镜/CAMUS 报告改名替代。

建立 JSON 配置，路径相对配置文件目录。例如：

```json
{
  "schema": "pld-demo-metrics-input-v1",
  "records": [{"cpu": "skin_cpu/benchmark.json", "fpga": "skin_pl/benchmark.json"}]
}
```

```text
python scripts/prepare_demo_metrics.py --config work/metrics_input.json --output work/demo_metrics.json
```

导入器复用原有输出比较，检查完整报告、模型、输入集、有效区域和原始 NPY 文件哈希。自动耗时比仍遵守原比较器的同板、同输入、重复输出一致与整数逐元素一致要求。跨平台结果可并列展示，不能直接标为 FPGA 对优化 CPU 的加速。CPU NumPy INT8 数值参考须保留其非优化库说明。

在入口及显示子进程环境设置 `PLD_DEMO_METRICS` 为生成文件的绝对路径。配置不随源码包上传，实际启动目录与共享模块映射须在部署时核实。每个模型/运行时身份只能有一条记录；重新测量请输出新文件并切换配置。导入脚本拒绝覆盖已存在文件。

网页可浏览报告里各版本，明确展示模型/运行时 SHA。HDMI 只选模型 SHA 和运行时 SHA 都匹配当前保存任务的记录；没有匹配、资料损坏或缺少运行时身份则显示未测量，不挪用其他版本成绩。CAMUS 的新结果补充 runtime_sha256（配套 bit 文件的 SHA），沿用原 schema 的兼容新增字段；旧结果没有该身份时不会自动匹配。生成身份在硬件分配前读取，原有完整运行包与 ABI 检查继续执行；它不改变推理或计时口径。

## 附加证据配置

下面可选字段加入每条 records 对象。缺少证据时，对应项保持未测量。SHA 均为真实文件完整 SHA256，不填写示例数值。

| 配置字段 | 必要内容 |
|---|---|
| utilization | path 指向布线后 Vivado utilization.rpt；sha256 为该文件哈希；model_sha256 和 runtime_sha256 精确匹配 FPGA 报告。显示 LUT、实际 FF、BRAM36 等效数和 DSP 的 used 值 |
| power.fpga，kind=chip_estimate | path 指向 Vivado power.rpt，并提供 sha256、model_sha256、runtime_sha256；显示芯片工具估计 W，不称整板实测 |
| power.cpu 或 power.fpga，kind=board_samples | path 指向原 power-samples JSON，提供 sha256 和 benchmark_sha256；采样帧数必须与绑定报告相同；显示仪器整板输入平均 W。实际采样区间仍须人工核对 |
| workload | path 指向含 model_sha256、macs_per_frame、method 的运算量记录，并提供 sha256；用该运算量除以报告累计网络墙钟得到有效 GMAC/s，不称理论峰值 |
| accuracy | cpu/fpga 指向原 labeled_accuracy 报告；cpu_benchmark_sha256、fpga_benchmark_sha256 明确声明模型关联；要求同 GT 集、类别、split 和标签坐标范围；显示逐图平均 Dice/IoU 与候选减基线的百分点变化 |

DDR 字段从完整逐帧 DMA 包字节数除以累计网络墙钟计算（十进制 MB/s）；缺包记录则不填。它排除 PS 额外拷贝、缓存回写和 HDMI，不是 DDR 控制器总带宽或峰值。软件 CPU 的 FPGA 资源为不适用，CPU DDR 总带宽没有测量时仍是未测量。

处理帧率是完成数除以累计处理时间；完整循环吞吐率另外包含工具采样、哈希和结果保存开销。样本数、基线、主机与计时范围一并保留。单次测试的分位数不能代表稳定长跑分布。功耗两列若边界不同，不计算能效比。输出完全一致只说明数值一致，不自动得到 Dice=1 或精度变化为零。

本轮没有可绑定当前板端的配对实测报告，没有填入论文或附件中的示例值；功能、示例配置和合成界面检查不构成新的实板性能成绩。板卡仍不可用，后续沿已有独立目录、回退与实板验证授权继续。
