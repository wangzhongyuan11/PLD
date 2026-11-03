# 灰度模块仿真起点

当前未配置 FPGA 工具；本测试台需要 SystemVerilog 支持。若安装了 Icarus Verilog，可在项目根目录运行：

```powershell
iverilog -g2012 -o runs/gray_sim hardware/rtl/rgb_to_gray.v hardware/sim/tb_rgb_to_gray.sv
vvp runs/gray_sim
```

先创建 `runs/`（运行软件示例也会创建）。通过时输出 `PASS rgb_to_gray basic vectors`。
测试覆盖黑白、三原色、无效周期与同步低有效复位；不是完整系统验证。
后续增加随机 RGB、连续帧、背压、窗口边界和软件参考向量比对。
板卡确定后再添加器件工程、时钟/引脚约束和板级测试。
