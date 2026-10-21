# ACDC Board Standalone V2（实施中）

目标和完整验收范围见[独立板端计划](../docs/BOARD_STANDALONE_V2_PLAN_20261021.md)。V1 保留在提交 `69e18ec`，V2 工作位于独立任务分支。当前还没有完整 V2 Web/网格/导出交付，不能把本目录的数值验证时间称为上传到显示时间。

## 已实现

- 原生 C 的归一化、Torch 等价双线性缩放、INT8 RNE 量化、SciPy 等价三维线性重采样。
- 三切片 LRU；经过参考环境生成的 Softmax 查表；单期概率体与分块原空间恢复，不提前 argmax。
- 轴对齐坐标缓存、一次检查后复用的采样器，避免分块反复扫描完整概率体。
- `NumericalPipeline` 提供 ED/ES 数值链路、单硬件执行者、输入预取/输出恢复重叠、有界 future、取消和失败状态。当前输出状态为 `numerics_completed`，明确不等于最终完整任务完成。
- 新增板端MPR采样、窗宽窗位、标签叠加/轮廓与PNG生成模块，已通过开发机像素/方向/几何测试；Web接入与ARM交互耗时仍待验证。
- 全20例/40体/1616片开发机输入/标签对照；板端ARM首期46片核对。证据见[原生模块记录](../docs/evidence/appliance_native_20261021.json)。这里使用保存的head验证后处理，新增PL推理次数为0。
- 已在独立候选完成真实P6两期92片：输入/head逐片与独立参考一致，原空间标签/affine/统计一致，数值链路19.483 s，单片P95 119.363 ms。见[两期PL记录](../docs/evidence/appliance_numerical_PL_20261021.json)；不包含初始化/上传/网格/导出/浏览器，尚非40秒整体验收。

板上候选目录与旧部署分离，PYNQ/系统NumPy未升级。候选使用隔离的 Nibabel 5.2.1（需要 NumPy>=1.20），板端已有 NumPy1.21.5。PC参考继续使用原固定环境；实际实现版本和原生库SHA另行记录，不伪造原A绑定。

## 已有验证入口

在仓库根目录、原固定PC环境中执行；路径需替换为实际工件位置：

```text
python -m pytest acdc/tests/test_appliance_native.py acdc/tests/test_appliance_pipeline.py -q
python -m acdc.appliance.qualify --model <冻结模型> --model-sha <spec_SHA> --results <已有40体结果根> --output <新报告.json>
python -m acdc.appliance.assets --model <冻结模型> --model-sha <spec_SHA> --output <新派生常量目录>
```

`qualify`核对MRI/原结果manifest及检查点身份，比较实际原生工作体、全部整数输入与原空间标签。它不执行PL，不能用于板端FPS或完整任务速度验收。

`benchmark_replay`在板端读取已校验的单期fixture：原始MRI、冻结exp表、原始工作体/输入/标签SHA及保存的head。fixture必须由PC参考准备；它不连接远程推理或用CPU代替PL。

```text
python -m acdc.appliance.benchmark_replay --fixture <fixture目录> --output <新ARM报告.json>
```

编译器默认 `gcc`，可设 `ACDC_IMAGE_CC`；首次构建在系统临时目录生成按源码、编译器、flags和机器身份绑定的共享库，以后核验并复用。禁用fast-math/FMA收缩；ARM使用Cortex-A9/NEON hard-float选项。编译缓存准备时间单列，不纳入已就绪数值阶段。

精简VTK的隔离构建入口为 `native/mesh/build_vtk.sh`，需要VTK9.6.0官方源码、CMake和C++工具链；不会安装系统Python VTK或修改全局PYNQ环境。编译完成、网格程序和质量/耗时核验均仍待完成，不能将正在构建记为成功。

## 下一阶段

真实P6串行/流水配对对照；完整40体ARM/PL验证；精简VTK表面和平滑；板端MPR、LAN Web、完整导出、开机readiness和第二台电脑访问。最终按同一浏览器从上传开始到两期结果、平滑展示、ZIP和首帧全部可用验收 `<40 s`。
