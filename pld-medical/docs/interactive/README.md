# 皮肤病灶 / CAMUS 独立交互接口

2026-10-16。Codex 主执行设计、实现和验收；DeepSeek 执行材料清点、固定回归与请求示例整理。本次增加软件交互接口，没有进行训练、校准或重新导出模型。

## 启动与打开

两项服务独立启动，各自维护模型、推理缓存和请求队列。可同时运行：

```powershell
& 'E:\嵌赛\pld-medical\host\interactive\start_skin.cmd'
```

```powershell
& 'E:\嵌赛\pld-medical\host\interactive\start_camus.cmd'
```

启动后分别打开：

|服务|网页|预测接口|
|---|---|---|
|皮肤病灶分割|http://127.0.0.1:8766/skin/|POST http://127.0.0.1:8766/api/skin/predict|
|CAMUS 心脏超声|http://127.0.0.1:8767/camus/|POST http://127.0.0.1:8767/api/camus/predict|

服务只绑定本机地址。端口已占用时先使用现有实例；不要重复启动。关闭对应窗口或按 Ctrl+C 可停止该服务。也可以在项目根目录运行 `.venv\Scripts\python.exe -B -m software.interactive_segmentation.server --model skin --port 8766`，CAMUS 使用 `--model camus --port 8767`。

直接使用现有项目 `.venv`，需要 numpy、SciPy、PyTorch、Pillow、nibabel，无新增第三方依赖。此扩展依赖项目中已验收的模型和原数值推理模块；单独复制网页文件不能执行分割。

## 网页操作

1. 选择图像，或载入已有开发样例。默认使用 INT8 软件模拟，可切换 FP32。CAMUS 另外提供原交付 / 隐藏层直方图候选选择。
2. 在原图单击添加前景点或背景点；选择“目标框”后拖出包住目标的矩形。前景点表示当前所选结构；背景点表示不属于该结构。
3. ROI 裁剪在模型推理之前执行，区域外输出背景。更新 ROI 会清除旧提示，结果始终映射回完整原图。
4. 调整阈值后自动重新处理。皮肤默认 0.45；CAMUS 默认保留多类 argmax，勾选独立阈值后才改变所选结构的概率门槛。
5. 可撤销、清除提示，下载目标二值掩膜、全部类别标签以及包含参数/模型身份的结果 JSON。“精确定位”可直接输入原图像素坐标。

## 真实实现与边界

原有 U-Net 不接收点或框作为网络输入。本次使用冻结模型的概率图、原图外观和用户种子，增加 `seeded_min_cut` 图割：框外限制为非目标，前景/背景点作为硬约束，轮廓根据图像和概率计算。矩形本身不会直接填充为分割结果。算法基于 [SciPy maximum_flow](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.maximum_flow.html)。

图割网格最长边上限为 256，之后映射回原分辨率并落实点击像素约束。外观信息以 0.2 的固定权重辅助判断，避免少量局部颜色样本推翻大片稳定的模型预测；硬种子仍然优先。距离过近的相反种子如果落在同一网格，会明确报错；可以扩大提示间距或使用更小的 ROI。精细边缘仍受原模型和网格分辨率影响。增加提示并不保证精度提高，本次验收证明接口和算法实际运行，没有新增交互精度或临床结论。

CAMUS 保留原固定 `island5_hole1` 后处理；有交互时先执行固定后处理，再进行提示修正，使最后的点击约束不会被后处理覆盖。除所选结构新增/移除的位置外，其余类别保留原判断；移除位置按非目标类别的最高概率归类。

页面显示像素数、面积占比和本次软件处理耗时，没有参考标注时不显示 Dice。ROI 面积占比以 ROI 像素数为分母。INT8 指已有整数模型的软件模拟，不代表本次完成 FPGA 上板交互或板端时延验收。既有 `PLD_Web` 板侧界面与原交付包保持不变。

## 接口文档与证据

- [皮肤专用接口](skin_api.md) · [请求示例 JSON](skin_api_examples.json)
- [CAMUS 专用接口](camus_api.md) · [请求示例 JSON](camus_api_examples.json)
- 通用参数和输出约定：[接口约定](protocol.md)
- 源码：`software/interactive_segmentation/`；网页和启动入口：`host/interactive/`。
- 单元 / HTTP 回归：`runs/interactive_interfaces/20261016/deepseek_tests04/TEST_RESULT.json`，16 项全部通过。
- 真实模型验收：`runs/interactive_interfaces/20261016/owner/real_smoke03/RESULT.json`，8 次真实前向；CAMUS 原交付和候选默认 INT8 输出各与已验收保存预测逐像素一致。
- 网页验证及最终交付汇总：同任务目录的 `owner/INTERFACE_ACCEPTANCE.json`。

开发样例只使用现有干净 ISIC2017 validation 图像和 CAMUS 开发患者图像，不读取标注或封存测试。本任务原始问题和记录均保留：首轮图割背景样本不足、首轮 CAMUS packed 描述符读取失败、图片下载未落盘，以及颜色提示权重过强。修正后分别通过新的回归、真实冒烟、实际下载和视觉核验记录。
