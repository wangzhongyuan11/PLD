# 皮肤镜 ROI 与 HDMI 触控

当前 ROI 工作台仅支持皮肤镜的最终 INT8 模型，阈值固定 0.45。CAMUS 与 ACDC 保持普通分割页面。

## 使用

1. 在 8088 统一入口选择皮肤镜，等待就绪，点击 ROI / 交互分割。
2. 上传 JPG / PNG，长边自动限制为 1024 像素。点击运行取得初始结果。
3. 选择前景点、背景点、ROI 或目标框，在原图单击或拖动，再点击运行。重画 ROI 清除旧点和目标框；重新选图开始新工作。
4. 点击显示到 HDMI。触屏顶部 FG / BG / ROI / Box 对应四种工具，左侧原图接受点选和拖框，Run 计算，右侧显示修正后的叠加，Web / Stop 退出显示。

网页与 HDMI 共用图片、提示和结果；同时操作发生版本冲突时同步后重试。提示改变后旧结果隐藏，必须运行才产生新结果。页面只保留选图、四种工具、运行和 HDMI 开关，撤销、恢复、清除、回放与下载入口已移除。

## 处理与显示

ROI 裁剪后送现有 FPGA 模型推理；点和框使用已有概率在 PS/CPU 上图割修正，模型没有经过提示式训练。目标框限制框外不属于病灶，不会整框填满。ROI 外输出背景；面积按处理图像素计算，有 ROI 时以 ROI 为分母。

点和框更新复用概率，无需切走 HDMI。ROI 或图片改变时，原后端先停止并回收显示子进程，再执行 FPGA 推理；完成后按显示开关自动恢复交互 HDMI。沿用 Console.start_hdmi / stop_hdmi 和原 Skin8 硬件释放流程，未改动模型、位流或推理核心。普通任务历史和 ZIP 保留原始结果。

## 源码与部署

- interactive.html / interactive.js：精简网页与 Pointer Events 点选、拖框。
- interactive_workspace.py：网页和 HDMI 的共享提示、版本、显示控制。
- interactive_bridge.py：调用既有推理与图割服务；旧回放 API 仅保留兼容，不在页面开放。
- interactive_hdmi.py：640x480 触控画面与坐标映射。
- ../touch_drag.py：从已有 ACDC 拖动读取器复用 evdev 手势；不更改 ACDC 实现。

增量部署使用独立目录和 rollback.json / backup，正常停止入口和 hub 后更新网页/显示适配文件。原模型、数据、位流和其他模型服务保留。HTTP 检查或注入输入事件不等于用户实屏与手指操作验收。

2026-10-24 已部署到 `/home/xilinx/skin_roi_touch_20261024`。板端完成初始分割、HDMI 显示、ROI 改变后重新推理及自动恢复、点/框缓存修正、关闭显示后重新推理与基线一致检查；用户确认实屏显示、手指拖框和点选正常。记录见该目录 `verification.json`、`alignment.json`。需要回退时在板端运行 `sudo /usr/local/share/pynq-venv/bin/python3 /home/xilinx/skin_roi_touch_20261024/rollback_display.py`；脚本先核对文件版本并正常释放服务，再恢复备份。
