# 交互工作台 v5（2026-10-17）：前后对照、完整历史、单次上传与可校验回放

本轮只改软件（网页/接口/图割缓存），**未改动 RTL、模型、位流、冻结交付与板端部署**，
也**没有**连接板卡或做真实模型推理；4 FPS 多字节/bank 并行属于后续硬件候选，本版不宣称。

## 实现

| 能力 | 位置 | 说明 |
|---|---|---|
| 前后对照 | `service.py`、`interactive.js/html` | 服务端返回 `baseline`（固定后处理、图割前）+ 交互后 `outputs`；页面并排显示叠加图、目标像素差与两段标签哈希。提示不会修改基线；参数改变后旧基线明确失效 |
| 完整状态历史 | `interactive.js` | 每次改动前保存完整快照（roi/box/points/target_class/阈值开关与值）；目标类别改变清空提示；undo 恢复全部；「恢复初始分割」保留 ROI/类别/阈值、只清框与点并重新生成真实无提示结果 |
| 单次上传 token | `interactive_bridge.py` `POST /interactive/images` | 校验 model/generation、PNG/JPEG、像素与字节上限；返回不可猜测 token 与服务端像素 SHA；改点/框/阈值只提交 token + 参数；旧 base64 请求仍兼容 |
| 诚实执行信息 | `execution`、`timings_ms` | `pl_executed_this_request`、`probability_cache_hit`、`baseline_cache_hit`；阶段互不重叠：校验/预测/判定/基线合成/基线 PNG/图割上下文/图割全流程/最终合成/最终 PNG/原图 PNG |
| 可校验回放 | `replay`、`POST /interactive/replay` | 导出含 schema、模型/spec/代次、规范可提交参数（`backend=int8`）、处理图 PNG、原图尺寸与变换、基线/最终标签哈希与 outputs；导入时服务端解码 PNG、校验尺寸与 **RGB 像素哈希**，浏览器有 WebCrypto 时再复核一次，没有则以上传回执哈希为准；失败保留原工作，导入后再点运行才推理 |
| 基线缓存 | `SegmentationService._baseline` | 键含图像像素哈希、ROI、模型、target、阈值、variant/backend、代次/spec/source_job 与概率来源；点/框变化复用，图像/ROI/类别/阈值/模型改变失效；返回副本，避免显示输出污染缓存；只保留一份 |
| 图割上下文复用 | `refinement.CutContext` | 每个 crop 的颜色特征、网格边与 pairwise 权重只构建一次并设为只读；unary、硬约束与 CSR 建图仍每次重建，`maximum_flow` 仍真实求解；输入不匹配拒绝并重建；清除/切换模型释放 |

## 测试（本机，无板卡）

- Python：仓库根目录运行 `python -m unittest discover -s display_web/unified/tests -p 'test_*.py' -v`，
  **67 项全部通过**，包含合成 CAMUS 输出头的实际概率恢复、一次解码/一次预测、多次提示、ROI 重新推理、
  上传与切换串行、回放篡改拒绝、基线独立性、安装与回滚、冻结 v4 的全部 9 项升级哈希。
- Node：`test_interactive.cjs`（坐标映射与计时文本）、`test_ui.cjs`（主页状态机）、
  `test_interactive_state.cjs`（完整撤销/恢复、忙时保护、token 请求、失败保留状态、无 WebCrypto 回放及再导出）全部 PASS。
- 独立比较 main `29aaaefcc661767fe13706716aa104bee9de950e` 的原始图割实现：
  3 种尺寸、3 种合成图像、4 种提示/阈值配置，共 36 组；27 组标签与 flow 完全一致，9 组冲突提示拒绝一致。
  主机合成用例的总耗时中位数：原实现 19.29 ms、新版冷路径 20.52 ms、复用准备后 16.90 ms。
  环境 Python 3.12.14 / NumPy 2.3.5 / SciPy 1.18.1 / Pillow 12.3.0；各次仍真实求解最大流，**不代表板端性能或医学精度**。
- 浏览器以合成概率完成登录、上传、基线显示、背景点更新、旧结果失效与 JSON 导出。
  将实际导出的 133686 字节 JSON 经认证 HTTP 重新校验并提交，基线/最终标签哈希均与记录一致，
  三次请求仅生成一次合成概率；前端导入路径另由 Node 驱动实际事件处理器验证。
- v5 软件补丁通过 CRC、载荷哈希、重复打包一致性及升级来源检查；本轮未部署。

完整本地验收记录保存在 `pld-medical/runs/interactive_optimization/20261017/owner/`，不随协作源码提交。

## 未完成 / 边界

- 全部为**未板测**软件改动；真实点/框交互、分阶段板端耗时、叠加与 HDMI 仍待独立验收。
- 前端不做像素哈希硬依赖（板端 `http://192.168.2.99` 属非安全上下文，通常无 WebCrypto）。
- 回放按"处理图 PNG + 参数"重建；若旧格式缺 `image_png`，明确拒绝并提示。
- 硬件吞吐后续候选（不在本 PR 宣称）：单帧多 bank/多字节并行、PL 侧后处理、PS 侧零拷贝路径、
  批量点提示的增量 unary 计算。
