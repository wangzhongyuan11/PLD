# PLD 三模型分割演示小程序

正式发布的微信小程序，通过**微信云开发**中转，调用 PYNQ-Z2 板卡上现有的三个模型服务：

团队仓库中的工程入口是 `wechat_miniprogram/`，包含前端、电脑桥接、测试与部署说明。
微信开发者工具应打开此目录；下文相对路径也以此目录为根。本机原开发副本位于 `D:\PLD小程序`。

| 模型 | 板卡服务 | 输入 | 结果 |
|---|---|---|---|
| 皮肤镜分割 | hub 8765 → display_web | 皮损照片（JPG/PNG） | 六视图 PNG、面积占比等指标、结果包 |
| CAMUS 心超分割 | hub 8765 → camus | 超声图像或 NPZ/NPY/NIfTI/MHD | 叠加/分割/标签图、各腔室面积、结果包 |
| ACDC 心脏 MRI | entry 8088 + acdc 8092 | 成对 ED/ES .nii.gz 体数据 | 六张 MPR 视图、统计/心功能表、结果包 |

## 架构与独立性

```
手机小程序（miniprogram/）
   │ wx.cloud（云存储 / 云数据库，免域名免备案）
   ▼
微信云开发环境 pld-d7gzvyxp10bdc9437
   ▲ 出站轮询（api.weixin.qq.com，AppID+AppSecret）
   │
桥接程序（bridge/，跑在电脑上，纯 HTTP 客户端）
   │ 8088 /entry/status、/entry/switch（三模型切换，手动触发）
   │ 8765 /hub/login→代际URL（皮肤镜、CAMUS 任务）
   │ 8092 /api/v2/jobs（ACDC ED/ES 裸流上传）
   ▼
板卡现有服务（沿用现有 HTTP、推理与硬件调度）
```

关键约束（有意设计，勿破坏）：

- 桥接不开监听端口、不直接访问 FPGA/DMA。ACDC 主程序只增加可选的任务级显示开关，省略开关时保留原网页行为。
- 皮肤镜/CAMUS 提交 `auto_display=false`；ACDC 提交 `X-Auto-Display: false`。后者需要同时安装更新的 `acdc/appliance/web.py` 和 `engine.py`，健康状态会声明 `task_auto_display=true`；旧板端仍可运行，但可能按全局设置自动显示，桥接会记录提示。
- 桥接不修改全局 HDMI 设置。推理仍会按现有流程关闭 HDMI、恢复推理硬件；模型切换会同步影响网页端和显示器。
- 当前模型的任务按创建时间领取；板卡未就绪或忙碌时继续等待。模型不符时等待用户在首页手动切换，双模型降级时走已有 `/hub/switch`。
- 结果图片回传后即可预览，ZIP 完成后再显示下载入口。云端结果上传在后台执行，板端推理保持串行，最多缓冲两套结果。结果回传失败可选择“重传结果”，复用仍存在的板端结果；“重新运行”则重新推理。
- 三模型入口（8088）未部署时自动降级为双模型（皮肤镜/CAMUS），ACDC 任务明确报错。

## 手机预览、ROI 与保存（2026-10-30）

- 桥接在皮肤镜/CAMUS `preview_ready`、ACDC `quick_preview_ready` 时先回传最长边 768 像素的 JPEG 预览，不等待完整 PNG/ZIP；活动任务的只读状态查询间隔最多 300ms，空闲查询不变。旧服务没有预览能力时继续等待完整结果；预览失败不会中断计算。ACDC 快速预览仅为 ED 中心切片，不代表体数据已处理完成。
- 皮肤镜选图后可开启“ROI 裁剪分割”，拖动框选。坐标相对于手机显示的 EXIF 方向修正图；电脑裁剪后调用原分割接口，面积占比针对 ROI。此功能不连接 `/interactive/*`，不修改网页/HDMI 共享交互工作区，也不增加点提示或图割计算。普通全图 JPEG 直传路径保持不变。
- 保存先解码检查图片，再按真实 PNG/JPEG 格式创建临时文件，处理小程序和 iOS 系统相册权限，返回设置后继续保存；失败显示原始错误，临时副本会清理。基础库与本机现有开发配置统一为 3.17.5。
- 查询、确定字段赋值、固定路径同内容文件上传可在断连后重连一次；条件领取/新增不盲目重放。领取响应丢失按唯一 claim 标记核实；板端接受后的任务编号立即保存，单桥接安装重启时可接回原任务。板端拒绝或未返回编号的提交不能伪称已恢复。
- 本次仅修改本目录，不需要安装网页、HDMI、模型、位流或计算服务。前端需要在微信开发者工具重新编译/预览；已运行的 Python 桥接进程需要在空闲时重启才能加载新代码。

本地验证：`python -m unittest discover -s tests -v`（在 `bridge/`），以及 `node tools/test_api_js.cjs`、`node tools/test_regressions.cjs`、`node tools/test_mobile_features.cjs`。这些检查不替代 iOS 真机保存、真实手机首图耗时及 PL 推理验收。

## 目录

```
miniprogram/            小程序前端（6 页面 + 隐私组件，直接使用）
bridge/                 电脑常驻桥接（Python 标准库 + Pillow）
  bridge.py             主循环：心跳 / 切换命令 / 任务
  board_client.py       板卡三服务 HTTP 客户端（Origin/Cookie/代际 URL）
  wechat_api.py         微信云开发服务端 API（db / 存储 / token）
  preprocess.py         皮肤镜输入合同（≤300万像素 JPG）
  tools/                演示数据：NIfTI 写入器 + 云端预置脚本
  tests/                单测与真实服务格式回归（mock 板卡 + 本地 HTTP）
docs/DEPLOY.md          部署与联调（先读这个）
docs/RELEASE.md         提审与发布
```

## 测试

```bash
cd bridge
D:/conda/python.exe -m unittest discover -s tests -p "test_*.py"
# 本机若有 nibabel（如 D:\PLD\acdc\.venv）可再交叉验证 NIfTI：
D:/PLD/acdc/.venv/Scripts/python.exe -m unittest tests.test_nifti
```

前端检查：在根目录运行 `python tools/check_miniprogram.py`、`node tools/test_api_js.cjs`
和 `node tools/test_regressions.cjs`。新回归覆盖真实 CAMUS 面积字典、ACDC CSV、同名 ED/ES、
队列过滤、命令失败与心跳过期、部分结果预览/重传、后台上传和连接复用。
历史联调证据见 docs/DEPLOY.md 第 8～9 节；这些证据不替代本次更新后的真机验证。

## 部署

见 [docs/DEPLOY.md](docs/DEPLOY.md)；提审发布见 [docs/RELEASE.md](docs/RELEASE.md)。
