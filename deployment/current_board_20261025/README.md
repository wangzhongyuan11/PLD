# 当前板端源码对齐（2026-10-25）

本目录记录 10 月 25 日运行中的应用、评测工具和有效 systemd 配置。源码以仓库常规开发目录为主，通过 [SOURCE_MAP.json](SOURCE_MAP.json) 对应板端绝对路径，避免再复制一套网页与应用代码。10 月 24 日目录和 Release 保留为历史快照；`delivery/current.json` 仍指向原正式冻结资产，不在本轮改写。

**2026-10-26 仓库更新说明：** 常规源码和三份服务 drop-in 已有后续修改，新增共享 `pld-hdmi.service` 也作为源码配置候选保存。上表旧 `board_sha256` 继续表示当时采集值，不改写为新部署证明；在新工作树中运行旧清单的本地校验出现差异是预期结果。最新仓库内容使用[独立源码清单](../repository_20261026/README.md)，本轮未通过 SSH 重新采集板端文件。

## 当前实际入口与源码

| 板端组件 | 当前运行位置 | 仓库入口 |
|---|---|---|
| 8088 三模型入口 | `/home/xilinx/latency_opt_20261024/entry` | `display_web/entry/` |
| 双模型 hub、ROI、评测页面 | `/home/xilinx/jupyter_notebooks/pld/releases/pld_dual_20261017/hub` | `display_web/unified/` |
| 皮肤镜网页 | 上述 release 的 `web/` | `display_web/` |
| Skin8 双模型适配 | `/home/xilinx/skin8_candidate_20261023` | `experimental/skin8/tools/` |
| 实际 CAMUS 网页和显示 | `/home/xilinx/jupyter_notebooks/PLD_Dev/camus` | `camus/` |
| ACDC V2 应用 | `/home/xilinx/acdc_board_v2/releases/latency_opt_20261024` | `acdc/appliance/` |
| CPU/FPGA 评测 | hub 的 `evaluation_tools/` | `perf_eval/` 与下述冻结依赖映射 |

ACDC 应用身份为 `80c3b92f0fce9c3ad3da191d5f4c417852fdc17175bbee90e8ea87ae0d635c9f`，运行时清单身份为 `1f0bdf8739980ea8e8e2a06ee349b57ed81d041d87515e4e9400cd30b7ae7747`。systemd 原 unit 和全部有效 drop-in 位于 `systemd/`，按原绝对路径对应保存；多层覆盖都保留，不能仅用旧 base unit 推断当前入口，也不要把快照直接覆盖安装到其他板卡。

230 项映射包括 212 个应用/工具源码条目和 18 个 unit/drop-in。多个运行目录的公共模块指向同一仓库源码；`board_sha256` 核对板端原始字节，`lf_sha256` 仅规范化 CRLF/LF，不忽略逻辑差异。

同日 HDMI 后分割优化已更新到源码表：皮肤镜网页复用完整 PNG，Skin8 会话构建复用本次已校验参数文件，CAMUS 接回已有原生图像处理并使用快速无损四宫格编码。补齐两份 C 源文件；位流、模型及推理驱动未修改。单次同图 HDMI 后对照为皮肤镜 6.756→4.314 秒；CAMUS 较大图片最终 HDMI 后任务为 6.230 秒，仍包含硬件恢复，不能称为完整 3 秒达标。验证边界见 `ALIGNMENT.json` 的 `latency_followup`；本次板端备份为 `/home/xilinx/latency_followup_20261025/rollback.json`。

## 最新行为与验收边界

- 三模型通过正常暂停/恢复交接硬件，保留软件进程；皮肤镜与 CAMUS 同位流热切换。新的完整输入按设置自动分割、结果成功后自动 HDMI 显示；手动停止取消自动显示。
- ROI 只针对皮肤镜，固定当前 INT8 模型和 0.45 阈值。网页“清除所有标记”与 HDMI 右下角 Clear all 清掉 ROI、目标框和全部前景/背景点，保留图片；点 Run 更新分割。
- 用户已确认该优化候选的三模型画面、触控与 ROI 操作正常。清除按钮的本地定向检查、网页资源和板端源码身份核对已通过；本次同步没有重新运行推理、切模型或重复物理验收。
- 随机静止闪烁未现场复现；按用户要求保留原 HDMI 驱动/时序，没有将其标为已修复。
- 热切换代表记录约 2.4～7.3 秒；首次新进程仍约 27～33 秒。大图 ROI 初始到首帧曾达 27.8 秒，不能宣称所有路径秒级。详见 [优化测量与限制](../../docs/validation/MODEL_HDMI_OPTIMIZATION_20261025.json)。

## 必须保留的源码差异

1. 板端 hub 的 CPU/FPGA 对比入口、评测占用检查和退出等待已同步到 `display_web/unified/`，评测器的父 hub 协调与进度输出已同步到 `perf_eval/`。没有删除原板端功能。
2. 评测子目录中的 CAMUS `web.py`、`hdmi.py` 是早期冻结副本，与当前普通 CAMUS 服务不同，精确源码保存在 `evaluation_frozen/camus/`。其他评测依赖复用源码表中相同的仓库文件。该评测配置使用原 release 的 bit/runtime，不能把它的成绩称为当前 Skin8 服务成绩。
3. Skin8 冻结 `frame_driver.py`、`verify_board.py` 以及 Linux ACDC `surface_hdmi.py` 与 10 月 24 日快照逐字节规范化后相同，因此映射到已有快照，不覆盖本地其他实验驱动。常规 ACDC 源码省略未使用的 Linux-only `fcntl` 导入，以支持 Windows 测试；Linux 实装版本仍精确可取。
4. `installed_tools/install_interactive.py` 是板上留存的旧安装工具，不是正在运行的服务。最新维护版在 `display_web/unified/install_interactive.py`，执行前仍需核对目标安装的版本与路径。

## 核对与后续更新

在仓库根目录运行只读核对，不需要模型、PYNQ 或板卡：

```text
python scripts/verify_current_board_sources.py
```

在板上用同一脚本和源码表执行 `--board --manifest <SOURCE_MAP.json绝对路径>`，按原始字节核对当前文件；不启动服务、不配置 FPGA、不运行推理。任何后续应用变更应刷新对应映射和应用身份，不能删掉哈希检查。修改源码后出现不一致是预期提醒，不能把旧源码表当作新部署证明。

网关包由 `scripts/package_dual_model_web.py` 生成，包含评测页面/适配器与 `evaluation_config.example.json`。评测工具包由 `scripts/package_perf_eval.py --output <新暂存目录>` 生成；若要还原本表的既有评测闭包，再以 `evaluation_frozen/camus/` 的两个文件覆盖包内 `pld_source/camus/`，并重新生成该包的 SHA256 清单。示例配置仅包含模型、工具与数据目录路径，需要目标环境已有模型/原生库/位流；不自动安装依赖或启动评测。

本轮源码整理不改变运行中的应用。清除按钮增量的回退入口为 `/home/xilinx/display_stability_20261025/rollback.py`；完整性能候选回退入口为 `/home/xilinx/latency_opt_20261024/rollback.py`，均通过正常服务退出与身份检查恢复。上传数据、访问码、SSH 凭据、模型、位流、原生二进制和系统依赖不随本源码提交分发。
