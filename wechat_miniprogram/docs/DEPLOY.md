# 部署与联调（DEPLOY）

按顺序执行。桥接沿用板卡现有 HTTP 服务；停止桥接会停止领取新任务，已有板端推理仍按原服务执行。

## 2026-10-25 修复版兼容说明

- 小程序和桥接需一起更新。云集合名称和原 `pending/running/done/error/waiting_model` 状态保持兼容；`running` 任务允许提前带有部分 `result.views`，新页面会立即显示这些图片。
- `result_workers` 默认 2（范围 1～3）；`async_results` 默认 true。调试时设为 false 可等待本次云回传完成后再处理下一项。后台上传最多缓冲两套结果，板端仍只串行运行一个推理任务。
- 更新的 ACDC `web.py`/`engine.py` 支持任务级 `X-Auto-Display: false`，健康接口声明 `task_auto_display=true`。浏览器不传该头时，行为与更新前相同。仅更新桥接而继续使用旧 ACDC 服务时，任务仍可运行，但其自动显示服从原全局设置，日志会提示；不能称为已禁止自动显示。
- 推理会按板端已有流程释放 HDMI、恢复推理硬件。桥接不修改全局 `auto_hdmi`；首页模型切换会影响网页端和显示器。
- 结果回传失败时“重传结果”复用仍存在的板端结果和已上传文件，不重新推理；板端结果已删除或服务重启导致结果不可用时，可选择“重新运行”。
- ED/ES 即使同名也使用独立云路径，总体积限制是 32MiB；旧任务若指向同一个 fileID，需重新上传。皮肤镜合规 JPEG 保留原始字节。
- 首页跟踪具体切换命令，失败后显示原因并允许重试；心跳超过 60 秒未更新会显示离线。板端短暂忙碌保持排队，提交响应中断且无法确认是否已接收时不会自动重复提交。
- 本次源码修复不等于已重启常驻桥接、发布小程序或升级板端。下方第 8～9 节保留历史测试条件，不能继承为新版本验收。

本地同负载对照（6 张 PNG + report + ZIP，模拟每次上传 40ms、数据库写入 10ms，各 3 次）：
旧串行逻辑的首批预览/完整结果中位均为 0.344s；新版分别为 0.051s/0.173s，结果文件字节一致。
这只衡量结果回传的调度开销，不是微信云网络或板端推理测速。日志会记录首批预览、板端结果读取、
云端回传和桥接总时间，方便实际升级后按同一输入复测。

## 0. 前提

- 微信开发者工具已安装并登录小程序（AppID `wxadaa7f8991e1721a`）。
- 云开发环境已开通：`pld-d7gzvyxp10bdc9437`。
- 电脑与板卡保持现有网线直连（板卡 `192.168.2.99`）。
- 三模型功能需要板卡已部署三模型入口（`pld-entry`，端口 8088）。
  未部署时小程序自动降级为双模型（皮肤镜/CAMUS），ACDC 不可用。

### 版本矩阵（2026-10-24 与 origin/main 逐文件核对）

远端 `origin/main@47c38f3` 的 `deployment/current_board_20261024/` 是**板端实际部署快照**
（三模型入口 8088、ACDC v2 8092、皮肤镜/CAMUS 经 skin8 组合位流运行时+hub 8765）。
下表是此前对照记录。当前板端源码路由以 `D:\PLD\deployment\current_board_20261025` 为准；本次 ACDC 任务级显示开关是其后的兼容性增量：

| 合同 | 部署快照 | 本地最新（codex/three-model-entry） |
|---|---|---|
| entry 8088 `/entry/status` `/entry/switch`（Origin 检查） | 一致 | 超集（新增 stage/settings 字段，桥接已透传 stage） |
| hub 8765 登录/status/switch/代际代理 | 一致 | 超集（新增 interactive_busy，桥接不依赖） |
| 皮肤镜 8766（经 hub，Bearer+JSON base64） | 一致（web_entry 复用原 server.Console，token 同源） | 一致 |
| CAMUS 8090（经 hub，files[] 提交） | 一致（web_entry 补丁为 status 增加 active 键并换 PL 内核） | 一致 |
| ACDC 8092 `/api/v2/jobs` 裸流+X-ED-Bytes+assets | 一致（仅缺本地新增的 lifecycle 本机限制，桥接不调） | 超集 |

Skin8 运行时保留原 HTTP 入口。CAMUS 面积字段应按字典读取，ACDC 心功能 CSV 应按实际表头读取，不能用早期演示 mock 的数组/两列 CSV 代替真实合同。

## 1. 云开发控制台一次性配置

mp.weixin.qq.com → 云开发 → 环境 `pld-d7gzvyxp10bdc9437`：

1. **数据库 → 创建集合**（共 4 个）：`tasks`、`commands`、`board`、`demos`。
   （2026-10-25 实测：创建集合的 API 未开放、隐式建表也不行，只能在控制台点。）
2. **权限设置**——只需改 2 个，其余保持创建时的默认（仅创建者可读写）：
   | 集合 | 权限 |
   |---|---|
   | `tasks` | 默认（仅创建者可读写）——用户只能看到自己的任务 |
   | `commands` | 默认（仅创建者可读写）——切换指令只有发起人需要读 |
   | `board` | **所有用户可读，仅管理端可写**（首页板卡状态） |
   | `demos` | **所有用户可读，仅管理端可写**（ACDC 演示病例） |
3. **存储**：保持默认（所有用户可读，仅创建者可写）即可；桥接以服务端身份读写不受限。

## 2. 桥接程序配置（电脑上）

1. `bridge/config.example.json` 复制为 `bridge/config.json`（已被 .gitignore 排除）。
   本机若已代为生成（`hub_token` 已从板卡读取填好），只需补 `app_secret`。
2. 填两项：
   - `app_secret`：mp 后台 开发管理 → 开发设置 → 生成小程序密钥（AppSecret）。
   - `hub_token`：皮肤镜访问码（板端 hub 的 `config.json` → `token_file`
     文件内容，或网页启动链接 `#` 后的一串）。
   - 其余保持默认即可。
3. 安装依赖：桥接仅用 Python 标准库 + Pillow（`D:\conda` 已具备）。
   皮肤镜任务要求 Pillow（>300 万像素自动缩小、PNG→JPG）。

> 线上适配说明（2026-10-25 实测）：线上板卡的 hub 可能仅绑定 127.0.0.1
> （8765 对外不通）。桥接已内置双模式访问：直连 8765 失败时自动改走
> 8088 入口代理（同代数同路径），无需任何配置。当日线上入口位于
> `/home/xilinx/latency_opt_20261024`（比部署快照更新），合同兼容。

## 3. 预置演示数据（ACDC 内置病例）

```bash
cd bridge
D:/conda/python.exe tools/upload_demo_data.py --config config.json
# 输出 DEMO_SEED_DONE 即成功
```

## 4. 启动桥接

```bash
cd bridge
D:/conda/python.exe bridge.py --config config.json
# 或双击 start_bridge.cmd
```

日志：控制台 + `bridge/logs/bridge.log`。看到"桥接已启动"即可。

**开机自启（可选）**：管理员 PowerShell 运行一次

```powershell
schtasks /create /tn PLDBridge /tr "D:\conda\python.exe D:\PLD小程序\bridge\bridge.py --config D:\PLD小程序\bridge\config.json" /sc onlogon /rl limited /f
```

注意：桥接所在电脑关机 = 小程序不可用（板卡本身不受任何影响）。

## 5. 小程序端

1. 微信开发者工具打开 `D:\PLD小程序`，编译预览（环境 ID 已写入 `app.js`）。
2. 真机预览：首页应直接显示板卡"在线 + 当前模型"（无访问码，直接使用）。
   （若显示离线：检查桥接是否在跑、config 的 board_host、网线。）

## 6. 联调清单（按序）

| # | 步骤 | 通过标准 |
|---|---|---|
| 1 | 桥接启动、打开小程序首页 | 板卡状态在线、当前模型正确 |
| 2 | 皮肤镜：相册上传一张皮损 JPG | 任务完成、六视图可看可存相册、指标显示面积占比 |
| 3 | 网页端同时访问（可选） | 板端忙时小程序继续等待；推理/HDMI 按原互斥流程交接，切换会影响双方当前模型 |
| 4 | 首页切换到 CAMUS，上传超声图/NPZ | 切换进度可见、任务完成、四图 + 面积 |
| 5 | 切换到 ACDC，运行内置演示病例 | 任务完成、六张 MPR、EDV/ESV/SV/EF 指标 |
| 6 | 任务页"转发"结果包 ZIP | 文件可发到微信聊天 |
| 7 | 拔网线再插（可选） | 小程序显示离线再恢复；任务不丢 |

已知提示（2026-10-24 更正）：hub 切换要求后端 status 含 `active` 键
（display_web/unified/hub.py:104），裸跑旧 `pld-camus.service` 时会报
"后台缺少任务状态"。**当前板端部署已解决**——skin8 运行时的
`web_entry.py` 给 CAMUS `/api/status` 打了补丁补上 `active`。仅当板端
单独启动未打补丁的 camus web 时才会复现；此时把日志发回来定位。

## 7. 回退

- 停桥接进程（或删除计划任务 `schtasks /delete /tn PLDBridge /f`）。
- 小程序端不动；云开发集合数据可保留或清空，均不影响板卡。
- 若另外安装了 ACDC 任务级显示开关，按该次安装的源码/服务备份回退；本轮未提供自动安装或改写部署快照。

## 8. 本机已完成的验证（2026-10-24，未连接板卡时）

| 验证项 | 方式 | 结果 |
|---|---|---|
| 桥接逻辑 | 30 个单测（mock 板卡三服务 + mock 微信 API + 内存云），连跑 3 轮 | 全绿 |
| 演示数据 NIfTI | 生成 4 个 ED/ES 文件，逐个用 nibabel 5.4.2 加载校验 | 形状/2mm 间距/值域/ED>ES 全过 |
| 小程序静态门禁 | `python tools/check_miniprogram.py`（页面/组件引用、WXML 标签闭合、事件处理器存在、WXSS/JSON） | PASS |
| 小程序 JS 逻辑 | `node tools/test_api_js.cjs`（状态映射/时间格式化/键唯一性等纯函数，真实执行） | PASS |
| 真实微信 API 错误路径 | 占位密钥运行 `bridge.py --once/--常驻 10s`，命中真实 api.weixin.qq.com 返回 `invalid appsecret`+rid | 网络/TLS/JSON/错误解析全链路可用；密钥错误 30s 退避防配额消耗 |
| 桥接 CLI 冒烟 | 占位密钥 + 死端口板卡：启动、心跳/命令/任务三循环优雅降级、日志落盘 | 退出码 0、循环不崩溃 |
| `D:\PLD` 零改动 | git status 与会话前快照逐行 diff | 无本工程产生的任何改动 |

## 9. 真板三模型闭环实测（2026-10-25，经桥接同款代码路径）

| 验证 | 输入 | 结果 |
|---|---|---|
| 皮肤镜任务 | 真实 ISIC_0001769.jpg（合同化 433KB） | PASS：FPGA 推理→六视图+report+7.1MB zip，面积占比 3.60% |
| 切换 skin→camus | /entry/switch | PASS（switching→ready） |
| CAMUS 任务 | 真实 patient0001_2CH_ED.nii.gz | PASS：四图+report+zip |
| 切换 camus→acdc | /entry/switch | PASS |
| ACDC 任务 | 合成演示病例 1（ED/ES） | PASS：ED/ES 各 16 切片推理→32/32→MPR+CSV+3.3MB zip |
| 切换 acdc→skin 恢复 | /entry/switch | PASS，auto_hdmi 恢复 true |

全部任务经 auto_hdmi 临时保护（False→任务→True），未触碰 HDMI 画面。
实测发现并修复：合成 NIfTI 需显式 xyzt_units=mm（ACDC 合同拒绝未知单位）。
另：线上 hub 当前对外可达；若后续改为仅回环，桥接自动走入口代理（见 §2 适配说明）。

**真实编译验证的唯一前置**：微信开发者工具 → 设置 → 安全设置 → 开启
**服务端口**。开启后即可用命令行真实编译（或在 IDE 中打开工程即时查看）：

```bash
"C:\Program Files (x86)\Tencent\微信web开发者工具\cli.bat" auto-preview --project "D:\PLD小程序"
```

服务端口未开启时 CLI 会一直等待 IDE 响应（本机已实测此现象），不是工程问题。
