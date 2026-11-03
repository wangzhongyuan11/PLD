# 本机交互分割接口约定

两套服务具有独立路由：皮肤前缀 `/api/skin`，CAMUS 前缀 `/api/camus`。一个服务不会接收另一个模型的路由。

## 路由

|方法|路径（将 `<model>` 替换为该服务名）|作用|
|---|---|---|
|GET|`/health`|服务存活与所属模型；不表示已加载模型|
|GET|`/api/<model>/capabilities`|支持的版本、类别、阈值、交互能力|
|GET|`/api/<model>/sample`|可用的本地开发样例，返回 PNG data URL；不返回 GT|
|POST|`/api/<model>/image`|解码并规范化上传图像，返回原图尺寸与 RGB PNG data URL|
|POST|`/api/<model>/predict`|冻结模型推理与可选交互分割|

POST 使用 `Content-Type: application/json`，不使用 multipart。没有文件路径或远程 URL 读取接口。上传原文件上限 16 MiB，解码图像上限 1600 万像素。支持 PNG/JPEG/BMP/WebP/TIFF；CAMUS 另外支持 2D 单帧 `.nii/.nii.gz`，多帧文件需先选取单帧。NIfTI 按既有 CAMUS X/Y 轴约定转换，原始灰度限于 0～255 后转 uint8，不做每张图的自适应拉伸。

请求格式：

```json
{
  "image": "data:image/png;base64,<图像字节的Base64>",
  "filename": "image.png",
  "parameters": {
    "variant": "delivery",
    "backend": "int8",
    "target_class": 1,
    "threshold": 0.45,
    "roi": null,
    "box": null,
    "points": []
  }
}
```

也接受不含 data URL 前缀的纯 Base64。`filename` 只用于识别格式，不作为本地路径打开；NIfTI 必须保留正确后缀。若先调用 `/image` 得到 PNG，再请求 `/predict`，应使用 `filename:"normalized.png"`。未知参数会报错，避免拼写错误静默失效。

## 参数

|参数|约定|
|---|---|
|`variant`|皮肤仅 `delivery`；CAMUS 为 `delivery` 或 `candidate`，默认 `delivery`|
|`backend`|`int8` 或 `fp32`，默认 `int8`|
|`target_class`|皮肤为整数 1；CAMUS 为整数 1 左室腔 / 2 心肌 / 3 左房，默认 1|
|`threshold`|有限数值且严格在 (0,1)；皮肤省略或 null 使用 0.45，CAMUS 省略或 null 保留 argmax|
|`roi`|`{x0,y0,x1,y1}`，模型推理前裁剪；缺省/null 为整图|
|`box`|同样的坐标对象，限定目标范围并触发图割；缺省/null 不使用框提示|
|`points`|最多 128 个 `{x,y,label}`；label 为 `foreground` / `background`；缺省为空列表|

所有坐标为**规范化后的原图像素**，与网页缩放尺寸无关。矩形左上包含、右下不包含，即 `[x0,x1) × [y0,y1)`；至少 2×2 像素。浮点矩形左上向下取整、右下向上取整；点坐标向下取整。矩形必须在图内；box 必须在 ROI 内；全部点必须在 ROI 内，前景点还必须在 box 内。同像素相反标签、越界、非有限值或图割缩放后冲突均返回明确错误。

## 返回值

`outputs` 的三个字段均为 PNG data URL，可解码后保存：

|字段|内容|
|---|---|
|`labels_png`|原图尺寸、8 bit 类别编号。皮肤 0/1；CAMUS 0/1/2/3|
|`target_mask_png`|原图尺寸、8 bit 目标掩膜，0 为非目标，255 为所选目标|
|`overlay_png`|RGB 原图叠加所选结构及其轮廓，不将目标矩形绘入掩膜|

`parameters` 回显实际生效值；`identity` 包含版本、后端、冻结权重和模型规格 SHA256、缓存命中及软件/板端标志。相同图像、ROI、版本和后端可复用概率缓存，改变提示或阈值时仍重新执行相应分割步骤。

`statistics` 含目标像素、实际处理像素、区域内各类别计数、ROI/整图面积比及图割改变像素数。区域外标签固定为 0。`elapsed_ms` 是服务本次请求的实际软件耗时（包括缓存影响），不能当作板端时延。`network_prompt_conditioned:false` 明确原网络未提示式训练；`refinement` 为 null 或真实图割信息。

没有上传参考标注的功能，故 `reference_available:false`、`accuracy_metrics:null`。`warnings` 披露空目标、ROI 限制及交互精度未独立验证等情况。

## 错误与并发

400 参数/图像格式错误；403 非本机 Host 或跨站 Origin；404 路由不属于该模型服务；409 同服务前一帧仍在推理；413 请求体超限；415 请求不是 JSON；500 模型文件校验或推理失败；503 开发样例不可用。出错时返回 `{ "error": "原因" }`，不会返回先前预测作为本次结果。

网页对操作做短暂合并、串行推理，参数或图像改变后废弃旧请求的返回；下载仅对当前有效结果启用。其他客户端收到 409 后可等待并重新发起。请求图像、提示和推理缓存驻留内存，不自动写入训练集或上传外部服务。
