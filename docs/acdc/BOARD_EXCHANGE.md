# PC 与板端堆栈交换接口

完整运行闭包由 Release 提供。PC 先准备全部三方向的中心层任务，B 以独立CAM2设备逐项执行，PC核验head后恢复体数据和重建。导出任务和收集结果均不自动连接板卡、不加载位流，也不以CPU结果补齐缺失的板端响应。

```text
python -m software.role_a_acdc.board_jobs export --image ED.nii.gz --models models --phase ED --output ED_jobs --geometry-policy acdc-direction-only-sform-v1
python -m software.role_a_acdc.board_jobs collect --jobs ED_jobs --output ED_result
```

一般NIfTI使用默认strict几何策略；只有明确识别为ACDC提供方的direction-only sform格式才指定上例策略。输出目录必须是新目录。

`VOLUME_JOBS.json`锚定原图、几何sidecar、模型spec和每个request的SHA；三个方向的子目录里，每层都有`request.json`与196608 B的`input_int8.bin`。输入已归一化和量化，板端不重复resize、归一化或重排。中心层索引在各自方向的完整2mm网格内，邻层越界时复制。

请求必含`architecture_id`、`model_id`、`model_spec_sha256`、`request_id`、`run_id`、`volume_id`、`phase`、`orientation`、`center_index`、`neighbor_indices`、`slice_count`、`geometry_sha256`及shape/layout/量化参数。`model_id`等于模型spec的`profile_id`。每个请求选择对应权重、bias、Q24、CAT LUT及head scale，不能只根据shape命中旧缓存。

B把262144 B的四通道INT8 head和`response.json`写回同一层目录。响应结构如下，身份和尺度必须从当前请求准确回填，散列须对应实际输出文件：

```json
{
  "architecture_id": "ACDC25D-B16-P3-v1",
  "model_id": "acdc_axial_v1",
  "model_spec_sha256": "<from request>",
  "request_id": "<from request>",
  "input_sha256": "<request.input.sha256>",
  "backend": "pynq_pl",
  "synthetic": false,
  "head_scale": "<numeric value from request.head_scale>",
  "output": {
    "path": "head_int8.bin",
    "sha256": "<actual output SHA256>",
    "shape": [1, 4, 256, 256],
    "dtype": "int8",
    "layout": "NCHW",
    "bytes": 262144
  }
}
```

尖括号是说明占位，不能作为实际值；`head_scale`必须为JSON数值。软件模拟应如实标记后端，不能填`pynq_pl`，生产收集器会拒绝合成结果。响应身份与输入散列、完整字节数、合法INT8值、目录边界或head尺度任一不匹配均拒绝；未返回的层不生成最终重建。

PC对各模型head使用其共享scale反量化、softmax，裁去letterbox补边，映射到原生物理网格后等权平均概率，最后argmax。不能直接平均不同scale的INT8值。输出为原生网格NIfTI、三类PLY、体积JSON和离线四面板查看器；ED/ES配对后另提供容积和EF。板端声明的后端身份仍需实际数值、时序和故障恢复验收，不能替代独立板测。
