# ACDC内部模型交换格式 v1

> 2026-10-19实施更新：`release.py`已适配A的`acdc-int8-model-v1`，通过固定manifest验证所需子集、重算40体验收门槛并保留原spec SHA。`load_bundle`可直接读取发布根目录或Axial int8目录；`import-release`生成65文件独立子集。A只交付输入/head，21节点由B独立INT64/Torch见证交叉验证，未伪称A已交付逐节点张量。当前软件复现和硬件限制见[实施状态](IMPLEMENTATION_STATUS.md)。以下是保留的B内部/合成格式说明。

该格式是B侧已实现的内部规范化包；不声称等于A尚未提供的MODEL_CONTRACT/BOARD_EXCHANGE文件布局。保持ACDC25D-B16-P3-v1的固定拓扑、数值与类别，未来适配A实际导出时只转换文件描述，不改变数值。`create_synthetic_bundle`可生成一个完整可执行的非生产样例。

目录根为model_spec.json。所有张量为连续raw bytes，不使用pickle/npy对象；每个文件记录字段为path（相对根、禁止越界/符号链接）、bytes、sha256、dtype（`|i1`/`<i4`/`<f8`）、shape。生产spec SHA必须由外部参数固定。

spec字段：

| 字段 | 规定 |
|---|---|
| schema | pld-acdc-model-v1 |
| architecture_id / model_id / orientation | ACDC25D-B16-P3-v1 / acdc_axial_v1 / axial |
| preprocessing_id | acdc-production-numerics-20261018-v1 |
| labels | [background,RV,MYO,LV] |
| synthetic | 显式boolean；生产为false，且必须满足下方资格门 |
| layers | 15行，name=C01..C15，relu前14为true；input_scale/output_scale与固定图边连续 |
| 每层参数 | weight（OIHW）、bias、multiplier、weight_scales文件描述；形状按计划固定 |
| concats | 3行，level=2/1/0；source_scales、common_scale=max、multiplier_q24两路RNE |
| head_scale | C15输出scale，四类共享正数 |

生产spec额外引用（同样必须是长度/hash校验的文件描述）：

- acceptance：JSON资格报告，status=passed、orientation=axial、integer_oracle_mismatches=0、verified_conv_layers=15；volumes必须是20名患者各ED/ES共40行，包含patient/phase、3元素fp32_dice/int8_dice及new_empty_nonempty_gt=0。导入器重新计算A的总体/逐类/单体Dice下降门槛。
- fp32_checkpoint：源权重工件，只校验SHA，不执行不可信pickle。
- implementation_binding：JSON，preprocessing_id匹配；versions给出numpy/scipy/nibabel/torch全部冻结版本，后两项必须5.4.2/2.11.0。PC执行时检查当前环境；板端不安装Torch。
- golden：input、head文件描述和按固定21节点排序的nodes列表；每个nodes行包含node和file描述。`verify-model --full`实际执行独立参考和tile emulator，并对A的节点与head逐元素比较。

integer范围检查按每通道abs(bias)+127*sum(abs(weight))≤INT32_MAX；M必须正INT32且等于scale推导的RNE。Pool保持scale；CAT两路分别转换。任何一个未通过项拒绝生产运行。

运行时交换协议由protocol.py实现：4字节小端JSON头长度、UTF8 JSON头、固定长度raw tensor。JSON头≤64KiB，tensor SHA/字节数校验。请求绑定Axial-only profile、phase、center/neighbors、model spec SHA和geometry SHA；响应附真实backend、synthetic、head_scale和计时。worker使用有界8帧幂等响应缓存，缓存不计新PL样本。

运行包的runtime_manifest.json绑定模型spec、所有工件、实际执行源码和硬件ID/ABI；构建证书绑定RTL/IP和bit/HWH/XSA。加载器不把文件自称passed当成已完成实际板测。
