# 软件参考实现

在项目根目录运行：

```powershell
py software/baseline.py --demo --output runs/demo
py software/baseline.py --demo --threshold 100 --no-opening --output runs/demo_variant
py -m unittest discover -s tests -v
```

`preview.svg` 提供输入、滤波、叠加和掩膜四个视图；`metrics.json` 是可复核记录。示例为 64×64 合成椭圆，仅验证流程。阈值 100 和 128 在该简单图上可能得到相近结果；使用极端阈值观察漏分。

## 真实图像转换（可选，尚未安装依赖）

先获取合法图像和标注，再按实际路径运行：

```powershell
py -m pip install -r software/requirements-optional.txt
py software/prepare_image.py data/raw/example.jpg --mask data/raw/example_mask.png --output data/processed/example
py software/baseline.py --input data/processed/example/input.ppm --mask data/processed/example/truth.pgm --valid-mask data/processed/example/valid.pgm --output runs/example
```

上面的 example 是占位文件名，项目不包含这些数据。无标注时省略两条命令中的 `--mask` 参数，不会输出 Dice/IoU。转换器保持宽高比并输出有效区域；评估必须传入 `--valid-mask`，排除补边。所有掩膜按严格 0/255 读取。

参考算法采用灰度暗区域假设，并不能可靠覆盖所有皮肤镜病灶。暂未实现自动阈值、毛发去除、最大连通域、网络训练、硬件通信或批量数据集评估。

真实数据结果在处理后的有效区域评估，不能未经协议对齐就与论文指标比较。转换器依赖 Pillow，初次安装后记录具体版本以便复现。
