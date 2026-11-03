from pathlib import Path
from baseline import read_pnm
from learning_02_function import mask_shape, check_binary
from learning_02_function import count_pixels, compute_metrics
from baseline import read_pnm, write_pnm

project_dir = Path(__file__).resolve().parent.parent
image_path = project_dir / "data" / "learning" / "tiny_gray.pgm"

width, height, values = read_pnm(image_path, "P2")

print("宽、高：", width, height)
print("像素：", values)
print(values[3])

threshold = 170
mask = []
for value in values:
    if value < threshold:
        mask.append(1)
    else:
        mask.append(0)

print("输入数量：", len(values))
print("输出数量：", len(mask))
print("掩膜：", mask)

if len(mask) != width * height:
    raise ValueError("掩膜长度与图像尺寸不一致")

mask_2d = []
for row in range(height):
    start = row * width
    row_values = mask[start:(start + width)]  # 用切片取出从 start 开始的一行
    mask_2d.append(row_values)
print("二维掩膜：", mask_2d)

print("行、列：", mask_shape(mask_2d))
check_binary(mask_2d, "预测")
print("二值检查通过")

reference = [[0, 1, 0], [1, 0, 0]]
counts = count_pixels(reference, mask_2d)
metrics = compute_metrics(counts)

print("阈值：", threshold)
print("目标面积：", counts["TP"] + counts["FP"])
print("计数：", counts)
print("指标：", metrics)

display_values = []
for value in mask:
    display_values.append(value * 255)  # 将 0/1 映射为 0/255

output_dir = project_dir / "runs" / "learning"
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / f"tiny_mask_t{threshold}.pgm"
write_pnm(output_path, width, height, display_values)

saved_width, saved_height, saved_values = read_pnm(output_path, "P2")

print("保存位置：", output_path)
print("回读宽高：", saved_width, saved_height)
print("回读像素：", saved_values)
print("尺寸一致：", (saved_width, saved_height) == (width, height))
print("像素一致：", saved_values == display_values)