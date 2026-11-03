"""学习练习：补全 TN、FP、FN 的判断条件，再运行核对输出。"""

# 二维列表：外层表示行，内层表示这一行的各列。
reference = [[1, 1, 0], [0, 1, 0]]
prediction = [[1, 2, 0], [0, 1, 0]]

# TODO：将下面三个 None 换成条件表达式。
# g 是参考值，p 是预测值；例如真正例为 g == 1 and p == 1。
def classify(g, p):
    if g == 1 and p == 1:
        return "TP"
    elif g == 0 and p == 0:  # TODO：参考为 0，预测也为 0。
        return "TN"
    elif g == 0 and p == 1:  # TODO：参考为 0，预测为 1，误选背景。
        return "FP"
    elif g == 1 and p == 0:  # TODO：参考为 1，预测为 0，漏选目标。
        return "FN"
    return "未分类"

def mask_shape(mask):
    if len(mask) == 0:
        raise ValueError("掩膜不能没有行")

    width = len(mask[0])
    if width == 0:
        raise ValueError("掩膜不能没有列")

    for row in mask:
        if len(row) != width:
            raise ValueError("掩膜各行必须等宽")
    return (len(mask), width)

g_shape = mask_shape(reference)
p_shape = mask_shape(prediction)

if g_shape != p_shape:  # 替换为：两张掩膜形状不一致
    raise ValueError("预测与参考尺寸不一致")
    
def check_binary(mask, name):
    for row_no, row in enumerate(mask, start=1):
        for col_no, value in enumerate(row, start=1):
            if value not in (0, 1):  # value 不属于允许的 0、1
                raise ValueError(
                    f"{name}({row_no},{col_no})={value}，只允许 0 或 1"
                )

check_binary(reference, "参考")
check_binary(prediction, "预测")

counts = {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "未分类": 0}
for row in range(len(reference)):
    for col in range(len(reference[row])):
        g = reference[row][col]
        p = prediction[row][col]
        label = classify(g, p)
        counts[label] += 1
        print(f"位置({row + 1},{col + 1})：参考={g}，预测={p} → {label}")

print("计数：", counts)
if counts["未分类"]:
    print("练习尚未完成：请补全 classify 中的三个 None 条件。")
else:
    print("所有位置已分类，请逐项对照手算；完成分类不等于分类正确。")

tp = counts["TP"]
fp = counts["FP"]
fn = counts["FN"]

dice_den = 2 * tp + fp + fn
if dice_den == 0:
    dice = 1.0
else:
    dice = 2 * tp / dice_den
  
iou_den = tp + fp + fn
if iou_den == 0:
    iou = 1.0
else:
    iou = tp / iou_den

print(f"Dice={dice:.4f}, IoU={iou:.4f}")
