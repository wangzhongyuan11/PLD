def classify(g, p):
    if g == 1 and p == 1:
        return "TP"
    elif g == 0 and p == 0:  # TODO：
        return "TN"
    elif g == 0 and p == 1: 
        return "FP"
    elif g == 1 and p == 0:  
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
    
def check_binary(mask, name):
    for row_no, row in enumerate(mask, start=1):
        for col_no, value in enumerate(row, start=1):
            if value not in (0, 1):  
                raise ValueError(
                    f"{name}({row_no},{col_no})={value}，只允许 0 或 1"
                )

def count_pixels(reference, prediction):
    g_shape = mask_shape(reference)
    p_shape = mask_shape(prediction)
    if g_shape != p_shape:  
        raise ValueError("预测与参考尺寸不一致")
    check_binary(reference, "实际")
    check_binary(prediction, "预测")
    counts = {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "未分类": 0}
    for row in range(len(reference)):
        for col in range(len(reference[row])):
            g = reference[row][col]
            p = prediction[row][col]
            label = classify(g, p)
            counts[label] += 1
            print(f"位置({row + 1},{col + 1})：参考={g}，预测={p} → {label}")
    return counts

def compute_metrics(counts):
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
    return {"dice": dice, "iou": iou}

def main():
    g = [[1, 1, 0], [0, 1, 0]]
    p1 = [[1, 0, 0], [0, 1, 0]]
    p2 = [[1, 1, 0], [0, 1, 0]]

    c1 = count_pixels(g, p1)
    c2 = count_pixels(g, p2)

    print("第一次：", c1)
    print("第二次：", c2)

    m1 = compute_metrics(c1)
    m2 = compute_metrics(c2)

    print("第一次指标：", m1)
    print("第二次指标：", m2)

    empty = [[0, 0, 0], [0, 0, 0]]
    c3 = count_pixels(empty, empty)
    print("双空指标：", compute_metrics(c3))
    
if __name__ == "__main__":
    main()