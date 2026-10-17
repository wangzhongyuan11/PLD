"""最小可运行接口检查：B_C_接口说明.md 第 7 节示例的原样版本（用于验证文档可执行）。

与文档示例的唯一差别：本文件位于 software/role_a/final_docs/ 下，需要先把仓库根加入 sys.path；
文档示例面向"在仓库根目录执行"的场景，故不含这段。
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from software.role_a import quant, verify_export
from software.role_a.data import ISICDataset
from software.role_a.train_overall_v1 import assemble, build_default_spec

PACKAGE = ROOT / "deliverables" / "role_a_final"
spec = PACKAGE / "int8" / "model_spec.json"          # 必须是 Path，不是 str
checkpoint = PACKAGE / "best.pt"

print("verify_export:", verify_export.verify(spec))

model, _spec = quant._load_quantized(spec)
model.eval()
assembly = assemble(build_default_spec())
dev = assembly.dev_sets["isic2016_test"]
dataset = ISICDataset(list(dev.items[:1]), root=None, size=256)
item = dataset[0]
image = item.image.unsqueeze(0)
with torch.no_grad():
    logits = model(image)
probability = torch.sigmoid(logits).squeeze(0).squeeze(0).numpy()
valid = item.valid.squeeze(0).numpy() > 0
binary = (probability >= 0.45) & valid
print("image:", dev.items[0].stem, "| valid pixels:", int(valid.sum()),
      "| predicted foreground:", int(binary.sum()),
      "| total pixels:", int(valid.size))
