"""测试辅助：路径设置、依赖缺失检查与合成 ISIC 数据。

依赖缺失时测试必须**跳过并说明原因**，不能假装通过；这里统一提供判断。
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROLE_A_DIR = Path(__file__).resolve().parent.parent


def ensure_role_a_on_path() -> Path:
    """把 ``software/role_a`` 加入 ``sys.path``，返回该目录。"""
    if str(ROLE_A_DIR) not in sys.path:
        sys.path.insert(0, str(ROLE_A_DIR))
    return ROLE_A_DIR


def missing_dependency_note() -> str | None:
    """返回缺失依赖的说明；全部就绪时返回 ``None``。"""
    missing = [name for name in ("numpy", "PIL", "torch")
               if importlib.util.find_spec(name) is None]
    if not missing:
        return None
    return ("缺少依赖 " + ", ".join(missing)
            + "；请先 py -m pip install torch numpy Pillow 再跑这些测试")


MISSING_DEPENDENCY = missing_dependency_note()


@dataclass(frozen=True)
class SyntheticPair:
    """一对合成 ISIC 风格文件的路径。"""

    root: Path
    image: Path
    mask: Path
    stem: str


def write_pair(
    root: Path,
    stem: str,
    width: int = 96,
    height: int = 64,
    box: tuple[int, int, int, int] | None = None,
    mask_values: tuple[int, int] = (0, 255),
    seed: int = 0,
) -> SyntheticPair:
    """写出一对 ``ISIC_*.jpg`` / ``ISIC_*_Segmentation.png``。

    参数:
        box: 前景矩形 ``(x0, y0, x1, y1)``（右下开区间）；省略时取中心 1/4 区域。
        mask_values: 背景/前景像素值，用于构造非严格二值掩膜。
    """
    import numpy as np
    from PIL import Image

    root = Path(root)
    images_dir = root / "images"
    masks_dir = root / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    rgb = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    if box is None:
        box = (width // 4, height // 4, 3 * width // 4, 3 * height // 4)
    x0, y0, x1, y1 = box
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y0:y1, x0:x1] = mask_values[1]

    image_path = images_dir / f"{stem}.jpg"
    mask_path = masks_dir / f"{stem}_Segmentation.png"
    Image.fromarray(rgb, mode="RGB").save(image_path, format="JPEG", quality=95)
    Image.fromarray(mask, mode="L").save(mask_path, format="PNG")
    return SyntheticPair(root=root, image=image_path, mask=mask_path, stem=stem)


def make_dataset(root: Path, count: int = 6, stems: list[str] | None = None) -> list[SyntheticPair]:
    """生成 ``count`` 对合成样本，尺寸不同以覆盖 letterbox 的两种情况。"""
    if stems is None:
        stems = [f"ISIC_{index:07d}" for index in range(1, count + 1)]
    pairs: list[SyntheticPair] = []
    for index, stem in enumerate(stems):
        if index % 2 == 0:
            pairs.append(write_pair(root, stem, width=96, height=64, seed=index))
        else:
            pairs.append(write_pair(root, stem, width=64, height=96, seed=index))
    return pairs


class TemporaryRoot:
    """``tempfile.TemporaryDirectory`` 的轻封装，让测试用 ``with`` 更短。"""

    def __init__(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="role_a_test_")
        self.path = Path(self._tmp.name)

    def __enter__(self) -> Path:
        return self.path

    def __exit__(self, *exc_info: object) -> None:
        self._tmp.cleanup()
