"""训练/评估集成测试：真实跑 CLI 的三个子命令（CPU、小图、极少样本）。

它检查的是"管线是否真的能用、产物是否齐全、测试集是否只走独立命令"，
**不检查精度**，也不会产生任何可引用的精度结论。
"""

from __future__ import annotations

import json
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from _support import MISSING_DEPENDENCY, TemporaryRoot, ensure_role_a_on_path  # noqa: E402

ensure_role_a_on_path()

import torch  # noqa: E402

from _support import make_dataset  # noqa: E402

import train  # noqa: E402


def _run(arguments: list[str]) -> tuple[int, str]:
    """调用 ``train.main``，返回 ``(exit_code, stdout)``；错误信息留在 stderr。"""
    buffer = StringIO()
    with redirect_stdout(buffer):
        code = train.main(arguments)
    return code, buffer.getvalue()


@unittest.skipIf(MISSING_DEPENDENCY is not None, MISSING_DEPENDENCY or "")
class CliPipelineTests(unittest.TestCase):
    def test_windows_spawn_compatible_loader(self) -> None:
        with TemporaryRoot() as root:
            make_dataset(root, count=2)
            pairs = train.enumerate_isic_pairs(root / "images", root / "masks")
            loader = train.make_loader(pairs, root / "images", size=32,
                                       batch_size=1, shuffle=False, seed=17,
                                       num_workers=1)
            self.assertEqual(tuple(next(iter(loader))["image"].shape), (1, 3, 32, 32))

    def _prepare(self, root: Path, count: int = 6) -> dict[str, Path]:
        make_dataset(root, count=count)
        split_dir = root / "split"
        code, output = _run([
            "prepare-split",
            "--image-dir", str(root / "images"),
            "--mask-dir", str(root / "masks"),
            "--output-dir", str(split_dir),
            "--seed", "17",
            "--val-fraction", "0.34",
            "--no-stratify",
        ])
        self.assertEqual(code, 0, output)
        report = json.loads(output)
        self.assertEqual(report["pairs"], count)
        self.assertEqual(report["train"] + report["val"], count)
        self.assertNotIn("test", report)
        return {"split": split_dir / "split.json", "split_dir": split_dir}

    def test_prepare_split_writes_reproducible_files(self) -> None:
        with TemporaryRoot() as root:
            paths = self._prepare(root)
            payload = json.loads(paths["split"].read_text(encoding="utf-8"))
            self.assertEqual(payload["seed"], 17)
            self.assertEqual(set(payload["splits"]), {"train", "val"})
            self.assertEqual(len(payload["pairs"]), 6)
            csv_text = (paths["split_dir"] / "split.csv").read_text(encoding="utf-8")
            self.assertIn("role,stem,image,mask", csv_text)
            self.assertEqual(len(csv_text.strip().splitlines()), 7)

            # 再跑一次必须得到完全相同的划分。
            again = root / "split_again"
            _run(["prepare-split", "--image-dir", str(root / "images"),
                  "--mask-dir", str(root / "masks"), "--output-dir", str(again),
                  "--seed", "17", "--val-fraction", "0.34", "--no-stratify"])
            self.assertEqual(
                (again / "split.json").read_text(encoding="utf-8"),
                paths["split"].read_text(encoding="utf-8"),
            )

    def test_train_writes_checkpoint_config_and_metrics(self) -> None:
        with TemporaryRoot() as root:
            paths = self._prepare(root)
            run_dir = root / "run"
            code, output = _run([
                "train",
                "--image-dir", str(root / "images"),
                "--mask-dir", str(root / "masks"),
                "--split", str(paths["split"]),
                "--output-dir", str(run_dir),
                "--epochs", "2",
                "--batch-size", "2",
                "--lr", "0.01",
                "--size", "32",
                "--base-channels", "4",
                "--num-pools", "2",
                "--seed", "5",
                "--device", "cpu",
            ])
            self.assertEqual(code, 0, output)
            for name in ("best.pt", "config.json", "split_used.json", "metrics.csv",
                         "metrics.json", "model_graph.json"):
                self.assertTrue((run_dir / name).is_file(), f"缺少产物 {name}")

            config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["seed"], 5)
            self.assertEqual(config["epochs"], 2)
            self.assertEqual(config["size"], 32)
            self.assertEqual(config["device"], "cpu")

            rows = (run_dir / "metrics.csv").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(rows), 3)  # 表头 + 2 个 epoch
            summary = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            self.assertIn("best_val_dice_mean", summary)
            self.assertEqual(summary["best_epoch"], 1)

            # checkpoint 必须能用 weights_only=True 加载（不依赖任意 pickle）。
            checkpoint = torch.load(run_dir / "best.pt", map_location="cpu",
                                    weights_only=True)
            self.assertIn("model_state", checkpoint)
            self.assertEqual(checkpoint["config"]["seed"], 5)
            self.assertEqual(checkpoint["epoch"], 1)

    def test_train_rejects_split_with_test_role(self) -> None:
        with TemporaryRoot() as root:
            paths = self._prepare(root)
            payload = json.loads(paths["split"].read_text(encoding="utf-8"))
            payload["splits"]["test"] = [payload["splits"]["train"][0]]
            leaky = root / "leaky.json"
            leaky.write_text(json.dumps(payload), encoding="utf-8")
            code, _ = _run([
                "train", "--image-dir", str(root / "images"), "--mask-dir", str(root / "masks"),
                "--split", str(leaky), "--output-dir", str(root / "run_leaky"),
                "--epochs", "1", "--size", "32", "--device", "cpu",
            ])
            self.assertEqual(code, 2)
            self.assertFalse((root / "run_leaky" / "best.pt").exists())

    def test_train_rejects_missing_file_in_split(self) -> None:
        with TemporaryRoot() as root:
            paths = self._prepare(root)
            payload = json.loads(paths["split"].read_text(encoding="utf-8"))
            payload["splits"]["val"] = ["ISIC_9999999"]
            broken = root / "broken.json"
            broken.write_text(json.dumps(payload), encoding="utf-8")
            code, _ = _run([
                "train", "--image-dir", str(root / "images"), "--mask-dir", str(root / "masks"),
                "--split", str(broken), "--output-dir", str(root / "run_broken"),
                "--epochs", "1", "--size", "32", "--device", "cpu",
            ])
            self.assertEqual(code, 2)

    def test_evaluate_test_requires_explicit_test_split(self) -> None:
        with TemporaryRoot() as root:
            paths = self._prepare(root)
            run_dir = root / "run"
            _run(["train", "--image-dir", str(root / "images"), "--mask-dir", str(root / "masks"),
                  "--split", str(paths["split"]), "--output-dir", str(run_dir),
                  "--epochs", "1", "--size", "32", "--base-channels", "4", "--num-pools", "2",
                  "--device", "cpu"])
            # 用只有 train/val 的划分去评估测试集：必须失败，而不是偷偷用 val。
            code, _ = _run([
                "evaluate-test", "--image-dir", str(root / "images"),
                "--mask-dir", str(root / "masks"), "--split", str(paths["split"]),
                "--checkpoint", str(run_dir / "best.pt"), "--size", "32", "--device", "cpu",
            ])
            self.assertEqual(code, 2)

    def test_evaluate_test_runs_and_ignores_model_selection(self) -> None:
        with TemporaryRoot() as root:
            make_dataset(root, count=6)
            # 人为把两个样本分到 test 角色（模拟官方测试集单独目录）。
            test_dir = root / "official_test"
            (test_dir / "images").mkdir(parents=True)
            (test_dir / "masks").mkdir(parents=True)
            stem = "ISIC_0000006"
            for source, target in ((root / "images" / f"{stem}.jpg",
                                    test_dir / "images" / f"{stem}.jpg"),
                                   (root / "masks" / f"{stem}_Segmentation.png",
                                    test_dir / "masks" / f"{stem}_Segmentation.png")):
                target.write_bytes(source.read_bytes())
                source.unlink()

            split_dir = root / "split_all"
            code, output = _run([
                "prepare-split", "--image-dir", str(root / "images"),
                "--mask-dir", str(root / "masks"), "--output-dir", str(split_dir),
                "--seed", "3", "--val-fraction", "0.4", "--no-stratify",
                "--include-test", "--test-image-dir", str(test_dir / "images"),
                "--test-mask-dir", str(test_dir / "masks"),
            ])
            self.assertEqual(code, 0, output)
            payload = json.loads((split_dir / "split.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["splits"]["test"], [stem])
            self.assertNotIn(stem, payload["splits"]["train"])
            self.assertNotIn(stem, payload["splits"]["val"])
            self.assertTrue(payload["pairs"][stem]["image"].endswith(f"{stem}.jpg"))
            self.assertIn(f"test,{stem},", (split_dir / "split.csv").read_text(encoding="utf-8"))

            run_dir = root / "run"
            train_split_dir = root / "split_train"
            split_code, split_output = _run([
                "prepare-split", "--image-dir", str(root / "images"),
                "--mask-dir", str(root / "masks"), "--output-dir", str(train_split_dir),
                "--seed", "3", "--val-fraction", "0.4", "--no-stratify",
            ])
            self.assertEqual(split_code, 0, split_output)
            train_code, train_output = _run(["train", "--image-dir", str(root / "images"), "--mask-dir", str(root / "masks"),
                  "--split", str(train_split_dir / "split.json"), "--output-dir", str(run_dir),
                  "--epochs", "1", "--size", "32", "--base-channels", "4", "--num-pools", "2",
                  "--device", "cpu"])
            self.assertEqual(train_code, 0, train_output)

            report_path = root / "test_report.json"
            code, output = _run([
                "evaluate-test", "--image-dir", str(root / "images"),
                "--mask-dir", str(root / "masks"),
                "--test-image-dir", str(test_dir / "images"),
                "--test-mask-dir", str(test_dir / "masks"),
                "--split", str(split_dir / "split.json"),
                "--checkpoint", str(run_dir / "best.pt"),
                "--size", "32", "--device", "cpu",
                "--output-json", str(report_path),
                "--per-image-csv", str(root / "test_per_image.csv"),
            ])
            self.assertEqual(code, 0, output)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["test_images"], 1)
            self.assertIn("有效区域", report["evaluation_space"])
            self.assertIn("dice_mean", report)
            self.assertEqual(report["checkpoint_selected_by"], "val_dice_mean")
            per_image = (root / "test_per_image.csv").read_text(encoding="utf-8")
            self.assertIn("filename,dice,iou,empty_target,empty_prediction,valid_pixels", per_image)
            self.assertEqual(len(per_image.strip().splitlines()), 2)
            # 评估不会覆盖最佳 checkpoint。
            before = (run_dir / "best.pt").read_bytes()
            _run(["evaluate-test", "--image-dir", str(root / "images"),
                  "--mask-dir", str(root / "masks"),
                  "--test-image-dir", str(test_dir / "images"),
                  "--test-mask-dir", str(test_dir / "masks"),
                  "--split", str(split_dir / "split.json"),
                  "--checkpoint", str(run_dir / "best.pt"), "--size", "32", "--device", "cpu"])
            self.assertEqual(before, (run_dir / "best.pt").read_bytes())

    def test_describe_model_cli(self) -> None:
        code, output = _run(["describe-model", "--size", "64", "--base-channels", "8",
                             "--num-pools", "3"])
        self.assertEqual(code, 0)
        graph = json.loads(output)
        self.assertTrue(graph["allowed_only"])
        self.assertEqual(graph["output_shape"], [1, 1, 64, 64])

    def test_bad_device_is_reported_not_silently_ignored(self) -> None:
        with TemporaryRoot() as root:
            paths = self._prepare(root)
            code, _ = _run([
                "train", "--image-dir", str(root / "images"), "--mask-dir", str(root / "masks"),
                "--split", str(paths["split"]), "--output-dir", str(root / "run_bad"),
                "--epochs", "1", "--size", "32", "--device", "not-a-device",
            ])
            self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
