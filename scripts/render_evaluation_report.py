"""Render test-set metrics and explicitly labelled software masks from artifacts."""
import argparse
import hashlib
import html
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reference.datasets import validate_manifest
from reference.tinyunet import Model, DEFAULT_MODEL_ROOT


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--board", type=Path, help="completed full-set report from evaluate_dataset.py")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_ROOT)
    args = parser.parse_args()
    manifest_path = args.dataset/"manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest["records"]
    model = Model(args.model)
    if not validate_manifest(manifest, model):
        raise ValueError("report requires the complete named collection")
    count = len(records)
    collection_name = manifest.get("collection", {}).get("name", "isic2016_test379")
    threshold_q = model.threshold["threshold_q"]
    board = None
    if args.board:
        board = json.loads(args.board.read_text(encoding="utf-8"))
        if not board["passed"] or not board["full_test_set"] or board["dataset_manifest_sha256"] != sha(manifest_path):
            raise ValueError("board report incomplete or belongs to different references")
        by_id = {r["id"]: r for r in records}
        seen = set()
        for frame in board["frames"]:
            record = by_id[frame["id"]]
            if frame["output_sha256"] != record["head_sha256"] or frame["metrics"] != record["int8"]:
                raise ValueError("board output or metric differs from reference")
            seen.add(frame["id"])
        if seen != set(by_id):
            raise ValueError("board report does not cover the complete collection")
    args.output.mkdir(parents=True, exist_ok=True)
    f = np.array([r["fp32"]["dice"] for r in records])
    q = np.array([r["int8"]["dice"] for r in records])
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].scatter(f, q, s=15, alpha=.65, color="#21618c")
    axes[0].plot([0, 1], [0, 1], "--", color="#999999", lw=1)
    axes[0].set(xlabel="FP32 Dice", ylabel="INT8 software Dice", title=f"{count} images: {collection_name}", xlim=(0, 1), ylim=(0, 1))
    axes[1].hist(q-f, bins=35, color="#2874a6", edgecolor="white")
    axes[1].axvline(0, color="#777777", lw=1)
    axes[1].set(xlabel="INT8 Dice - FP32 Dice", ylabel="Images", title=f"Mean change: {np.mean(q-f):+.6f}")
    fig.savefig(args.output/"accuracy.png", dpi=160)
    plt.close(fig)

    # Show failures as well as averages. Display quantized input, not a JPEG
    # redecoded by an unverified plotting runtime.
    selected = sorted(records, key=lambda r: r["int8"]["dice"])[:5]
    spec = model.root/"model_spec.json"
    if sha(spec) != manifest["model_spec_sha256"]:
        raise ValueError("plotting model scale does not match dataset")
    scale = json.loads(spec.read_text(encoding="utf-8"))["layers"][0]["input_scale"]
    fig, axes = plt.subplots(5, 4, figsize=(10, 12), constrained_layout=True)
    for row, record in enumerate(selected):
        path = args.dataset/record["file"]
        if sha(path) != record["sha256"]:
            raise ValueError("plot artifact hash mismatch")
        with np.load(path, allow_pickle=False) as data:
            valid = data["valid"]
            rgb = np.clip(data["input"][0].transpose(1, 2, 0).astype(np.float32)*scale, 0, 1)
            masks = [data["target"] & valid, data["fp32_mask"] & valid, (data["head"][0, 0] >= threshold_q) & valid]
            axes[row, 0].imshow(rgb)
            for col, mask in enumerate(masks, 1):
                axes[row, col].imshow(mask, cmap="gray", vmin=0, vmax=1)
        labels = [record["id"]+"\nQuantized input", "Ground truth",
                  f"FP32 Dice {record['fp32']['dice']:.4f}", f"INT8 software Dice {record['int8']['dice']:.4f}"]
        for col in range(4):
            axes[row, col].set_title(labels[col], fontsize=9)
            axes[row, col].axis("off")
    fig.suptitle("Five lowest INT8 Dice scores — engineering evaluation, not clinical diagnosis", fontsize=12)
    fig.savefig(args.output/"lowest_scores.png", dpi=130)
    plt.close(fig)
    extra = f"<p>完整{count}图PL验收尚未附入此报告；下表与掩膜为软件参考。</p>"
    if board:
        times = np.array([r["seconds"] for r in board["frames"]])
        online = np.array([r["online_b_seconds"] for r in board["frames"]])
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        axes[0].plot(times, lw=.8, label="Quantized input to head")
        if board["original_jpeg_inputs"]:
            axes[0].plot(online, lw=.8, label="With online preprocessing")
        axes[0].set(xlabel="Frame", ylabel="Seconds", title="B subsystem latency")
        axes[0].legend(fontsize=8)
        axes[1].plot([r["memory"]["VmRSS_KiB"]/1024 for r in board["frames"]], lw=.8)
        axes[1].set(xlabel="Frame", ylabel="RSS (MiB)", title="Process memory; image sizes vary")
        fig.savefig(args.output/"board.png", dpi=160)
        plt.close(fig)
        extra = (f"<p>附入板测共{len(times)}帧，覆盖{count}图，head哈希及指标计数全部一致。"
                 f"量化输入到head：中位{np.median(times):.6f}秒，P95 {np.percentile(times,95):.6f}秒。"
                 "计时不含JPEG解码、比较、文件I/O及C后处理/显示。</p><img src='board.png' alt='板上延迟与RSS'>")
    rows = "".join(f"<tr><td>{kind}</td><td>{manifest['means'][key]['dice']:.9f}</td><td>{manifest['means'][key]['iou']:.9f}</td></tr>"
                   for kind, key in (("FP32", "fp32"), ("INT8软件", "int8")))
    document = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>ISIC 模型评估</title>
<style>body{{font:16px/1.7 system-ui,sans-serif;max-width:1050px;margin:36px auto;padding:0 20px;color:#193247;background:#fafbfc}}img{{width:100%;height:auto}}td,th{{padding:8px 24px;border-bottom:1px solid #ccd6df;text-align:left}}code{{overflow-wrap:anywhere}}small{{color:#536878}}</style>
<h1>{html.escape(collection_name)}：{count}图评估</h1><p>仅用于工程教学与算法演示，不作为临床诊断依据。</p>
<p>模型分辨率letterbox有效区；最近邻GT；概率阈值{model.threshold_probability}，INT8 q≥{threshold_q}；逐图Dice/IoU宏平均。未加入后处理。</p>
<table><tr><th>路径</th><th>Dice</th><th>IoU</th></tr>{rows}</table>
<p>A记录核对：<code>{html.escape(json.dumps(manifest.get("a_metrics", {}), ensure_ascii=False))}</code></p>
{extra}<img src="accuracy.png" alt="量化前后逐图Dice与差异分布"><img src="lowest_scores.png" alt="五张最低分样例">
<p><small>参考manifest SHA256：<code>{sha(manifest_path)}</code><br>模型规格：<code>{html.escape(manifest['model_spec_sha256'])}</code></small></p></html>"""
    (args.output/"index.html").write_text(document, encoding="utf-8")
    print(f"EVALUATION_REPORT_PASS path={args.output/'index.html'} board_attached={board is not None}")


if __name__ == "__main__":
    main()
