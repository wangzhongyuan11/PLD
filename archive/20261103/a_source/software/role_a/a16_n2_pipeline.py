"""A16-N2 管线：官方原图下载（.part→校验→原子改名）、掩膜配对、规范化、对照页、重复筛查。

用法：
  python -m software.role_a.a16_n2_pipeline --smoke 2   # 2 张冒烟
  python -m software.role_a.a16_n2_pipeline --all       # 48 张全量
"""
from __future__ import annotations

import argparse, csv, hashlib, io, json, os, subprocess, zipfile
from pathlib import Path
from typing import Any, Sequence
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path("data/raw/isic2018_novice_pilot")
AUDIT = Path("runs/role_a_a16_novice_image_pilot")
PILOT = Path("runs/role_a_a16_novice_mask_audit/pilot_selection.csv")
API_META = AUDIT/"api_metadata.json"
MASK_PREFIX = "Masks/"
TARGET = (600, 450)          # 掩膜固定尺寸；宽高比 4:3
ASPECT_TOL = 0.01            # 1%
MAX_BYTES_PER_IMAGE = 40_000_000


def sha256_file(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as s:
        for b in iter(lambda: s.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def load_inputs() -> tuple[list[dict], dict[str, dict], Path]:
    rows = list(csv.DictReader(PILOT.open(encoding="utf-8", newline="")))
    meta = json.loads(API_META.read_text(encoding="utf-8-sig"))  # 该文件由 PowerShell 写入，带 UTF-8 BOM（不修改原文件以保持已验证 SHA256）
    by_id = {m["isic_id"]: m for m in meta["images"]}
    return rows, by_id, ROOT/"Masks.zip"


def head_length(url: str) -> int | None:
    """以 HTTP 响应头 Content-Length 作为权威长度（API files.full.size 已被证伪，见 REPORT）。"""
    out = subprocess.run(["curl.exe", "-sSL", "-I", "--max-time", "30", url],
                         capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.lower().startswith("content-length:"):
            return int(line.split(":", 1)[1].strip())
    return None


def download(url: str, target: Path, expected: int, ledger: list) -> dict:
    """写 .part → 校验字节/SHA256 → 原子改名。有界：总超时 + 低速终止 + 最多 1 次重试。"""
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
    if target.exists() and target.stat().st_size == expected:
        return {"reused": True, "bytes": expected, "sha256": sha256_file(target)}
    total = if_true = 0
    for attempt in (1, 2):
        before = part.stat().st_size if part.exists() else 0
        cmd = ["curl.exe", "-sS", "-L", "--fail", "--http1.1", "-C", "-",
               "--max-time", "120", "--speed-limit", "2048", "--speed-time", "30",
               "-o", str(part), url]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        after = part.stat().st_size if part.exists() else 0
        ledger.append({"attempt": attempt, "exit": proc.returncode, "before": before,
                       "after": after, "stderr": proc.stderr.strip()[:200]})
        if proc.returncode == 0 and after == expected:
            break
    if not part.exists() or part.stat().st_size != expected:
        raise RuntimeError(f"下载未达预期长度: {part} {part.stat().st_size if part.exists() else 0} != {expected}")
    digest = sha256_file(part)
    os.replace(part, target)
    return {"reused": False, "bytes": target.stat().st_size, "sha256": digest}


def mask_image(zf: zipfile.ZipFile, isic_id: str) -> Image.Image:
    name = f"{MASK_PREFIX}{isic_id}_UNET.png"
    with zf.open(name) as s:
        return Image.open(io.BytesIO(s.read())).convert("L")


def head_probe(url: str) -> dict:
    """HEAD 单个 URL；解析【最终】HTTP 响应块的状态码与 Content-Length，并记录 curl 退出码。"""
    for attempt in (1, 2):        # 单 URL 最多重试一次（官方源慢，已实测有瞬时超时）
        proc = subprocess.run(["curl.exe", "-sSL", "-I", "--max-time", "45", url],
                              capture_output=True, text=True)
        if proc.returncode == 0 and "HTTP/" in proc.stdout:
            break
    status, length = None, None
    for line in proc.stdout.splitlines():
        s = line.strip()
        if s.upper().startswith("HTTP/"):
            status, length = s, None          # 进入新的响应块：重置，避免中间代理头
        elif s.lower().startswith("content-length:"):
            try:
                length = int(s.split(":", 1)[1].strip())
            except ValueError:
                length = None
    return {"curl_exit": proc.returncode, "status_line": status, "content_length": length,
            "stderr": proc.stderr.strip()[:200]}


def precheck(rows: list[dict], meta: dict[str, dict]) -> dict:
    """批量预检：HEAD 全部 URL；任一异常或预算越界即判定失败（调用方据此非零退出并停止）。"""
    items, problems, total = [], [], 0
    ids = [r["isic_id"] for r in rows]
    if len(ids) != len(set(ids)):
        problems.append("存在重复 ISIC ID")
    for row in rows:
        i = row["isic_id"]
        info = meta.get(i) or {}
        url = info.get("full_url")
        if not url:
            problems.append(f"{i}: api_metadata 缺 full_url"); continue
        probe = head_probe(url)
        api_size = int(info.get("full_size") or 0)
        rec = {"isic_id": i, "url": url, "api_full_size": api_size, **probe}
        rec["delta"] = (None if probe["content_length"] is None
                        else probe["content_length"] - api_size)
        items.append(rec)
        if probe["curl_exit"] != 0:
            problems.append(f"{i}: curl exit={probe['curl_exit']} {probe['stderr']}")
        if not (probe["status_line"] or "").upper().startswith("HTTP/") or                 " 2" not in (probe["status_line"] or "")[:12]:
            problems.append(f"{i}: 非 2xx 最终响应 {probe['status_line']}")
        if not probe["content_length"] or probe["content_length"] <= 0:
            problems.append(f"{i}: Content-Length 缺失或非正 {probe['content_length']}")
        else:
            total += probe["content_length"]
    n = len(ids)
    est_raw = total
    est_norm = n * 250_000          # 实测规范化 PNG 约 0.16-0.45 MB/张，取 0.25 MB 保守估
    est_mask = n * 4_000
    est_misc = 2_000_000
    persistent = est_raw + est_norm + est_mask + est_misc
    peak = est_raw + est_norm + est_mask + est_misc + 5_000_000
    budget_ok = persistent < 250_000_000 and peak < 450_000_000
    if not budget_ok:
        problems.append(f"预算越界 persistent={persistent} peak={peak}")
    out = {"schema": "a16-n2-head-precheck-v1", "count": n, "unique_ids": len(set(ids)),
           "ok": not problems, "problems": problems,
           "total_content_length": total,
           "estimated_persistent_bytes": persistent, "estimated_peak_bytes": peak,
           "limits": {"persistent": 250_000_000, "peak": 450_000_000},
           "distinct_deltas": sorted({x["delta"] for x in items if x["delta"] is not None}),
           "items": items}
    (AUDIT/"head_precheck.json").write_text(json.dumps(out, ensure_ascii=False, indent=2)+"\n",
                                            encoding="utf-8")
    return out


def process(rows: list[dict], meta: dict[str, dict], zip_path: Path, *, out_dirs: dict,
            ledger: list, notes: list) -> list[dict]:
    results = []
    with zipfile.ZipFile(zip_path) as zf:
        for row in rows:
            i = row["isic_id"]
            info = meta.get(i)
            if not info or not info.get("full_url"):
                notes.append({"isic_id": i, "issue": "api_metadata 缺失 full_url"})
                continue
            expected = int(info["full_size"])
            raw = out_dirs["raw_images"]/f"{i}.jpg"
            probe = head_probe(info["full_url"])          # 权威长度（API size 已被证伪）
            if probe["curl_exit"] != 0 or probe["content_length"] is None or \
                    probe["content_length"] <= 0:
                raise RuntimeError(f"{i}: HEAD 失败 {probe}")
            real = probe["content_length"]
            if real is None:
                notes.append({"isic_id": i, "issue": "HEAD 未取到 Content-Length，隔离待审"})
                continue
            if real > MAX_BYTES_PER_IMAGE:
                notes.append({"isic_id": i, "issue": f"HEAD 长度 {real} B 超单图上限，隔离待审"})
                continue
            dl = download(info["full_url"], raw, real, ledger)
            with Image.open(raw) as img:
                img = img.convert("RGB")
                w, h = img.size
                fmt = img.format or "JPEG"
                aspect = (w/h) / (TARGET[0]/TARGET[1])
                aspect_diff = abs(aspect - 1.0)
            mask = mask_image(zf, i)
            mvals = sorted(set(np.unique(np.asarray(mask)).tolist()))
            rec = {"isic_id": i, "area_band": row["area_band"], "area_ratio": float(row["area_ratio"]),
                   "image_url": info["full_url"], "image_license": info["license"],
                   "image_attribution": info["attribution"],
                   "raw_bytes": dl["bytes"], "raw_sha256": dl["sha256"], "reused": dl["reused"],
                   "api_full_size": expected, "http_content_length": real,
                   "api_size_delta": (real - expected),
                   "image_size": [w, h], "image_format": fmt,
                   "mask_size": list(mask.size), "mask_values": mvals,
                   "aspect_diff": round(aspect_diff, 6),
                   "aspect_ok": aspect_diff <= ASPECT_TOL}
            if aspect_diff <= ASPECT_TOL:
                norm = out_dirs["images"]/f"{i}.png"
                img.resize(TARGET, Image.Resampling.LANCZOS).save(norm, format="PNG")
                with Image.open(norm) as chk:
                    rec["normalized_size"] = list(chk.size)
                    rec["normalized_bytes"] = norm.stat().st_size
                    rec["normalized_sha256"] = sha256_file(norm)
                rec["registration_ok"] = tuple(rec["normalized_size"]) == tuple(mask.size)
            else:
                iso = out_dirs["isolated"]/f"{i}.jpg"
                iso.parent.mkdir(parents=True, exist_ok=True)
                Image.open(raw).save(iso)
                rec["registration_ok"] = None
                notes.append({"isic_id": i, "issue": f"宽高比差 {aspect_diff:.4%} > 1%，已隔离待 Codex 看图"})
            mask_out = out_dirs["masks"]/f"{i}_UNET.png"
            mask.save(mask_out)
            rec["mask_bytes"] = mask_out.stat().st_size
            rec["mask_sha256"] = sha256_file(mask_out)
            rec["mask_attribution"] = "ViDIR Group, Department of Dermatology, Medical University of Vienna"
            results.append(rec)
    return results


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--precheck", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args(list(argv) if argv is not None else None)
    rows, meta, zip_path = load_inputs()
    if args.precheck:
        out = precheck(rows, meta)
        print(json.dumps({"ok": out["ok"], "count": out["count"], "unique_ids": out["unique_ids"],
                          "problems": out["problems"][:5], "problems_count": len(out["problems"]),
                          "total_content_length": out["total_content_length"],
                          "estimated_persistent_bytes": out["estimated_persistent_bytes"],
                          "estimated_peak_bytes": out["estimated_peak_bytes"],
                          "distinct_deltas": out["distinct_deltas"]}, ensure_ascii=False, indent=2))
        return 0 if out["ok"] else 3
    if args.smoke:
        tiny = [r for r in rows if r["area_band"] == "tiny"][:1]
        large = [r for r in rows if r["area_band"] == "large"][:1]
        rows = tiny + large
    elif not args.all:
        print("需要 --smoke N 或 --all"); return 2
    tag = "smoke" if args.smoke else "full"
    out_dirs = {k: (ROOT/k if k != "isolated" else ROOT/"isolated") for k in
                ("raw_images", "images", "masks", "isolated")}
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    ledger: list[dict] = []
    notes: list[dict] = []
    results = process(rows, meta, zip_path, out_dirs=out_dirs, ledger=ledger, notes=notes)
    (AUDIT/f"download_ledger_{tag}.json").write_text(
        json.dumps({"schema": "a16-n2-download-ledger-v1", "ids": [r["isic_id"] for r in rows],
                    "attempts": ledger, "notes": notes}, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    (AUDIT/f"pair_audit_{tag}.json").write_text(
        json.dumps({"schema": "a16-n2-pair-audit-v1", "count": len(results), "pairs": results,
                    "notes": notes}, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    ok = sum(1 for r in results if r.get("aspect_ok"))
    print(json.dumps({"tag": tag, "processed": len(results), "aspect_ok": ok,
                      "isolated": len(results)-ok, "notes": notes,
                      "total_raw_bytes": sum(r["raw_bytes"] for r in results)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())