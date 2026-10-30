# -*- coding: utf-8 -*-
"""Seed cloud-development demo data for the mini program.

Generates two deterministic synthetic ACDC ED/ES cases (no board, no real
patient data), uploads them to cloud storage and upserts the `demos` and
`config` collection documents.

Usage (from bridge/, with config.json filled in):
    D:/conda/python.exe tools/upload_demo_data.py [--out out_demo] [--local-only]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wechat_api import WechatCloud, WechatApiError  # noqa: E402
from nifti import write_nii_gz, cardiac_phantom  # noqa: E402

CASES = [
    {"id": "acdc_demo_1", "title": "合成演示病例 1（64×64×16）",
     "size": (64, 64, 16), "es_scale": 0.62, "seed": 7},
    {"id": "acdc_demo_2", "title": "合成演示病例 2（84×84×18）",
     "size": (84, 84, 18), "es_scale": 0.55, "seed": 23},
]


def upsert_doc(cloud, collection, doc_id, fields):
    updated = cloud.update_doc(collection, doc_id, fields)
    if not updated:
        doc = dict(fields)
        doc["_id"] = doc_id
        try:
            cloud.add_doc(collection, doc)
        except WechatApiError as exc:
            if "exist" not in str(exc) and "DUPLICATE" not in str(exc).upper():
                raise
    print("upsert %s/%s ok" % (collection, doc_id))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config.json"))
    parser.add_argument("--out", default=str(ROOT / "out_demo"))
    parser.add_argument("--local-only", action="store_true",
                        help="只生成 ED/ES 文件不上传（离线演练/校验）")
    args = parser.parse_args(argv)

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for case in CASES:
        files = []
        for phase, scale in (("ED", 1.0), ("ES", case["es_scale"])):
            local = out_dir / ("%s_%s.nii.gz" % (case["id"], phase))
            write_nii_gz(local, cardiac_phantom(case["size"], scale,
                                                case["seed"]),
                         spacing=(2.0, 2.0, 2.0))
            files.append({"name": "%s.nii.gz" % phase,
                          "local": str(local)})
            print("gen %s (%d bytes)" % (local, local.stat().st_size))
        if args.local_only:
            continue

    if args.local_only:
        print("GEN_ONLY_DONE")
        return 0

    cloud = WechatCloud(cfg["app_id"], cfg["app_secret"], cfg["env_id"])
    for case in CASES:
        files = []
        for phase in ("ED", "ES"):
            local = out_dir / ("%s_%s.nii.gz" % (case["id"], phase))
            # v2：2026-10-25 修复上传污染后重传；COS 同路径覆盖读回可能短暂返回旧
            # 数据，改用新路径保证读到的一定是干净字节。
            cloud_path = "demos/acdc/v2/%s/%s.nii.gz" % (case["id"], phase)
            file_id = cloud.upload_bytes(cloud_path, local.read_bytes())
            files.append({"fileID": file_id, "name": "%s.nii.gz" % phase})
            print("upload %s -> %s" % (cloud_path, file_id))
        upsert_doc(cloud, "demos", case["id"], {
            "model": "acdc",
            "title": case["title"],
            "desc": "内置合成数据，无需上传文件",
            "inputs": files,
            "params": {"ed_name": "ED.nii.gz", "es_name": "ES.nii.gz"},
            "updated_at": 0,
        })

    print("DEMO_SEED_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
