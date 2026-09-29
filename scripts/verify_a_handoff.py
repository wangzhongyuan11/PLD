#!/usr/bin/env python
"""对 A 交付的 INT8 golden 做逐位复算（闭环验证）。

用法: python scripts/verify_a_handoff.py [handoff根目录]
默认目录 D:/PLD/role_a_final。
检查内容：
  1) best.pt / weights_int8.bin / bias.bin 与 model_spec.json 的 SHA256 一致
  2) 15 层 8x8 小块：int32 卷积累加 vs acc_golden.bin（0 容差）
  3) Q24 ties-even requant vs output_golden.bin（0 容差）
  4) 逐层 weight.bin/bias.bin 与拼接 blob 的偏移切片一致
退出码 0=全部通过，1=有失败。
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "role_a_final")


def conv_requant(x: np.ndarray, w: np.ndarray, b: np.ndarray, mult_q24, pad: int):
    """int32 卷积累加 + Q24 ties-even requant，返回 (acc_int32, out_int8)。"""
    _, ci, h, wid = x.shape
    kh, kw = w.shape[2], w.shape[3]
    co = w.shape[0]
    ho, wo = h + 2 * pad - kh + 1, wid + 2 * pad - kw + 1
    xp = np.zeros((ci, h + 2 * pad, wid + 2 * pad), dtype=np.int64)
    xp[:, pad : pad + h, pad : pad + wid] = x[0]
    acc = np.zeros((co, ho, wo), dtype=np.int64)
    for o in range(co):
        acc[o] += b[o]
        for i in range(ci):
            for y in range(kh):
                for z in range(kw):
                    acc[o] += xp[i, y : y + ho, z : z + wo] * int(w[o, i, y, z])
    mult = np.asarray(mult_q24, dtype=np.int64)[:, None, None]
    nq = acc * mult
    q = np.floor_divide(nq, 1 << 24)
    r = nq - q * (1 << 24)
    q += ((2 * r > (1 << 24)) | ((2 * r == (1 << 24)) & ((q & 1) == 1)))
    out = np.clip(q, -127, 127).astype(np.int8)
    return acc.astype(np.int32), out


def main() -> int:
    spec = json.loads((ROOT / "int8" / "model_spec.json").read_text(encoding="utf-8"))
    ok = True

    for file, want in [
        (ROOT / "best.pt", spec["checkpoint_sha256"]),
        (ROOT / "int8" / "weights_int8.bin", spec["weights_int8_sha256"]),
        (ROOT / "int8" / "bias.bin", spec["bias_int32_sha256"]),
    ]:
        got = hashlib.sha256(file.read_bytes()).hexdigest()
        good = got == want
        ok &= good
        print(f"[{'PASS' if good else 'FAIL'}] sha256 {file.name} {got[:16]}...")

    wblob = (ROOT / "int8" / "weights_int8.bin").read_bytes()
    bblob = (ROOT / "int8" / "bias.bin").read_bytes()

    for lay in spec["layers"]:
        d = ROOT / "int8" / lay["directory"]
        cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
        x = np.frombuffer((d / "input.bin").read_bytes(), dtype="<i1").reshape(cfg["input_shape"])
        w = np.frombuffer((d / "weight.bin").read_bytes(), dtype="<i1").reshape(cfg["weight_shape"])
        b = np.frombuffer((d / "bias.bin").read_bytes(), dtype="<i4").reshape(w.shape[0])
        acc, out = conv_requant(x, w, b, cfg["multiplier_q24"], cfg["padding"])

        acc_g = np.frombuffer((d / "acc_golden.bin").read_bytes(), dtype="<i4").reshape(cfg["output_shape"])
        out_g = np.frombuffer((d / "output_golden.bin").read_bytes(), dtype="<i1").reshape(cfg["output_shape"])
        acc_ok = np.array_equal(acc, acc_g[0])
        out_ok = np.array_equal(out, out_g[0])
        wsl = wblob[lay["weight_offset_bytes"] : lay["weight_offset_bytes"] + lay["weight_nbytes"]]
        bsl = bblob[lay["bias_offset_bytes"] : lay["bias_offset_bytes"] + lay["bias_nbytes"]]
        sl_ok = wsl == (d / "weight.bin").read_bytes() and bsl == (d / "bias.bin").read_bytes()
        good = acc_ok and out_ok and sl_ok
        ok &= good
        msg = f"[{'PASS' if good else 'FAIL'}] {lay['directory']:<18} acc={acc_ok} req={out_ok} slice={sl_ok}"
        if not acc_ok:
            msg += f"  first_acc_mismatch@{np.argwhere(acc != acc_g[0])[:1].tolist()}"
        if not out_ok:
            msg += f"  first_out_mismatch@{np.argwhere(out != out_g[0])[:1].tolist()}"
        print(msg)

    print("ALL PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
