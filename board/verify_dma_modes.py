"""Verify new DMA APIs on the retained +1 overlay; no CNN claim."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from verify_overlay import Loopback, SIZES, sha256


class APIAdapter:
    def __init__(self, dma, mode):
        self.dma, self.mode = dma, mode

    def __getattr__(self, name):
        return getattr(self.dma, name)

    def transfer(self, source, destination, nbytes, *, receive_only=False):
        if receive_only:
            raise ValueError("fault injection uses the original loopback verifier")
        if self.mode == "split":
            # One AXIS beat fits in incr's output register while S2MM is idle.
            # Longer TX-first packets would correctly stall and time out.
            if nbytes > 8:
                raise ValueError("split test exceeds one-beat PL storage")
            self.dma.send(source,nbytes)
            self.dma.receive(destination,nbytes)
        else:
            self.dma.submit(source,destination,send_nbytes=nbytes,receive_nbytes=nbytes)
            self.dma.wait()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bit",required=True)
    parser.add_argument("--report",required=True)
    args=parser.parse_args()
    bit=Path(args.bit).resolve()
    report={"passed":False,"packets":0,"scope":"incr split send/receive and submit/wait; unequal lengths require compute overlay"}
    loop=None
    try:
        report["sha256"]={p.name:sha256(p) for p in (bit,bit.with_suffix(".hwh"),Path(__file__),
                           Path(__file__).with_name("simple_dma.py"),Path(__file__).with_name("verify_overlay.py"))}
        loop=Loopback(bit,"incr",2.0)
        engine=loop.dma
        rng=np.random.default_rng(20260926)
        for mode in ("split","submit"):
            loop.dma=APIAdapter(engine,mode)
            for _ in range(16 if mode=="split" else 1):
                for n in ((1,7,8) if mode=="split" else SIZES):
                    loop.packet(rng.integers(0,256,n,dtype=np.uint8))
                    report["packets"]+=1
            print(f"DMA_MODE_PASS {mode}",flush=True)
        loop.dma=engine
        report["passed"]=True
        print(f"DMA_MODES_PASS packets={report['packets']}",flush=True)
        return 0
    except Exception as exc:
        report["error"]=str(exc)
        print(f"DMA_MODES_FAIL {exc}",file=sys.stderr,flush=True)
        if loop is not None:
            try:
                loop.dma.reset()
                loop.reload()
                report["cleanup_recovered"]=True
            except Exception as recovery:
                report["cleanup_error"]=str(recovery)
                return 2
        return 1
    finally:
        Path(args.report).write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    raise SystemExit(main())
