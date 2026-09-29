"""Measure XRT whole-BO versus prefix synchronization; no DMA is started."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    from pynq import allocate
    import pyxrt
    buffer = allocate(shape=(131072,), dtype=np.uint8, cacheable=True)
    view = np.asarray(buffer)
    report = {'scope': 'cache sync only, no DMA or inference', 'passed': False,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'cases': []}
    try:
        if buffer.coherent or buffer.offset != 0:
            raise ValueError('requires a noncoherent root allocation')
        direction = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        for length in (256, 280, 1024, 8192, 32768, 65536, 131072):
            aligned = (length+63)//64*64
            samples = {'full': [], 'range': []}
            for repeat in range(8):
                methods = [('full', buffer.flush), ('range', lambda: buffer.bo.sync(direction, aligned, 0))]
                if repeat & 1:
                    methods.reverse()
                for name, method in methods:
                    elapsed = 0.0
                    for iteration in range(100):
                        view[:length] = iteration
                        start = time.perf_counter(); method(); elapsed += time.perf_counter()-start
                    if repeat >= 2:
                        samples[name].append(elapsed/100)
            record = {'length': length, 'aligned': aligned, 'samples': samples,
                      'median_seconds': {k: float(np.median(v)) for k,v in samples.items()}}
            report['cases'].append(record)
            print(json.dumps({k:v for k,v in record.items() if k!='samples'}), flush=True)
        report['passed'] = True
    finally:
        buffer.freebuffer()
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
