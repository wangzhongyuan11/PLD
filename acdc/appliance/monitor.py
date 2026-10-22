"""Bounded, read-only Linux resource sampling for a standalone board run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time


def numbers(path, wanted):
    result = {}
    try:
        for line in Path(path).read_text().splitlines():
            key, _, value = line.partition(':')
            if key in wanted:
                result[key] = int(value.strip().split()[0])
    except (FileNotFoundError, ProcessLookupError):
        result['exited'] = True
    return result


def sample(pids):
    row = dict(monotonic=time.monotonic(), memory_kib=numbers('/proc/meminfo', {
        'MemAvailable', 'MemFree', 'SwapFree', 'CmaTotal', 'CmaFree'}), processes={})
    for pid in pids:
        proc = Path('/proc') / str(pid)
        row['processes'][str(pid)] = dict(
            status_kib=numbers(proc/'status', {'VmRSS', 'VmHWM', 'VmSwap', 'Threads'}),
            smaps_kib=numbers(proc/'smaps_rollup', {'Rss', 'Pss', 'Swap'}),
            io_bytes=numbers(proc/'io', {'read_bytes', 'write_bytes'}))
    vm = {}
    for line in Path('/proc/vmstat').read_text().splitlines():
        key, value = line.split()
        if key in ('pswpin', 'pswpout', 'oom_kill'):
            vm[key] = int(value)
    row['vmstat'] = vm
    row['cpu'] = Path('/proc/stat').read_text().splitlines()[0]
    # Some PYNQ kernels omit per-process /proc/<pid>/io. Preserve the global
    # physical SD counters separately, without inventing process I/O evidence.
    row['sd_diskstats'] = [line.split() for line in Path('/proc/diskstats').read_text().splitlines()
                           if line.split()[2] == 'mmcblk0']
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pid', type=int, action='append', required=True)
    parser.add_argument('--seconds', type=int, default=2400)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 7200 or any(pid <= 1 for pid in args.pid):
        parser.error('bounded duration and explicit service PIDs required')
    end = time.monotonic() + args.seconds
    with args.output.open('x', buffering=1) as stream:
        while time.monotonic() < end:
            tick = time.monotonic()
            stream.write(json.dumps(sample(args.pid), separators=(',', ':'))+'\n')
            time.sleep(max(0, 1-(time.monotonic()-tick)))


if __name__ == '__main__':
    main()
