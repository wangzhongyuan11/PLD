"""Physical-capacity tile planner for the experimental spatial compute engine.

This is an offline schedule, not a board performance claim. Output subtiles fit
the existing two 1024-word Co8 output slots; wider streaming strips come later.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math

from acdc.contract import SHAPES
from .layout import BankGeometry, LAYOUT


@dataclass(frozen=True)
class Profile:
    name: str
    p: int
    input_words: int
    q24_lanes: int = 4
    output_words: int = 1024
    weight_words: int = 512

    @property
    def input_bram36(self):
        return 2*self.p*math.ceil(self.input_words/512)


PROFILES = {'P256': Profile('P256', 4, 4608),
            'P384': Profile('P384', 6, 3584)}


def tile_traffic(layer, y, x, th, tw):
    """Payload bytes, including C8 padding but excluding AXI protocol overhead."""
    ci, co, h, k = SHAPES[layer]
    xs = range(max(0, x-k//2), min(h, x+tw+k//2))
    ys = range(max(0, y-k//2), min(h, y+th+k//2))
    if layer in (8, 10, 12):
        split = {8: 64, 10: 32, 12: 16}[layer]
        source = len(xs)*len(ys)*split
        source += len({v//2 for v in xs})*len({v//2 for v in ys})*(ci-split)
    else:
        source = len(xs)*len(ys)*math.ceil(ci/8)*8
    output = math.ceil(co/8)*8*th*tw
    pool = co*(th//2)*(tw//2) if layer in (1, 3, 5) else 0
    # Input-stationary, one complete padded layer's weights fetched per tile.
    weights = math.ceil(co/8)*math.ceil(ci/8)*64*k*k
    parameters = math.ceil(co/8)*8*8  # bias/M INT32
    return dict(input=source, output=output, pool=pool,
                weights=weights, parameters=parameters)


def layer_plan(index, profile, bandwidth=300e6, mhz=100, efficiency=.8):
    if not (bandwidth > 0 and mhz > 0 and 0 < efficiency <= 1):
        raise ValueError('invalid prediction assumptions')
    ci, co, h, k = SHAPES[index]
    candidates = []
    heights = sorted(set([2, 4, 8, 12, 16, 24, 32, 48, 64, h]))
    widths = sorted(set([profile.p, profile.p*2, profile.p*4,
                         profile.p*6, profile.p*8, 64, 128, h]))
    reduction = math.ceil(ci/8)*k*k
    for th in heights:
        for tw in widths:
            if th > h or tw > h or th*tw > profile.output_words:
                continue
            if index in (1, 3, 5) and (th % 2 or tw % 2):
                continue
            try:
                geometry = BankGeometry(profile.p, ci, th+k-1, tw+k-1,
                                        profile.input_words)
            except ValueError:
                continue
            if reduction > profile.weight_words:
                continue
            tiles, cycle_sum = [], 0
            byte_sum = {name: 0 for name in ('input','output','pool','weights','parameters')}
            for y in range(0, h, th):
                for x in range(0, h, tw):
                    oh, ow = min(th, h-y), min(tw, h-x)
                    groups = oh*math.ceil(ow/profile.p)*math.ceil(co/8)
                    cycles = groups*max(reduction, math.ceil(profile.p*8/profile.q24_lanes))
                    traffic = tile_traffic(index, y, x, oh, ow)
                    cycle_sum += cycles
                    for name, value in traffic.items():
                        byte_sum[name] += value
                    tiles.append(dict(y=y, x=x, height=oh, width=ow,
                                      service_cycles=cycles, bytes=sum(traffic.values())))
            seconds = max(cycle_sum/(mhz*1e6*efficiency),sum(byte_sum.values())/bandwidth)
            # No arbitrary tile-cycle penalty hidden here: DMA/setup must be
            # measured and added by the event-level scheduler before acceptance.
            candidates.append((seconds, len(tiles), sum(byte_sum.values()),
                               dict(layer=f'C{index+1:02}', ci=ci, co=co, kernel=k,
                                    height=h, width=h, tile_hw=[th,tw],
                                    input_words_per_bank=geometry.used_words,
                                    service_cycles=cycle_sum, payload_bytes=byte_sum,
                                    estimated_seconds=seconds, tiles=tiles)))
    if not candidates:
        raise ValueError(f'no physical tile fits layer {index+1}')
    return min(candidates, key=lambda row: row[:3])[3]


def plan(name='P256', bandwidth=300e6, mhz=100, efficiency=.8):
    if name not in PROFILES:
        raise ValueError('unknown hardware profile')
    profile = PROFILES[name]
    layers = [layer_plan(i,profile,bandwidth,mhz,efficiency) for i in range(15)]
    parameter_bytes = sum(math.ceil(co/8)*8*8+math.ceil(co/8)*math.ceil(ci/8)*64*k*k
                          for ci,co,h,k in SHAPES)
    return dict(schema='pld-acdc-spatial-plan-v1', layout=LAYOUT,
                profile=asdict(profile), board_measured=False,
                assumption=dict(mhz=mhz, aggregate_payload_bytes_per_second=bandwidth,
                                compute_service_efficiency=efficiency),
                resources=dict(main_dsp=profile.p*32, q24_dsp=16,
                               input_bram36=profile.input_bram36, weight_bram36=16,
                               output_bram36=4),
                model_storage_bytes=parameter_bytes,
                service_cycles=sum(row['service_cycles'] for row in layers),
                payload_bytes=sum(sum(row['payload_bytes'].values()) for row in layers),
                estimated_seconds=sum(row['estimated_seconds'] for row in layers),
                excluded=['DMA command/burst fragmentation and response latency',
                          'pipeline boundaries and tile switching', 'pool/CAT extra stalls',
                          'PS layout/copy/cache/submit/wait overhead', 'network and PC processing'],
                layers=layers)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=tuple(PROFILES),default='P256')
    parser.add_argument('--output',type=str)
    args=parser.parse_args()
    result=plan(args.profile)
    text=json.dumps(result,indent=2)+'\n'
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(text,encoding='utf-8')
    else:
        print(text,end='')
