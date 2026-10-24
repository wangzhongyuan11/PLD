"""Frozen-model layer ledger. Estimates are explicitly NOT measured FPS."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
MODELS = {
    'skin': ('role_a_final/int8/model_spec.json', 'fe137ca5b128fc1983fabad9bee39b3208cda4256e0d04c4e27cef5c07666aef', 675282944),
    'camus': ('camus_candidate_hidden_histogram_int8_20261015/int8_export/model_spec.json', '68b4f425cc3612ad72d5e83955c2018981edee912fbc0937e9ac8c1718e7b5ae', 2656043008),
}

def main():
    report = {'kind': 'static_estimate_not_board_measurement', 'baseline_commit': '5a061c4c7c75049a815a87a82773bf1347119ab2',
              'architecture': {'spatial': 2, 'output_channels': 8, 'input_channels': 8, 'target_clock_hz': 125000000}, 'models': {}}
    for name, (path, sha, expected_macs) in MODELS.items():
        raw = (REPO / path).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == sha, f'{name} model changed'
        spec = json.loads(raw)
        rows = []
        for i, layer in enumerate(spec['layers']):
            co, ci, kh, kw = layer['weight_shape']
            _, _, h, w = layer.get('output_shape', layer.get('full_output_shape'))
            macs = h*w*co*ci*kh*kw
            # Channels padded per kernel position. Adjacent output positions share weights.
            dots = h*math.ceil(w/2)*math.ceil(co/8)
            beats = kh*kw*math.ceil(ci/8)
            issue_cycles = dots*beats
            rows.append({'index': i, 'name': layer['name'], 'ci': ci, 'co': co, 'h': h, 'w': w, 'kernel': kh,
                         'macs': macs, 'dot_groups': dots, 'beats_per_group': beats, 'issue_cycles': issue_cycles,
                         'lane_utilization': macs/(128*issue_cycles), 'weight_bytes': co*ci*kh*kw,
                         'output_bytes': co*h*w})
        total_macs = sum(r['macs'] for r in rows)
        assert total_macs == expected_macs
        cycles = sum(r['issue_cycles'] for r in rows)
        report['models'][name] = {'spec_sha256': sha, 'layers': rows, 'macs': total_macs, 'issue_cycles': cycles,
            'ideal_issue_seconds_125mhz': cycles/125000000,
            'issue_seconds_at_70_percent_feeder_duty': cycles/125000000/.7,
            'remaining_125ms_budget_at_70_percent_duty': .125-cycles/125000000/.7,
            'exclusions': ['memory stalls', 'DMA', 'layer transitions', 'pool/LUT mapping', 'requantization stalls', 'PS/web'],
            'skip_bytes': sum(rows[i]['output_bytes'] for i in (1,3,5))}
    (ROOT/'evidence').mkdir(exist_ok=True)
    (ROOT/'evidence/layer_budget.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    for name, data in report['models'].items():
        print(name, {k:v for k,v in data.items() if k not in ('layers','exclusions')})

if __name__ == '__main__': main()
