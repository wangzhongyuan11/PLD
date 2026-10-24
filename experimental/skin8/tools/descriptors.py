"""Expand checked software descriptors for the future PL sequencer.

JSON is an executable specification for tests, not a supported hardware ABI.
Addresses are arena-relative; caller must validate physical allocation bounds.
"""
import hashlib
import json
from budget import ROOT


def main():
    schedule_path = ROOT/'evidence/stripe_schedule.json'
    schedules = json.loads(schedule_path.read_text())
    receipt = {'kind': 'descriptor_plan_not_running_controller',
               'schedule_sha256': hashlib.sha256(schedule_path.read_bytes()).hexdigest(), 'models': {}}
    for name, schedule in schedules['models'].items():
        manifest_path = ROOT/'out/compiled_models'/name/'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        assert manifest['spec_sha256'] == schedule['spec_sha256']
        plans = schedule['layers']
        descriptors = []
        epoch = 0
        for layer, params in zip(plans, manifest['layers']):
            li = layer['index']
            assert li == params['index']
            groups = params['input_groups']
            skip = layer['skip_source_layer']
            previous = plans[li-1]['output_offset'] if li else 0
            for stripe in layer['stripes']:
                epoch += 1
                branches = []
                def branch(source, count, offset, mode, table):
                    branches.append({'source_offset': source, 'groups': count, 'group_offset': offset,
                                     'mode': mode, 'table': table, 'clear': len(branches) == 0})
                if skip is None:
                    branch(previous, groups, 0, 1 if layer['mode'] == 'maxpool2' else 0, 0)
                else:
                    skip_groups = (plans[skip]['co']+7)//8
                    branch(plans[skip]['output_offset'], skip_groups, 0, 0, 0)
                    branch(previous, groups-skip_groups, skip_groups, 2, 1)
                assert sum(b['groups'] for b in branches) == groups
                for b in branches:
                    sw = layer['w']*(2 if b['mode'] == 1 else 1)
                    sh = layer['h']*(2 if b['mode'] == 1 else 1)
                    if b['mode'] == 2:
                        sw //= 2
                        sh //= 2
                    end = b['source_offset']+sw*sh*b['groups']*8
                    assert end <= layer['output_offset']
                    assert b['source_offset'] % 8 == 0
                for output_group in range(params['output_groups']):
                    destination = layer['output_offset']+8*((output_group*layer['h']+stripe['y'])*layer['w'])
                    end = destination+8*stripe['height']*layer['w']
                    assert end <= layer['output_offset']+layer['output_bytes']
                    weight_offset = output_group*params['weight_block_bytes']
                    parameter_offset = output_group*64
                    assert weight_offset+params['weight_block_bytes'] <= manifest['artifacts'][f'{li:02d}_weights.bin']['bytes']
                    assert parameter_offset+64 <= manifest['artifacts'][f'{li:02d}_params.bin']['bytes']
                    descriptors.append({'layer': li, 'stripe_epoch_within_frame': epoch,
                        'width': layer['w'], 'height': layer['h'], 'y': stripe['y'], 'rows': stripe['height'],
                        'groups': groups, 'kernel': layer['kernel'], 'relu': layer['relu_in_quantizer'],
                        'output_group': output_group, 'reuse_features': output_group != 0,
                        'branches': branches, 'destination_offset': destination,
                        'weights_file': f'{li:02d}_weights.bin', 'weights_offset': weight_offset,
                        'params_file': f'{li:02d}_params.bin', 'params_offset': parameter_offset,
                        'lut_file': f'{li:02d}_luts.bin', 'load_luts': stripe['y'] == 0 and output_group == 0})
        assert len(descriptors) == schedule['totals']['conv_tile_starts']
        out = ROOT/'out/compiled_models'/name/'descriptors.json'
        data = json.dumps({'kind': 'planned_interface_not_hardware_abi',
            'epoch_contract': 'combine frame epoch with stripe epoch; do not reuse across frames',
            'arena_bytes': schedule['arena_bytes_without_reuse'], 'descriptors': descriptors}, indent=2).encode()
        out.write_bytes(data)
        receipt['models'][name] = {'descriptors': len(descriptors), 'stripes': epoch,
            'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            'descriptors_sha256': hashlib.sha256(data).hexdigest(), 'path': str(out)}
        print(name, len(descriptors), 'descriptors checked', flush=True)
    receipt['tool_sha256'] = hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest()
    (ROOT/'evidence/descriptors.json').write_text(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
