"""Reuse a source-bound offline plan at the application boundary."""
import copy
import json
from pathlib import Path

from acdc.contract import ContractError, file_sha


def generate(root):
    """Prepare the unchanged planner's result when assembling a release, off the request path."""
    from acdc import contract
    from acdc.optimization import layout,planner,program
    root=Path(root)
    build=json.loads((root/'overlay/build_manifest.json').read_text())
    name='P384' if build['spatial_p']==6 else 'P256'
    record=dict(schema='pld-acdc-source-bound-plan-v1',
        sources={module.__name__:file_sha(Path(module.__file__)) for module in (contract,layout,planner,program)},
        plans={name:planner.plan(name)})
    path=root/'v2_assets/spatial_plan_cache.json'
    path.parent.mkdir(exist_ok=True)
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(record,separators=(',',':')))
    temporary.replace(path)
    return path


def install_plan_cache(root):
    from acdc import contract
    from acdc.optimization import layout, planner, program
    path = Path(root) / 'v2_assets' / 'spatial_plan_cache.json'
    if not path.exists():
        return  # Existing releases keep their original planner.
    record = json.loads(path.read_text('utf-8'))
    if record.get('schema') != 'pld-acdc-source-bound-plan-v1':
        raise ContractError('invalid cached spatial plan')
    for module in (contract, layout, planner, program):
        if record['sources'].get(module.__name__) != file_sha(Path(module.__file__)):
            raise ContractError('cached spatial plan source drift: '+module.__name__)
    original = program.plan
    plans = record['plans']

    def plan(name='P256', bandwidth=300e6, mhz=100, efficiency=.8):
        # The frozen compiler requests the default offline assumptions.
        if bandwidth == 300e6 and mhz == 100 and efficiency == .8 and name in plans:
            return copy.deepcopy(plans[name])
        return original(name, bandwidth, mhz, efficiency)

    program.plan = plan


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Prepare an ACDC plan before creating its application certificate')
    parser.add_argument('root',type=Path)
    print(generate(parser.parse_args().root))
