"""Run isolated xsim tests; persist source hashes and tool logs on desktop."""
import hashlib
import json
from pathlib import Path
import subprocess
import time
import argparse

ROOT = Path(__file__).resolve().parents[1]
VIVADO = Path(r"C:\AMDDesignTools\2026.1\Vivado\bin")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('unit', nargs='?', default='board_top', choices=['dot128', 'feature_banks', 'requant24', 'conv_pair_tile', 'model_tiles', 'ddr_read64', 'ddr_write64', 'map8', 'stripe_load', 'pair_to_words', 'stripe_engine', 'pool2_rows', 'nearest2_row', 'row_transform', 'tensor_load', 'transformed_engine', 'graph_engine', 'frame_engine', 'graph_control', 'frame_registers', 'board_top'])
    unit = parser.parse_args().unit
    stamp = time.strftime('%Y%m%d_%H%M%S')
    out = ROOT / 'out' / ('sim_' + unit + '_' + stamp)
    out.mkdir(parents=True, exist_ok=False)
    sources = [ROOT / f'rtl/{unit}.sv', ROOT / f'sim/tb_{unit}.sv']
    if unit == 'model_tiles':
        sources = [ROOT / 'rtl/conv_pair_tile.sv', ROOT / 'rtl/requant24.sv', ROOT / 'rtl/quant_pair.sv', ROOT / 'rtl/tile_int8.sv', ROOT / 'sim/tb_model_tiles.sv']
    if unit in ('conv_pair_tile', 'model_tiles'):
        sources = [ROOT / 'rtl/dot128.sv', ROOT / 'rtl/feature_banks.sv'] + sources
    if unit == 'stripe_load':
        sources = [ROOT/'rtl/map8.sv', ROOT/'rtl/ddr_read64.sv'] + sources
    if unit == 'nearest2_row':
        sources = [ROOT/'rtl/pair_to_words.sv'] + sources
    if unit == 'row_transform':
        sources = [ROOT/'rtl'/name for name in ('pair_to_words.sv','nearest2_row.sv','pool2_rows.sv','map8.sv')] + sources
    if unit == 'tensor_load':
        sources = [ROOT/'rtl'/name for name in ('pair_to_words.sv','nearest2_row.sv','pool2_rows.sv','map8.sv','row_transform.sv','ddr_read64.sv')] + sources
    if unit in ('stripe_engine', 'transformed_engine', 'graph_engine', 'frame_engine', 'board_top'):
        sources = sorted((ROOT/'rtl').glob('*.sv')) + [ROOT/f'sim/tb_{unit}.sv']
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    if unit in ('model_tiles', 'stripe_engine'):
        vector_name='cases.txt' if unit=='model_tiles' else 'engine_cases.txt'
        for p in [ROOT/'out/model_vectors'/vector_name,ROOT/'tools/model_vectors.py',ROOT/'evidence/model_vectors.json']:
            hashes[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    if unit == 'row_transform':
        for p in [ROOT/'out/model_vectors/transitions.txt',ROOT/'tools/transition_vectors.py',ROOT/'evidence/transition_vectors.json']:
            hashes[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    if unit in ('transformed_engine','board_top'):
        for p in [ROOT/'out/model_vectors/transformed_cases.txt', ROOT/'tools/transformed_vectors.py', ROOT/'evidence/transformed_vectors.json']:
            hashes[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    if unit in ('graph_engine','frame_engine'):
        for p in [ROOT/'tools/graph_vectors.py', ROOT/'evidence/graph_vectors.json',ROOT/'tools/pack_frame.py',ROOT/'evidence/packed_frame.json'] + list((ROOT/'out/model_vectors').glob('graph_*/*')):
            hashes[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    commands = [
        [str(VIVADO / 'xvlog.bat'), '--sv', *map(str, sources)],
        [str(VIVADO / 'xelab.bat'), 'tb_' + unit, '-s', 'skin8_' + unit, '--debug', 'typical'],
        [str(VIVADO / 'xsim.bat'), 'skin8_' + unit, '-runall'],
    ]
    result = {'sources': hashes, 'timestamp': stamp, 'passed': False, 'output_dir': str(out), 'commands': []}
    try:
        for i, command in enumerate(commands):
            p = subprocess.run(command, cwd=out, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=1800 if unit in ('graph_engine','frame_engine') else 180, text=True)
            (out / f'{i}.log').write_text(p.stdout, encoding='utf-8')
            result['commands'].append({'argv': command, 'returncode': p.returncode})
            print(p.stdout[-6000:], flush=True)
            if p.returncode: raise RuntimeError(f'command {i} failed')
        if f'PASS {unit} checked=' not in p.stdout or 'Fatal:' in p.stdout:
            raise RuntimeError('missing successful scoreboard result')
        result['passed'] = True
    finally:
        (ROOT / 'evidence').mkdir(exist_ok=True)
        (ROOT / 'evidence' / f'{unit}_{stamp}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))

if __name__ == '__main__': main()
