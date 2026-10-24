"""Package only a timing-qualified build paired with matching full-frame RTL evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from budget import ROOT


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--build',type=Path,required=True)
    parser.add_argument('--simulation',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    build=json.loads(args.build.read_text());sim=json.loads(args.simulation.read_text())
    if not build.get('passed') or not sim.get('passed'):raise ValueError('build and simulation must pass')
    if not any('tb_frame_engine.sv' in k for k in sim['sources']):raise ValueError('requires single-submit full-frame evidence')
    for name,digest in build['sources'].items():
        path=Path(name)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('source changed after build: '+name)
        if path.suffix=='.sv' and sim['sources'].get(str(path.relative_to(ROOT)))!=digest:
            raise ValueError('simulation/build source mismatch: '+name)
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    files={}
    def copy(source,destination):
        target=out/destination;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target);files[destination]=hashlib.sha256(target.read_bytes()).hexdigest()
    for name in ('skin8.bit','skin8.hwh'):
        source=Path(build['output_dir'])/name
        if hashlib.sha256(source.read_bytes()).hexdigest()!=build['artifacts'][name]:raise ValueError('artifact drift')
        copy(source,name)
    copy(args.build,'build_receipt.json');copy(args.simulation,'simulation_receipt.json')
    descriptor_receipt=json.loads((ROOT/'evidence/descriptors.json').read_text())
    for model in ('skin','camus'):
        base=ROOT/'out/compiled_models'/model
        manifest=json.loads((base/'manifest.json').read_text())
        for filename,digest in [('manifest.json',descriptor_receipt['models'][model]['manifest_sha256']),
                                ('descriptors.json',descriptor_receipt['models'][model]['descriptors_sha256'])]:
            if hashlib.sha256((base/filename).read_bytes()).hexdigest()!=digest:raise ValueError('compiled metadata drift')
            copy(base/filename,'models/'+model+'/'+filename)
        for filename,record in manifest['artifacts'].items():
            if hashlib.sha256((base/filename).read_bytes()).hexdigest()!=record['sha256']:raise ValueError('model artifact drift')
            copy(base/filename,'models/'+model+'/'+filename)
    for name in ('board_image.py','frame_driver.py','verify_board.py'):copy(ROOT/'tools'/name,name)
    (out/'package.json').write_text(json.dumps({'kind':'isolated_board_candidate_not_web_deployment',
        'files':files,'scope':'full-frame local RTL verified; hardware correctness/performance not yet verified'},indent=2))
    print(out)


if __name__=='__main__':main()
