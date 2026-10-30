"""Prepare CAMUS frames or ACDC ED/ES cases on the PC for direct board inference."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['camus','acdc'])
    p.add_argument('sources',type=Path,nargs='+',help='CAMUS files, or ACDC patient directories containing Info.cfg')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--acdc-manifest',type=Path,help='Frozen ACDC model MANIFEST.json, for source geometry identity')
    args=p.parse_args()
    if len(args.sources)>5:p.error('prepare at most five cases per invocation')
    args.output.mkdir(parents=True,exist_ok=True);rows=[]
    if args.mode=='acdc':
        if args.acdc_manifest is None:p.error('--acdc-manifest is required for ACDC geometry')
        manifest=json.loads(args.acdc_manifest.read_text('utf-8'))
        binding=dict(recognized_acdc_images={r['sha256']:r['path'] for r in manifest['files']
            if r['path'].startswith('datasets/acdc/training/') and r['path'].endswith('.nii.gz') and not r['path'].endswith('_gt.nii.gz')})
        from acdc.appliance.native import load_native
        from acdc.appliance.prepared import prepare
        kernel=load_native()
    else:
        from camus.prepared import prepare
        from camus.native import load_native
        load_native()  # Initialize/build once, outside per-case preprocessing timing.
    for source in args.sources:
        began=time.perf_counter();outputs=[]
        if args.mode=='camus':
            name=source.name.removesuffix('.gz').removesuffix('.nii')
            destination=args.output/(name+'.pldc')
            meta=prepare(source,destination)
            outputs.append(dict(source=str(source.resolve()),path=str(destination.resolve()),input_shape=meta['input_shape']))
        else:
            info=dict(line.split(':',1) for line in (source/'Info.cfg').read_text().splitlines() if ':' in line)
            for phase in ('ED','ES'):
                path=source/(source.name+f'_frame{int(info[phase]):02d}.nii.gz')
                destination=args.output/(source.name+'_'+phase+'.plda')
                meta=prepare(path,destination,kernel,binding=binding)
                outputs.append(dict(source=str(path.resolve()),path=str(destination.resolve()),input_shape=meta['input_shape']))
        elapsed=time.perf_counter()-began
        row=dict(case=source.name,pc_preprocess_compress_seconds=elapsed,outputs=outputs,
            source_bytes=sum(Path(item['source']).stat().st_size for item in outputs),
            prepared_bytes=sum(Path(item['path']).stat().st_size for item in outputs))
        rows.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
    (args.output/'preparation.json').write_text(json.dumps(dict(mode=args.mode,rows=rows),ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
