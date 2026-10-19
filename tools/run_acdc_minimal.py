"""Run the released ACDC demo; no training or physical-board access."""
from pathlib import Path
import argparse,json,os,sys

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent,help='extracted minimal package root')
    parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--backend',choices=('fp32','int8'),default='fp32')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--ed',type=Path);parser.add_argument('--es',type=Path)
    args=parser.parse_args();root=args.root.resolve()
    if bool(args.ed)!=bool(args.es):parser.error('--ed and --es must be supplied together')
    if not (root/'verify_package.py').is_file() or not (root/'MANIFEST.json').is_file():
        parser.error('Provide --root pointing to the extracted minimal package, containing MANIFEST.json')
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(root))
    from verify_package import verify
    integrity=verify(root)
    if args.verify_only:
        print(json.dumps(integrity,indent=2));return 0
    output=(args.output or root.parent/(root.name+'_results')).resolve()
    if output.is_relative_to(root):parser.error('--output must be outside the verified package')
    if output.exists():parser.error('--output already exists; choose a new directory')
    ed=args.ed.resolve() if args.ed else root/'sample/patient039_frame01.nii.gz'
    es=args.es.resolve() if args.es else root/'sample/patient039_frame10.nii.gz'
    if not ed.is_file() or not es.is_file():parser.error('ED/ES input files not found')
    # The default PC demo has one CPU owner. Exact INT8 emulation can be slow.
    os.environ['CUDA_VISIBLE_DEVICES']=''
    try:
        import torch
        from software.role_a_acdc.pipeline import run_pair
    except ImportError as exc:
        parser.error(f'Missing dependency: {exc}. Follow README.md to install CPU PyTorch and requirements-cpu.txt')
    torch.set_num_threads(4)
    # Only the packaged ACDC sample uses its explicit direction-only sform repair.
    # Arbitrary caller MRI keeps the runtime strict geometry policy.
    geometry='strict' if args.ed else 'acdc-direction-only-sform-v1'
    result=run_pair(ed,es,output,args.backend,geometry,str(root/'models'))
    print(json.dumps({'backend':args.backend,'device':'cpu','output':str(output),
        'viewer':str(output/'index.html'),'result':result},ensure_ascii=False,indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())
