"""CAMUS CLI: shared pipeline for reference, tiled emulator and explicit PL."""
import argparse,json,sys,time
from pathlib import Path
from .model import DEFAULT_MODEL_ROOT,load_model
from .plan import build_plan
from .pipeline import CamusPipeline

def main(argv=None):
    p=argparse.ArgumentParser(prog='python -m camus')
    p.add_argument('--model',type=Path,default=DEFAULT_MODEL_ROOT)
    p.add_argument('--backend',choices=('reference','emulator','pl'),default='reference')
    p.add_argument('--runtime',type=Path,help='verified extracted CAMUS runtime for PL')
    p.add_argument('--timeout',type=float,default=3.)
    p.add_argument('--plan',action='store_true')
    p.add_argument('--spacing',nargs=2,type=float,metavar=('Y_MM','X_MM'))
    p.add_argument('--no-html',action='store_true')
    p.add_argument('input',type=Path,nargs='?');p.add_argument('output_dir',type=Path,nargs='?')
    args=p.parse_args(argv)
    try:
        if args.plan:
            if args.input or args.output_dir:raise ValueError('--plan takes no input/output')
            print(json.dumps(build_plan(load_model(args.model)),indent=2));return 0
        if args.input is None or args.output_dir is None:raise ValueError('INPUT and new OUTPUT_DIR are required')
        started=time.perf_counter()
        with CamusPipeline(args.model,args.backend,args.runtime,args.timeout) as pipeline:
            result=pipeline.run(args.input,args.output_dir,args.spacing,not args.no_html)
        result.pop('report');result['command_wall_seconds']=time.perf_counter()-started
        print(json.dumps(result));return 0
    except (OSError,ValueError,RuntimeError) as exc:
        print('camus: '+str(exc),file=sys.stderr);return 2

if __name__=='__main__':raise SystemExit(main())
