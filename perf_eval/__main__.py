"""python -m perf_eval --help; safe offline commands import no PYNQ."""
import argparse
import json
from pathlib import Path


def save(path,value):
    path=Path(path)
    if path.exists():raise ValueError('choose a new output file')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('REPORT_WRITTEN',path)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    def model_args(cmd):
        cmd.add_argument('--source',type=Path,required=True,help='existing complete PLD source root')
        cmd.add_argument('--model',choices=('skin','camus'),required=True)
        cmd.add_argument('--model-root',type=Path,required=True,help='int8 directory with model_spec.json')
    prep=sub.add_parser('prepare',help='prepare frozen int8/float32 inputs once; no inference');model_args(prep)
    prep.add_argument('--input',type=Path,action='append',default=[]);prep.add_argument('--synthetic',action='store_true');prep.add_argument('--output',type=Path,required=True)
    run=sub.add_parser('bench',help='run existing reference or explicit PL backend');model_args(run)
    run.add_argument('--backend',choices=('cpu-int8','cpu-fp32','pl'),default='cpu-int8')
    run.add_argument('--dataset',type=Path,required=True);run.add_argument('--output',type=Path,required=True)
    run.add_argument('--repeat',type=int,default=3);run.add_argument('--warmup',type=int,default=1)
    run.add_argument('--checkpoint',type=Path);run.add_argument('--runtime',type=Path);run.add_argument('--bit',type=Path)
    run.add_argument('--native-copy');run.add_argument('--native-control');run.add_argument('--fuse',action='store_true')
    for name in ('utilization','power-report','power-samples','timing','dma-report','score'):
        cmd=sub.add_parser(name);cmd.add_argument('input',type=Path);cmd.add_argument('--output',type=Path,required=True)
    change=sub.add_parser('precision-change');change.add_argument('baseline',type=Path);change.add_argument('candidate',type=Path);change.add_argument('--output',type=Path,required=True)
    comp=sub.add_parser('compare');comp.add_argument('first',type=Path);comp.add_argument('second',type=Path);comp.add_argument('--output',type=Path,required=True)
    doc=sub.add_parser('report');doc.add_argument('--run',type=Path,action='append',default=[]);doc.add_argument('--evidence',type=Path,action='append',default=[]);doc.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv)
    try:
        if args.action=='prepare':
            from .benchmark import prepare
            print('DATASET_WRITTEN',prepare(args.source,args.model,args.model_root,args.input,args.output,args.synthetic))
        elif args.action=='bench':
            if not 1<=args.repeat<=1000 or not 0<=args.warmup<=100:p.error('repeat 1..1000; warmup 0..100')
            from .benchmark import benchmark
            print('BENCHMARK_WRITTEN',benchmark(args))
        elif args.action=='compare':
            from .report import compare
            save(args.output,compare(args.first,args.second))
        elif args.action=='precision-change':
            from .report import precision_change
            save(args.output,precision_change(args.baseline,args.candidate))
        elif args.action=='score':
            from .report import score_pairs
            save(args.output,score_pairs(args.input))
        elif args.action=='report':
            from .report import bundle
            print('BUNDLE_WRITTEN',bundle(args.run,args.evidence,args.output))
        else:
            from .evidence import utilization,power_report,integrate_power,imported_timing,imported_dma
            handlers={'utilization':utilization,'power-report':power_report,'timing':imported_timing,'dma-report':imported_dma,
                      'power-samples':lambda path:integrate_power(json.loads(path.read_text(encoding='utf-8-sig')))}
            save(args.output,handlers[args.action](args.input))
    except (ValueError,OSError,KeyError,RuntimeError,ImportError) as exc:p.exit(1,'EVALUATION_FAILED: '+str(exc)+'\n')


if __name__=='__main__':main()
