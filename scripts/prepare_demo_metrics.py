"""Compile existing paired reports for web/HDMI; never runs CPU or PL inference."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from perf_eval.demo import prepare_catalog

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    result=prepare_catalog(args.config,args.output)
    print('DEMO_METRICS_PREPARED',len(result['records']),'paired records',str(args.output))
