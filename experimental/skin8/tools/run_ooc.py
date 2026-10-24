"""Source-bound OOC feasibility run. No full-BD timing or bitstream claims."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from run_sim import ROOT, VIVADO

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('top',choices=['dot128','requant24','feature_banks','conv_pair_tile','tile_int8','stripe_engine','tensor_load','frame_engine','graph_control'])
    top=parser.parse_args().top
    stamp=time.strftime('%Y%m%d_%H%M%S')
    out=ROOT/'out'/f'{top}_bound_{stamp}'
    out.mkdir(parents=True,exist_ok=False)
    evidence=ROOT/'evidence'/f'{top}_ooc_{stamp}'
    evidence.mkdir(parents=True,exist_ok=False)
    sources=list((ROOT/'rtl').glob('*.sv'))+[ROOT/'tools/dot_ooc.tcl',Path(__file__)]
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    result={'kind':'out_of_context_feasibility_only','top':top,'sources':hashes,'passed':False,
            'limitations':['external I/O unconstrained','HD.CLK_SRC unspecified','not full BD timing','not board performance'], 'output_dir':str(out)}
    (evidence/'receipt.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    env=os.environ.copy();env['SKIN8_BUILD_OUT']=str(out)
    cmd=[str(VIVADO/'vivado.bat'),'-mode','batch','-source',str(ROOT/'tools/dot_ooc.tcl'),'-nojournal','-log',str(out/'driver.log'),'-tclargs',top]
    try:
        with (out/'console.log').open('w',encoding='utf-8') as log:
            p=subprocess.run(cmd,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
        result['returncode']=p.returncode
        assert p.returncode==0 and (out/'complete.txt').exists(),'build failed'
        assert hashes=={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},'source changed during build'
        for name in ('timing_routed.rpt','utilization_routed.rpt'):
            shutil.copyfile(out/name,evidence/name)
        timing=(out/'timing_routed.rpt').read_text()
        match=re.search(r'^clk\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+\d+\s+\d+\s+(-?\d+\.\d+)',timing,re.M)
        assert match,'no timing summary'
        result['wns_ns']=float(match[1]);result['whs_ns']=float(match[3])
        utilization=(out/'utilization_routed.rpt').read_text()
        for label in ('Slice LUTs','Slice Registers','DSPs','Block RAM Tile'):
            match=re.search(r'\| '+re.escape(label)+r'\s+\|\s+([0-9.]+)',utilization)
            if match:result[label]=float(match[1])
        result['passed']=result['wns_ns']>=0 and result['whs_ns']>=0
    finally:
        (evidence/'receipt.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)
    if not result['passed']:raise SystemExit(1)

if __name__=='__main__':main()
