"""Source-bound full PS7 build. Artifacts are candidates until board validation."""
import hashlib
import json
import re
import shutil
import subprocess
import time
import zipfile
from pathlib import Path
from run_sim import ROOT, VIVADO


def main():
    stamp=time.strftime('%Y%m%d_%H%M%S')
    out=ROOT/'out'/('b'+stamp[4:].replace('_',''))
    out.mkdir(parents=True,exist_ok=False)
    evidence=ROOT/'evidence'/('board_build_'+stamp);evidence.mkdir()
    sources=sorted((ROOT/'rtl').glob('*.sv'))+[ROOT/'tools/build_board.tcl',Path(__file__),ROOT.parents[1]/'vivado/ps7_config.tcl']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    result={'kind':'full_bd_candidate_not_board_validation','sources':hashes,'passed':False,'output_dir':str(out)}
    def save(): (evidence/'receipt.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    save()
    command=[str(VIVADO/'vivado.bat'),'-mode','batch','-source',str(ROOT/'tools/build_board.tcl'),
             '-nojournal','-log',str(out/'driver.log'),'-tclargs',str(out),'bit']
    try:
        with (out/'console.log').open('w',encoding='utf-8') as fp:
            p=subprocess.run(command,cwd=out,stdout=fp,stderr=subprocess.STDOUT,timeout=7200)
        result['returncode']=p.returncode
        assert hashes=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},'source changed during build'
        console=(out/'console.log').read_text()
        result['critical_warnings']=sorted(set(re.findall(r'^CRITICAL WARNING:.*$',console,re.M)))
        for name in ('timing.rpt','utilization.rpt','drc.rpt'):
            if (out/name).exists():shutil.copyfile(out/name,evidence/name)
        assert p.returncode==0 and 'SKIN8_FULL_BUILD_PASS' in console,'full build failed'
        timing=(out/'timing.rpt').read_text()
        section=timing.split('Design Timing Summary',1)[1]
        line=re.search(r'^\s*(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(\d+)\s+(\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(\d+)\s+(\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(\d+)\s+(\d+)',section,re.M)
        assert line,'missing full timing summary'
        result.update(wns_ns=float(line[1]),whs_ns=float(line[5]),wpws_ns=float(line[9]))
        assert min(result[k] for k in ('wns_ns','whs_ns','wpws_ns'))>=0,'negative setup/hold/pulse-width slack'
        with zipfile.ZipFile(out/'skin8.xsa') as z:
            files={suffix:[n for n in z.namelist() if n.endswith(suffix)] for suffix in ('.bit','.hwh')}
            assert all(len(v)==1 for v in files.values()),'ambiguous bit/hwh pair'
            for suffix,names in files.items():(out/('skin8'+suffix)).write_bytes(z.read(names[0]))
        result['artifacts']={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in ('skin8.xsa','skin8.bit','skin8.hwh')}
        result['passed']=True
    finally:save()
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
