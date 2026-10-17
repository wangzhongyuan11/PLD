"""Package standalone evaluator plus required read-only source dependencies."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile
ROOT=Path(__file__).resolve().parents[1]
NAME='PLD_离线评测_20261017'

def package():
    dest=ROOT/'out'/NAME
    if dest.exists():raise ValueError('package staging already exists; choose a new output or inspect it')
    dest.mkdir(parents=True)
    def put(name,data):
        p=dest/name;p.parent.mkdir(parents=True,exist_ok=True)
        if isinstance(data,str):data=data.encode('utf-8')
        p.write_bytes(data)
    for p in (ROOT/'perf_eval').rglob('*'):
        if p.is_file() and p.suffix in ('.py','.md'):put('perf_eval/'+p.relative_to(ROOT/'perf_eval').as_posix(),p.read_bytes())
    tracked=subprocess.check_output(['git','-c','safe.directory='+ROOT.as_posix(),'-C',str(ROOT),'ls-files','-z']).decode().split('\0')
    for name in tracked:
        p=Path(name)
        if name and p.parts[0] in ('reference','board','camus') and p.suffix in ('.py','.json','.c','.h'):
            put('pld_source/'+name,(ROOT/name).read_bytes())
    put('run_eval.py','from perf_eval.__main__ import main\nif __name__=="__main__":main()\n')
    put('使用说明.md',(ROOT/'perf_eval/README.md').read_bytes())
    put('requirements.txt','numpy>=1.21\nPillow>=8.2\n# Optional FP32: torch (install separately for your CPU/platform)\n')
    put('templates/score_pairs.json',json.dumps(dict(classes=4,dataset_split='development',label_scope='native_labels_no_display_encoding',pairs=[dict(id='case_0001',prediction='prediction.npy',truth='truth.npy',valid='valid_bool.npy')]),indent=2))
    put('templates/power_samples.json',json.dumps(dict(scope='board_input',instrument='fill actual instrument name',frames=None,idle_power_w=None,samples=[]),indent=2))
    evidence=ROOT.parent/'outputs/perf_eval_20261017'
    for folder,new_name in [('skin_inputs','skin_inputs'),('camus_inputs','camus_inputs'),('skin_cpu_final','skin_cpu'),('camus_cpu_final','camus_cpu'),('offline_report','离线示例报告')]:
        for p in (evidence/folder).rglob('*'):
            if p.is_file():put('examples/'+new_name+'/'+p.relative_to(evidence/folder).as_posix(),p.read_bytes())
    for name in ('skin_resource_historical.json','skin_web_timing_historical.json','camus_dma_historical.json','pending_board_measurements.json'):
        put('examples/'+name,(evidence/name).read_bytes())
    put('测试与限制.md','''# 本次验证范围（2026-10-17）

20项自动测试通过：时间统计、DMA计数、Dice/IoU、同GT精度变化、误配拒绝、资源报告解析、HTML转义、模型占用检查、首帧/预热排除、失败及释放异常不标成功等。

两套真实冻结权重分别在本机Windows x86 CPU运行合成输入：1次首次前向+2次热态测量，输出重复一致；不是板端性能，不是原图业务性能，也没有医学GT精度结果。

已完成旧网页计时、皮肤镜历史Vivado资源、CAMUS历史DMA事件导入。导出JSON不携带历史病人路径或图像。旧证据不是新板端实测。

没有连接板子，没有部署或停止服务。PL接口未实测。当前没有PyTorch，FP32入口已编写但尚未执行。功耗缺原报告和仪器数据；DDR仅DMA包流量估算。源码不含新RTL优化或新训练模型。

先看使用说明，再查看examples/离线示例报告/report.html。性能表中的合成数据标志应保持展示，不可删除后当作真实医学测试。
''')
    put('verify_package.py','''from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent
manifest=json.loads((root/'SHA256.json').read_text(encoding='utf-8'))
for name,expected in manifest.items():
 p=root/name
 if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:raise SystemExit('VERIFY_FAILED '+name)
print('VERIFY_PASS',len(manifest),'files')
''')
    manifest={p.relative_to(dest).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.rglob('*') if p.is_file()}
    put('SHA256.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    archive=ROOT/'out'/(NAME+'.zip')
    with zipfile.ZipFile(archive,'w') as z:
        for p in sorted(dest.rglob('*')):
            if not p.is_file():continue
            info=zipfile.ZipInfo(NAME+'/'+p.relative_to(dest).as_posix(),(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100644<<16
            z.writestr(info,p.read_bytes())
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('archive CRC failed')
    print('PACKAGE_PASS',len(manifest),'files',archive,'SHA256',hashlib.sha256(archive.read_bytes()).hexdigest())
    return dest
if __name__=='__main__':package()
