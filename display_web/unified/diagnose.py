"""Read-only board diagnostics; never prints the access token."""
import http.client
import json
from pathlib import Path
import subprocess
from hub import memory

root=Path(__file__).resolve().parent
report={'memory':memory(),'services':{}}
for unit in ['pld-hub.service','pld-web.service','pld-camus.service']:
    r=subprocess.run(['systemctl','show',unit,'-p','ActiveState','-p','SubState','-p','Result','-p','MainPID','-p','WorkingDirectory'],capture_output=True,text=True,timeout=10)
    report['services'][unit]=dict(line.split('=',1) for line in r.stdout.splitlines() if '=' in line)
try:
    config=json.loads((root/'config.json').read_text());token=Path(config['token_file']).read_text().strip()
    conn=http.client.HTTPConnection('127.0.0.1',8765,timeout=10)
    conn.request('POST','/hub/login',json.dumps({'token':token}),{'Origin':'http://127.0.0.1:8765','Content-Type':'application/json'})
    r=conn.getresponse();cookie=r.getheader('Set-Cookie','').split(';')[0];r.read();conn.close()
    if r.status!=200:raise RuntimeError('统一入口登录失败：'+str(r.status))
    conn=http.client.HTTPConnection('127.0.0.1',8765,timeout=10)
    conn.request('GET','/hub/status',headers={'Cookie':cookie});r=conn.getresponse();report['hub']=json.loads(r.read());conn.close()
except Exception as exc:report['hub_error']=str(exc)
print(json.dumps(report,ensure_ascii=False,indent=2))
