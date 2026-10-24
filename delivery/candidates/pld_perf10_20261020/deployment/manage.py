"""Bounded candidate switching; never edits the original release or its units."""
import argparse, fcntl, hashlib, http.client, json, shutil, subprocess, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / "pld_dual_20261017"
STATE = ROOT / "switch_state.json"
UNIT = "pld-perf10.service"
OLD = ("pld-hub.service","pld-web.service","pld-camus.service","pld-reuse9-web.service")
PY = "/usr/local/share/pynq-venv/bin/python3"

def run(*args, check=True, timeout=150):
    p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if check and p.returncode:
        raise RuntimeError("Command failed: " + " ".join(args[:3]) + ": " + p.stderr[-1200:])
    return p.stdout.strip()

def fields(unit):
    raw = run("systemctl","show",unit,"-p","ActiveState","-p","MainPID","-p","Result","-p","ControlGroup","-p","UnitFileState")
    return dict(x.split("=",1) for x in raw.splitlines() if "=" in x)

def quiesced(unit):
    d = fields(unit)
    if d["ActiveState"] != "inactive" or d["MainPID"] != "0" or d["Result"] != "success":
        raise RuntimeError("Unconfirmed shutdown: " + unit + " " + repr(d))
    group = d.get("ControlGroup")
    p = Path("/sys/fs/cgroup" + group) / "cgroup.procs" if group else None
    if p and p.exists() and p.read_text().strip():
        raise RuntimeError("Remaining cgroup processes: " + unit)

def free_lock():
    with open("/run/pld-unified-fpga.lock","a+b") as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)

def token():
    return Path(json.loads((BASE/"hub/config.json").read_text())["token_file"]).read_text().strip()

def request(method, path, body=None, cookie="", original=False):
    c = http.client.HTTPConnection("127.0.0.1",8765,timeout=15)
    headers = {"Content-Type":"application/json"}
    if original: headers.update(Origin="http://127.0.0.1:8765",Cookie=cookie)
    else: headers["Authorization"]="Bearer "+token()
    try:
        c.request(method,path,None if body is None else json.dumps(body),headers)
        r=c.getresponse(); data=r.read(); session=r.getheader("Set-Cookie","").split(";")[0]
        if r.status not in (200,202): raise RuntimeError("HTTP "+str(r.status)+" "+path)
        return json.loads(data),session
    finally:c.close()

def original_ready():
    _,cookie=request("POST","/hub/login",{"token":token()},original=True)
    d,_=request("GET","/hub/status",cookie=cookie,original=True)
    if d.get("phase")!="ready": raise RuntimeError("Original hub not ready: "+str(d.get("phase")))
    return d,cookie

def original_idle():
    d,cookie=original_ready()
    model=d["model"]
    path="/"+model+"/g/"+str(d["generation"])+"/api/status"
    backend,_=request("GET",path,cookie=cookie,original=True)
    if backend.get("active") or backend.get("busy") or backend.get("warmup")=="running":
        raise RuntimeError("Original backend busy")
    return {"phase":d["phase"],"model":model,"busy":False}

def save(d):
    tmp=STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(d,indent=2));tmp.replace(STATE)

def wait_ready(original=False):
    deadline=time.monotonic()+150
    while time.monotonic()<deadline:
        try:
            if original:
                d,_=original_ready();return {"phase":d["phase"],"model":d["model"]}
            d,_=request("GET","/api/status")
            if d.get("model")=="skin" and not d.get("busy") and not d.get("fault") and not d.get("closed"):
                return {k:d[k] for k in ("model","busy","fault","closed")}
        except (OSError,RuntimeError,http.client.HTTPException):pass
        time.sleep(1)
    raise RuntimeError("Service readiness deadline exceeded")

def stop_old():
    run("systemctl","stop",OLD[0])
    run("systemctl","stop",*OLD[1:])
    for u in OLD:quiesced(u)
    free_lock()

def prepare():
    if STATE.exists():raise RuntimeError("Existing switch state; refusing overwrite")
    target=Path("/etc/systemd/system")/UNIT
    if target.exists():raise RuntimeError("Candidate unit already exists")
    from runtime import verify_payload
    count=verify_payload()
    original=original_idle()
    d={"schema":"pld-perf10-switch-v1","phase":"prepared","original_status":original,
       "original":{u:fields(u) for u in OLD},"payload_files":count,
       "original_unit_sha256":{u:hashlib.sha256(run("systemctl","cat",u).encode()).hexdigest() for u in OLD}}
    save(d)
    shutil.copyfile(ROOT/UNIT,target)
    run("systemd-analyze","verify",str(target))
    run("systemctl","daemon-reload")
    stop_old();d["phase"]="old_quiesced";save(d)
    print(json.dumps({"phase":d["phase"],"payload_files":count}),flush=True)

def start():
    d=json.loads(STATE.read_text())
    for u in OLD:quiesced(u)
    free_lock()
    from runtime import verify_payload
    verify_payload()
    run("systemctl","start",UNIT)
    status=wait_ready();d["phase"]="candidate_running";d["candidate_status"]=status;save(d)
    print(json.dumps(status),flush=True)

def activate():
    d=json.loads(STATE.read_text())
    status=wait_ready()
    run("systemctl","disable",*OLD)
    run("systemctl","enable",UNIT)
    d["phase"]="candidate_enabled";d["candidate_status"]=status;save(d)
    print("PERF10_ENABLED",flush=True)

def rollback():
    d=json.loads(STATE.read_text())
    if fields(UNIT)["ActiveState"]=="active":
        status,_=request("GET","/api/status")
        if status.get("busy"):raise RuntimeError("Candidate busy; finish the current job first")
        run("systemctl","stop",UNIT)
    quiesced(UNIT);free_lock()
    run("systemctl","disable",UNIT)
    for u,old in d["original"].items():
        enabled=old["UnitFileState"]
        if enabled not in ("enabled","disabled"):raise RuntimeError("Unexpected original enable state")
        run("systemctl","enable" if enabled=="enabled" else "disable",u)
        digest=hashlib.sha256(run("systemctl","cat",u).encode()).hexdigest()
        if digest!=d["original_unit_sha256"][u]:raise RuntimeError("Original unit changed: "+u)
    # The hub exclusively starts its backend; avoid the historical startup race.
    run("systemctl","start",OLD[0])
    status=wait_ready(original=True)
    d["phase"]="rolled_back";d["rollback_status"]=status;save(d)
    print(json.dumps({"rollback":status}),flush=True)

def resume():
    original_idle();stop_old();start()

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("mode",choices=("prepare","start","activate","rollback","resume"))
    a=p.parse_args()
    globals()[a.mode]()
