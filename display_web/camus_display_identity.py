"""Optional display sidecar after successful CAMUS inference; no core changes."""
import hashlib
import json
from pathlib import Path
import re

SCHEMA='pld-camus-display-identity-v1'


def _sha(value):
    return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None


def _read(path,limit):
    with Path(path).open('rb') as stream:
        data=stream.read(limit+1)
    if len(data)>limit:raise ValueError('display identity input too large')
    value=json.loads(data)
    if not isinstance(value,dict):raise ValueError('display identity requires an object')
    return data,value


def write_runtime_identity(pipeline,folder):
    """Use the manifest already validated by the successful hardware session.

    Does not hash/reopen the bitstream, load hardware, rewrite report.json or
    amend its ZIP. Unsupported drivers simply have no display identity.
    """
    folder=Path(folder);temporary=folder/'hud_runtime.json.tmp'
    try:
        if getattr(pipeline,'backend',None)!='pl':return None
        device=getattr(pipeline,'device',None)
        overlay=getattr(getattr(device,'io',None),'overlay',None)
        bit_name=getattr(overlay,'bitfile_name',None)
        if not isinstance(bit_name,(str,Path)):return None
        bit=Path(bit_name).resolve()
        if bit.name!='dma_top.bit' or bit.parent.name!='overlay':return None
        raw,report=_read(folder/'report.json',4*1024*1024)
        _,manifest=_read(bit.parent.parent/'runtime_manifest.json',2*1024*1024)
        model=report.get('model',{}).get('spec_sha256')
        if (report.get('schema')!='pld-camus-result-v1' or report.get('backend')!='pl'
                or manifest.get('schema')!='camus-runtime-v1'
                or not _sha(model) or manifest.get('model_spec_sha256')!=model):return None
        bit_sha=manifest.get('files',{}).get('overlay/dma_top.bit')
        if not _sha(bit_sha):return None
        record=dict(schema=SCHEMA,model_spec_sha256=model,bit_sha256=bit_sha,
                    source_report_sha256=hashlib.sha256(raw).hexdigest())
        temporary.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
        temporary.replace(folder/'hud_runtime.json')
        return bit_sha
    except (OSError,ValueError,TypeError,AttributeError):
        # Optional annotation must not turn a completed baseline inference into
        # a failed task. No fallback identity is guessed from another build.
        try:temporary.unlink(missing_ok=True)
        except OSError:pass
        return None


def read_runtime_identity(folder,model_sha):
    try:
        folder=Path(folder)
        _,record=_read(folder/'hud_runtime.json',4096)
        raw,_=_read(folder/'report.json',4*1024*1024)
        if (record.get('schema')==SCHEMA and record.get('model_spec_sha256')==model_sha
                and _sha(model_sha) and _sha(record.get('bit_sha256'))
                and record.get('source_report_sha256')==hashlib.sha256(raw).hexdigest()):
            return record['bit_sha256']
    except (OSError,ValueError,TypeError):pass
    return None
