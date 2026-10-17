"""Pure-Python statistics, memory snapshots and source provenance."""
import hashlib
import math
import os
from pathlib import Path
import platform
import sys
import time


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def positive(value,name,zero=False):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or (value<0 if zero else value<=0):
        raise ValueError(name+' must be finite '+('nonnegative' if zero else 'positive'))
    return float(value)


def percentile(values,q):
    rows=sorted(positive(x,'latency') for x in values)
    if not rows or not 0<=q<=1:raise ValueError('nonempty samples and q in 0..1 required')
    at=(len(rows)-1)*q;lo=int(at);hi=min(lo+1,len(rows)-1)
    return rows[lo]+(rows[hi]-rows[lo])*(at-lo)


def summarize(values,wall_seconds=None):
    rows=list(values)
    if not rows:raise ValueError('empty samples')
    total=sum(positive(x,'latency') for x in rows)
    result=dict(frames=len(rows),mean_seconds=total/len(rows),median_seconds=percentile(rows,.5),
        p95_seconds=percentile(rows,.95),min_seconds=min(rows),max_seconds=max(rows),
        active_seconds=total,active_fps=len(rows)/total,
        p95_scope='sample percentile; small N does not establish population P95')
    if wall_seconds is not None:
        wall_seconds=positive(wall_seconds,'wall_seconds')
        if wall_seconds+1e-6<total:raise ValueError('wall shorter than sequential frame times')
        result.update(wall_seconds=wall_seconds,wall_throughput_fps=len(rows)/wall_seconds)
    return result


def memory_snapshot():
    out=dict(timestamp_monotonic=time.monotonic(),rss_bytes=None,mem_available_bytes=None,cma_free_bytes=None,cma_total_bytes=None)
    # Current RSS, not ru_maxrss (a lifetime high-water mark).
    for name in ('/proc/self/status','/proc/meminfo'):
        try:
            for line in Path(name).read_text().splitlines():
                if ':' not in line:continue
                key,value=line.split(':',1)
                target={'VmRSS':'rss_bytes','MemAvailable':'mem_available_bytes','CmaFree':'cma_free_bytes','CmaTotal':'cma_total_bytes'}.get(key)
                if target:out[target]=int(value.split()[0])*1024
        except OSError:pass
    if out['rss_bytes'] is None and os.name=='nt':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(name,ctypes.c_size_t) for name in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
        counters=Counters();counters.cb=ctypes.sizeof(counters)
        current=ctypes.windll.kernel32.GetCurrentProcess;current.restype=wintypes.HANDLE
        measure=ctypes.windll.psapi.GetProcessMemoryInfo
        measure.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
        if measure(current(),ctypes.byref(counters),counters.cb):out['rss_bytes']=int(counters.WorkingSetSize)
    if out['rss_bytes'] is None:
        try:
            import psutil
            out['rss_bytes']=int(psutil.Process().memory_info().rss)
        except ImportError:pass
    return out


def environment():
    return dict(os=platform.system(),machine=platform.machine(),processor=platform.processor(),
        python=platform.python_version(),platform=platform.platform(),logical_cpus=os.cpu_count(),
        host_fingerprint=hashlib.sha256(platform.node().encode()).hexdigest(),
        thread_environment={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')},
        rss_monitor='sampled at frame boundaries; not continuous peak measurement')


def dma_summary(events,frame_seconds):
    tx=rx=0;wait=0.;count=0
    for event in events:
        if type(event.get('command')) is not int or event['command'] not in (1,2,3,4):raise ValueError('unknown DMA command')
        n=event.get('bytes')
        if type(n) is not int or n<0:raise ValueError('invalid DMA byte count')
        if event['command']==4:rx+=n
        else:tx+=n
        if 'dma_seconds' in event:wait+=positive(event['dma_seconds'],'dma_seconds',zero=True)
        count+=1
    seconds=positive(frame_seconds,'frame_seconds')
    return dict(command_count=count,tx_bytes=tx,rx_bytes=rx,total_packet_bytes=tx+rx,
        estimated_dma_MB_s=(tx+rx)/seconds/1e6,dma_wait_seconds=wait,
        scope='recorded accelerator DMA packets / frame wall time; not total DDR traffic or peak DDR bandwidth')
