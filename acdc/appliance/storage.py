"""Account immutable result jobs once; rescan only the last active writers."""
from pathlib import Path
import json
import stat
import threading

from acdc.contract import ContractError
from .engine import checked_job


class ResultUsage:
    def __init__(self,root,*,active_job=None,limit=4*1024**3):
        self.root=Path(root);self.limit=limit;self.lock=threading.Lock()
        names={p.name for sub in ('inputs','jobs') if (self.root/sub).is_dir()
               for p in (self.root/sub).iterdir() if p.is_dir()}
        self.sizes={name:self.measure(name) for name in names}
        self.dirty={active_job} if active_job else set()
        for name in names:
            try:status=json.loads((self.root/'jobs'/name/'status.json').read_text('utf-8'))['status']
            except (FileNotFoundError,ValueError,KeyError):status=None
            if status not in ('assets_ready','failed','cancelled'):self.dirty.add(name)

    def measure(self,job):
        total=0
        for sub in ('inputs','jobs'):
            for path in (self.root/sub/job).rglob('*'):
                try:info=path.stat()
                except FileNotFoundError:continue  # active job's atomic rename during Web startup
                if stat.S_ISREG(info.st_mode):total+=info.st_size
        return total

    def admit(self,job,length):
        """Call after the engine grants the exclusive upload reservation.

        The previous engine writer has finished. Files from older jobs are
        immutable except client timing, whose writer calls refresh(). External
        maintenance must restart Web to rebuild the initial disk inventory.
        """
        checked_job(job)
        with self.lock:
            for previous in self.dirty:self.sizes[previous]=self.measure(previous)
            self.dirty.clear()
            used=sum(self.sizes.values())
            if used+length>self.limit:
                raise ContractError('V2 result storage reached 4 GiB; export and clear old V2 jobs, then restart Web')
            self.dirty.add(job)
            return used

    def refresh(self,job):
        checked_job(job)
        with self.lock:
            self.sizes[job]=self.measure(job);self.dirty.discard(job)
