"""Reuse file hashes within one engine, while the original verifier still runs."""
from pathlib import Path


def install():
    from acdc import package
    original=package.file_sha
    cache={}

    def file_sha(path):
        path=Path(path)
        before=path.stat()
        signature=lambda value:(value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns)
        key=str(path.resolve());stamp=signature(before)
        previous=cache.get(key)
        if previous and previous[0]==stamp:return previous[1]
        result=original(path)
        # A write during hashing cannot become a reusable verification result.
        if signature(path.stat())==stamp:cache[key]=(stamp,result)
        else:cache.pop(key,None)
        return result

    package.file_sha=file_sha
    return cache.clear
