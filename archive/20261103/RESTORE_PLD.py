"""Restore the verified PLD closeout workspace using Python's standard library."""
from pathlib import Path, PurePosixPath
import argparse, json, hashlib, zipfile, fnmatch, os

def digest_file(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while chunk:=f.read(4*1024*1024):h.update(chunk)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--assets',type=Path,default=Path(__file__).resolve().parent)
    ap.add_argument('--catalog',type=Path)
    ap.add_argument('--target',type=Path)
    ap.add_argument('--include',action='append',default=[])
    ap.add_argument('--verify-only',action='store_true')
    args=ap.parse_args()
    assets=args.assets.resolve()
    catalog=args.catalog or assets/'PLD_Workspace_Catalog_20261103.zip'
    with zipfile.ZipFile(catalog) as z:
        rows=[json.loads(l) for l in z.read('WORKSPACE_FILES.jsonl').decode('utf-8').splitlines()]
        objects=[json.loads(l) for l in z.read('OBJECTS.jsonl').decode('utf-8').splitlines()]
        packs=json.loads(z.read('PACKS.json'))
    object_map={o['sha256']:o for o in objects}
    assert len(object_map)==len(objects), 'Duplicate object records'
    selected=[r for r in rows if not args.include or any(fnmatch.fnmatchcase(r['path'],p) for p in args.include)]
    assert selected, 'No selected files'
    needed={object_map[r['sha256']]['pack'] for r in selected}
    for p in packs:
        if p['name'] not in needed:continue
        path=assets/p['name']
        assert path.is_file(), f'Missing {path.name}'
        assert path.stat().st_size==p['bytes'] and digest_file(path)==p['sha256'], f'Pack checksum mismatch: {path.name}'
    if not args.verify_only:
        assert args.target is not None, '--target is required for restoration'
        target=args.target.resolve()
        assert not target.exists() or not any(target.iterdir()), 'Target must be new or empty; existing files will not be overwritten'
        target.mkdir(parents=True,exist_ok=True)
    checked=set(); opened={}
    try:
        for r in selected:
            rel=PurePosixPath(r['path'])
            assert not rel.is_absolute() and '..' not in rel.parts and not any(':' in part for part in rel.parts), 'Unsafe stored path'
            o=object_map[r['sha256']]
            if args.verify_only and r['sha256'] in checked:continue
            if o['pack'] not in opened:opened[o['pack']]=zipfile.ZipFile(assets/o['pack'])
            h=hashlib.sha256();count=0
            if not args.verify_only:
                dest=target.joinpath(*rel.parts)
                assert target in dest.resolve().parents
                dest.parent.mkdir(parents=True,exist_ok=True)
                output=dest.open('xb')
            else:output=None
            try:
                with opened[o['pack']].open(o['member']) as src:
                    while chunk:=src.read(4*1024*1024):
                        h.update(chunk);count+=len(chunk)
                        if output:output.write(chunk)
            finally:
                if output:output.close()
            assert count==r['bytes'] and h.hexdigest()==r['sha256'], f'File checksum mismatch: {r["path"]}'
            if not args.verify_only:os.utime(dest,ns=(r['mtime_ns'],r['mtime_ns']))
            checked.add(r['sha256'])
    finally:
        for z in opened.values():z.close()
    print(json.dumps({'status':'verified' if args.verify_only else 'restored','selected_paths':len(selected),'unique_objects_checked':len(checked)},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
