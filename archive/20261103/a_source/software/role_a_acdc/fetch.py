"""Official Girder API only; deliberately never lists or downloads testing/."""
from __future__ import annotations
import argparse, concurrent.futures, json, re, time, urllib.request, shutil, hashlib
from pathlib import Path
from .common import DATA, RUN, save_json, sha256, utc_now

API='https://humanheart-project.creatis.insa-lyon.fr/database/api/v1'
TRAIN_FOLDER='63721d7073e9f0047faa0525'

def get_json(url):
    cache=RUN/'source_api_cache'/ (hashlib.sha256(url.encode()).hexdigest()+'.json')
    if cache.exists(): return json.loads(cache.read_text(encoding='utf-8'))
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url,timeout=60) as f: value=json.load(f)
            save_json(cache,value)
            return value
        except Exception:
            if attempt==2: raise
            time.sleep(1+2*attempt)

def patient_files(folder):
    name=folder['name']
    if not re.fullmatch(r'patient(?:00[1-9]|0[1-9][0-9]|100)',name): raise ValueError(name)
    items=get_json(f"{API}/item?folderId={folder['_id']}&limit=100")
    selected=[i for i in items if i['name']=='Info.cfg' or re.fullmatch(name+r'_frame\d{2}(?:_gt)?\.nii\.gz',i['name'])]
    if len(selected)!=5: raise ValueError(f'{name}: expected Info and four ED/ES files')
    result=[]
    for item in selected:
        files=get_json(f"{API}/item/{item['_id']}/files?limit=10")
        if len(files)!=1 or files[0]['name']!=item['name']: raise ValueError('ambiguous item')
        f=files[0]
        result.append({'patient':name,'relative':f"training/{name}/{f['name']}",'file_id':f['_id'],
                       'bytes':int(f['size']),'url':f"{API}/file/{f['_id']}/download",'official_sha512':f.get('sha512')})
    return result

def enumerate_training():
    folders=get_json(f'{API}/folder?parentType=folder&parentId={TRAIN_FOLDER}&limit=200')
    if len(folders)!=100: raise ValueError(f'expected 100 training patients, got {len(folders)}')
    records=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for i,files in enumerate(pool.map(patient_files,folders),1):
            records.extend(files)
            if i%10==0: print(f'metadata {i}/100',flush=True)
    records.sort(key=lambda r:r['relative'])
    if sum(r['bytes'] for r in records)>1_000_000_000: raise ValueError('ED/ES-only budget exceeded')
    manifest={'schema':'acdc-official-training-source-v1','created':utc_now(),'official_api':API,
              'license':'CC BY-NC-SA 4.0; strictly non-commercial scientific research',
              'citation':'Bernard et al., IEEE TMI 2018, DOI 10.1109/TMI.2018.2837502',
              'patients':100,'files':records,'total_bytes':sum(r['bytes'] for r in records),
              'testing_enumerated':False,'cine_4d_downloaded':False,'selection':'ED/ES images and expert labels only'}
    save_json(DATA/'SOURCE_MANIFEST.json',manifest)
    print(json.dumps({'files':len(records),'bytes':manifest['total_bytes']}))

def download_one(record):
    target=DATA/record['relative']; target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if target.stat().st_size!=record['bytes']: raise ValueError(f'wrong existing size: {target}')
    else:
        temp=target.with_name(target.name+'.partial')
        for attempt in range(3):
            try:
                with urllib.request.urlopen(record['url'],timeout=90) as source,temp.open('wb') as dest:
                    shutil.copyfileobj(source,dest,1<<20)
                if temp.stat().st_size!=record['bytes']: raise ValueError('download size mismatch')
                temp.replace(target); break
            except Exception:
                if attempt==2: raise
                time.sleep(1+attempt)
    if record.get('official_sha512'):
        h=hashlib.sha512(target.read_bytes()).hexdigest()
        if h!=record['official_sha512']: raise ValueError('official SHA512 mismatch')
    return {**record,'sha256':sha256(target),'official_hash_available':bool(record.get('official_sha512'))}

def download_training():
    manifest=json.loads((DATA/'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    records=manifest['files']
    if len(records)!=500 or manifest['patients']!=100 or manifest['testing_enumerated']: raise ValueError('manifest not train-only')
    verified=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for i,r in enumerate(pool.map(download_one,records),1):
            verified.append(r)
            if i%25==0:
                save_json(RUN/'DOWNLOAD_PROGRESS.json',{'completed':i,'total':len(records),'time':utc_now()})
                print(f'download {i}/500',flush=True)
    save_json(DATA/'DOWNLOAD_VERIFIED.json',{'created':utc_now(),'source_manifest_sha256':sha256(DATA/'SOURCE_MANIFEST.json'),
                 'files':verified,'bytes':sum(r['bytes'] for r in verified),'patients':100,
                 'test_access':False,'hash_note':'Local SHA256 after TLS download; server hash only where explicitly provided'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['metadata','download']);a=p.parse_args()
    enumerate_training() if a.action=='metadata' else download_training()
