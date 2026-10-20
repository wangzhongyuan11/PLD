"""Frozen prepared-input manifests and bounded per-frame benchmarking."""
import json
from pathlib import Path
import time
import numpy as np
from .core import digest,environment,memory_snapshot,summarize,dma_summary
from .engines import Engine,SPECS,source_path


def write_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def prepare(source,model,model_root,inputs,output,synthetic=False):
    source_path(source)
    output=Path(output).resolve()
    if output.exists():raise ValueError('output directory must be new')
    engine=Engine(source,model,model_root,'cpu-int8')
    try:
        cases=[]
        if not synthetic and not inputs:raise ValueError('provide input files or --synthetic')
        output.mkdir(parents=True)
        for i,filename in enumerate([None] if synthetic else inputs):
            if synthetic:
                channels=3 if model=='skin' else 1
                real=np.zeros((1,channels,256,256),np.float32);quant=real.astype(np.int8);valid=np.ones((256,256),bool)
            elif model=='skin':
                from reference.preprocess import decode_rgb,letterbox_rgb,quantize_input
                with decode_rgb(filename) as image:real,valid,_=letterbox_rgb(image)
                quant=quantize_input(real,engine.model.layers[0]['input_scale'])
            else:
                from camus.preprocess import preprocess
                row=preprocess(filename);quant=row.input_int8;real=row.canvas_float32[None,None]
                g=row.letterbox_meta;valid=np.zeros((256,256),bool);valid[g['top']:g['top']+g['new_height'],g['left']:g['left']+g['new_width']]=True
            name=f'case_{i:04d}'
            paths={'int8':name+'_int8.npy','fp32':name+'_fp32.npy','valid':name+'_valid.npy'}
            for key,array in [('int8',quant),('fp32',real),('valid',valid)]:np.save(output/paths[key],array,allow_pickle=False)
            cases.append(dict(id=name,**paths,sha256={k:digest(output/v) for k,v in paths.items()},
                source_sha256=None if synthetic else digest(filename),synthetic=synthetic))
        doc=dict(schema='pld-prepared-input-v1',model=model,spec_sha256=SPECS[model],cases=cases,
                 label_scope='256x256 raw network-grid labels, letterbox padding excluded; no native restoration or island/hole processing',
                 source_scope='synthetic correctness/smoke data; not accuracy benchmark' if synthetic else 'user inputs; no ground truth supplied by preparation')
        write_json(output/'dataset.json',doc)
        return output/'dataset.json'
    finally:engine.close()


def load_dataset(path,model):
    path=Path(path).resolve();doc=json.loads(path.read_text(encoding='utf-8-sig'))
    if doc.get('schema')!='pld-prepared-input-v1' or doc.get('model')!=model or doc.get('spec_sha256')!=SPECS[model]:raise ValueError('dataset schema/model mismatch')
    rows=doc.get('cases')
    if not isinstance(rows,list) or not rows:raise ValueError('dataset must contain cases')
    seen=set()
    for row in rows:
        if not isinstance(row['id'],str) or not row['id'] or row['id'] in seen:raise ValueError('duplicate/invalid case ID')
        seen.add(row['id'])
        for key in ('int8','fp32','valid'):
            member=(path.parent/row[key]).resolve()
            if not member.is_relative_to(path.parent) or not member.is_file() or digest(member)!=row['sha256'][key]:raise ValueError('input path/hash mismatch: '+key)
    return path,doc


def _arrays(path,row,backend):
    path=Path(path).parent
    image=np.load(path/row['fp32' if backend=='cpu-fp32' else 'int8'],allow_pickle=False)
    valid=np.load(path/row['valid'],allow_pickle=False)
    if valid.shape!=(256,256) or valid.dtype!=np.bool_ or not valid.any():raise ValueError('valid must be nonempty bool[256,256]')
    return np.ascontiguousarray(image),valid


def benchmark(args):
    path,doc=load_dataset(args.dataset,args.model)
    output=Path(args.output).resolve()
    if output.exists():raise ValueError('choose a new output directory')
    output.mkdir(parents=True)
    report=dict(schema='pld-benchmark-v1',model=args.model,backend=args.backend,complete=False,
        dataset_sha256=digest(path),environment=environment(),frames=[],warmup_seconds=[],memory=[memory_snapshot()],
        label_scope=doc['label_scope'],synthetic=all(c.get('synthetic',False) for c in doc['cases']),
        timing_scope='inference excludes input read, output hashing, label generation and artifact saving; processing includes read+inference+labels; wall loop includes artifacts/telemetry',
        clinical_accuracy=None,model_initialization_seconds=None,first_forward_seconds=None,
        hardware_execution=args.backend=='pl',baseline='NumPy numerical reference, not optimized CPU library' if args.backend=='cpu-int8' else args.backend)
    engine=None;started=time.perf_counter()
    try:
        engine=Engine(args.source,args.model,args.model_root,args.backend,checkpoint=args.checkpoint,runtime=args.runtime,bit=args.bit,
                      native_copy=args.native_copy,native_control=args.native_control,fuse=args.fuse)
        report['model_initialization_seconds']=time.perf_counter()-started;report['provenance']=engine.provenance
        first,valid=_arrays(path,doc['cases'][0],args.backend)
        # Always measure one cold forward separately, never include it in hot FPS.
        start=time.perf_counter();head,_=engine.run(first,0);report['first_forward_seconds']=time.perf_counter()-start
        del head,first,valid
        for i in range(args.warmup):
            image,_=_arrays(path,doc['cases'][i%len(doc['cases'])],args.backend)
            start=time.perf_counter();head,_=engine.run(image,i+1);report['warmup_seconds'].append(time.perf_counter()-start)
            del image,head
        loop=time.perf_counter();case_hashes={};identities={};index=0
        for repetition in range(args.repeat):
            for case_index,row in enumerate(doc['cases']):
                processing_start=time.perf_counter();t=time.perf_counter();image,valid=_arrays(path,row,args.backend);read_seconds=time.perf_counter()-t
                t=time.perf_counter();head,events=engine.run(image,index+args.warmup+1);seconds=time.perf_counter()-t
                t=time.perf_counter();labels=engine.labels(head,valid);post=time.perf_counter()-t;processing=time.perf_counter()-processing_start
                head_sha=__import__('hashlib').sha256(head.tobytes()).hexdigest()
                key=row['id'];same=key not in case_hashes or case_hashes[key]==head_sha;case_hashes[key]=head_sha
                entry=dict(case_id=key,repetition=repetition,input_sha256=row['sha256']['fp32' if args.backend=='cpu-fp32' else 'int8'],
                    read_seconds=read_seconds,inference_seconds=seconds,label_seconds=post,processing_seconds=processing,
                    head_sha256=head_sha,repeated_output_consistent=same,dma=dma_summary(events,seconds) if events else None)
                if repetition==0:
                    stem=f'case_{case_index:04d}'
                    np.save(output/(stem+'_head.npy'),head,allow_pickle=False);np.save(output/(stem+'_labels.npy'),labels,allow_pickle=False)
                    identities[key]=dict(head=stem+'_head.npy',labels=stem+'_labels.npy',head_file_sha256=digest(output/(stem+'_head.npy')),
                        labels_file_sha256=digest(output/(stem+'_labels.npy')),valid_sha256=row['sha256']['valid'])
                report['frames'].append(entry);index+=1;report['memory'].append(memory_snapshot())
                print('FRAME_DONE',index,'inference_seconds',round(seconds,6),flush=True)
                del image,valid,head,labels,events
        wall=time.perf_counter()-loop
        report['summary']=dict(inference=summarize([r['inference_seconds'] for r in report['frames']]),
            processing=summarize([r['processing_seconds'] for r in report['frames']],wall))
        report['artifacts']=identities;report['complete']=True
    except BaseException as exc:
        report['error']=str(exc)
        raise
    finally:
        try:
            if engine is not None:engine.close()
        except BaseException as exc:
            report['complete']=False;report['close_error']=str(exc)
            raise
        finally:
            report['memory'].append(memory_snapshot());write_json(output/'benchmark.json',report)
    return output/'benchmark.json'
