"""Bounded sequential experiments with full epoch-boundary recovery and quota lease."""
from __future__ import annotations
import argparse,collections,json,math,os,random,time,tempfile,shutil
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .common import DATA,RUN,ARCHITECTURE,sha256,save_json,code_identity,utc_now
from .model import ACDCUNet
from .geometry import AXES,stack,plane,letterbox,undo_letterbox,put_plane,probabilities_to_native

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def torch_save(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('wb') as f:
        torch.save(payload,f);f.flush();os.fsync(f.fileno())
    if path.name=='last.pt' and path.exists():
        previous=path.with_name('previous.pt');saved_previous=previous.with_name('previous.pt.tmp')
        shutil.copy2(path,saved_previous);os.replace(saved_previous,previous)
    os.replace(temporary,path)

def get_rng():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}

def set_rng(r):
    random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch'])
    if torch.cuda.is_available() and r['cuda']:torch.cuda.set_rng_state_all(r['cuda'])

def quota_ok():
    lease=RUN/'QUOTA_LEASE.json'
    if not lease.exists():return False
    r=json.loads(lease.read_text(encoding='utf-8'))
    return min(r['primary_remaining'],r['weekly_remaining'])>10 and time.time()<r['valid_until_epoch']

class Volumes:
    def __init__(self,part,direction):
        manifest=json.loads((DATA/'CACHE_MANIFEST.json').read_text(encoding='utf-8'))
        if manifest['test_access']:raise ValueError('test access denied')
        self.records=[r for r in manifest['volumes'] if r['split']==part]
        self.direction=direction;self.axis=AXES[direction];self.cache=collections.OrderedDict();self.positive={}
        for r in self.records:
            y=np.load(DATA/r['relative']/'label.npy',mmap_mode='r')
            axes=tuple(a for a in range(3) if a!=self.axis)
            self.positive[r['key']]=np.where(np.any(y>0,axis=axes))[0].tolist()

    def load(self,r):
        key=r['key']
        if key not in self.cache:
            d=DATA/r['relative'];self.cache[key]=(np.load(d/'image.npy',mmap_mode='r'),np.load(d/'label.npy',mmap_mode='r'))
            if len(self.cache)>4:self.cache.popitem(last=False)
        else:self.cache.move_to_end(key)
        return self.cache[key]

    def batch(self,batch_size,baseline=False):
        images=[];labels=[]
        for _ in range(batch_size):
            r=random.choice(self.records);x,y=self.load(r);pos=self.positive[r['key']]
            index=random.choice(pos) if pos and random.random()<0.75 else random.randrange(x.shape[self.axis])
            sx,t=stack(x,self.direction,index,baseline=baseline)
            sy,_=letterbox(plane(y,self.axis,index),is_label=True)
            images.append(sx);labels.append(sy.astype(np.int64))
        return torch.from_numpy(np.stack(images)),torch.from_numpy(np.stack(labels))

def augment(x,y):
    n=x.shape[0];device=x.device
    angle=(torch.rand(n,device=device)*2-1)*math.radians(12)
    scale=.9+.2*torch.rand(n,device=device)
    flip=torch.where(torch.rand(n,device=device)<.5,-1.,1.)
    theta=torch.zeros(n,2,3,device=device)
    theta[:,0,0]=torch.cos(angle)*scale*flip;theta[:,0,1]=-torch.sin(angle)*scale
    theta[:,1,0]=torch.sin(angle)*scale*flip;theta[:,1,1]=torch.cos(angle)*scale
    theta[:,:,2]=(torch.rand(n,2,device=device)*2-1)*.08
    grid=F.affine_grid(theta,x.shape,align_corners=False)
    x=F.grid_sample(x+1,grid,mode='bilinear',padding_mode='zeros',align_corners=False)-1
    y=F.grid_sample(y[:,None].float(),grid,mode='nearest',padding_mode='zeros',align_corners=False)[:,0].long()
    gain=.9+.2*torch.rand(n,1,1,1,device=device);bias=(torch.rand(n,1,1,1,device=device)*2-1)*.05
    return torch.clamp(x*gain+bias,-1,1),y

def loss_fn(logits,target):
    logits=logits.float();prob=logits.softmax(1)
    gt=F.one_hot(target,4).permute(0,3,1,2).float()
    dice=(2*(prob*gt).sum((0,2,3))+1)/(prob.sum((0,2,3))+gt.sum((0,2,3))+1)
    return F.cross_entropy(logits,target,weight=logits.new_tensor([.2,1.,1.5,1.]))+(1-dice[1:]).mean()

@torch.inference_mode()
def predict_working(model,x,direction,device,baseline=False,batch_size=16):
    model.eval();axis=AXES[direction]
    probs=np.empty((4,*x.shape),dtype=np.float32)
    for start in range(0,x.shape[axis],batch_size):
        prepared=[stack(x,direction,i,baseline=baseline) for i in range(start,min(start+batch_size,x.shape[axis]))]
        inputs=torch.from_numpy(np.stack([a[0] for a in prepared])).to(device)
        with torch.autocast(device_type=device.type,enabled=device.type=='cuda',dtype=torch.float16):
            predictions=model(inputs).float().softmax(1).cpu().numpy()
        for j,(prediction,(_,transform)) in enumerate(zip(predictions,prepared)):
            full=undo_letterbox(prediction,transform)
            for c in range(4):put_plane(probs[c],axis,start+j,full[c])
    return probs

def dice_per_class(pred,label):
    scores=[]
    for c in (1,2,3):
        p=pred==c;g=label==c;den=int(p.sum()+g.sum())
        scores.append(float(2*np.count_nonzero(p&g)/den) if den else 1.0)
    return scores

@torch.inference_mode()
def validate(model,direction,device,baseline=False):
    data=Volumes('dev',direction);rows=[]
    for r in data.records:
        x,y=data.load(r);prob=predict_working(model,x,direction,device,baseline)
        scores=dice_per_class(prob.argmax(0),y)
        rows.append({'key':r['key'],'patient':r['patient'],'phase':r['phase'],'dice':scores})
    by_patient=collections.defaultdict(list)
    for r in rows:by_patient[r['patient']].append(r['dice'])
    patient_scores=[np.mean(v,axis=0) for v in by_patient.values()]
    mean=np.mean(patient_scores,axis=0)
    return {'mean_patient_macro_dice':float(mean.mean()),'class_dice':mean.tolist(),
            'patients':len(by_patient),'volumes':rows,'space':'working RAS 2mm; final acceptance also uses native grid',
            'test_access':False}

def train(config,resume=True):
    if not quota_ok():return {'state':'PROTECTED_QUOTA_NO_START'}
    device=torch.device(config['device']);seed_all(config['seed']);torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    model=ACDCUNet().to(device);model.contract()
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config['max_epochs'],eta_min=1e-5)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    folder=RUN/'training'/config['name'];folder.mkdir(parents=True,exist_ok=True)
    ident={n:h for n,h in code_identity().items() if n in ('common.py','geometry.py','model.py','train.py')}
    split_sha=sha256(DATA/'SPLIT.json');start=0;best=-1;bad=0;history=[]
    checkpoint=folder/'last.pt'
    if checkpoint.exists() and resume:
        state=torch.load(checkpoint,map_location='cpu',weights_only=False) # locally created full-state checkpoint
        if state['config']!=config or state['split_sha256']!=split_sha or state['code_identity']!=ident:raise ValueError('resume identity mismatch')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler']);start=state['epoch'];best=state['best'];bad=state['bad_validations'];history=state['history'];set_rng(state['rng'])
    dataset=Volumes('train',config['direction'])
    save_json(folder/'CONFIG.json',{**config,'architecture':ARCHITECTURE,'split_sha256':split_sha,'code_identity':ident})
    t0=time.time()
    if start>=config['max_epochs'] or bad>=config['patience_validations']:
        result={'state':'COMPLETED','epoch':start,'best_dev_dice':best,'best_sha256':sha256(folder/'best.pt'),
                'last_sha256':sha256(checkpoint),'seconds_this_resume':0,'time':utc_now()}
        save_json(folder/'STATUS.json',result);return result
    for epoch in range(start,config['max_epochs']):
        if (folder/'STOP_REQUEST').exists() or (RUN/'STOP_REQUEST').exists() or not quota_ok():
            save_json(folder/'STATUS.json',{'state':'PROTECTED_AT_EPOCH_BOUNDARY','epoch':epoch,'best':best,'time':utc_now()});return {'state':'PROTECTED','epoch':epoch}
        model.train();total=0;t=time.time()
        for step in range(config['steps_per_epoch']):
            x,y=dataset.batch(config['batch_size'],baseline=config['baseline']);x=x.to(device);y=y.to(device)
            x,y=augment(x,y);optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=device.type=='cuda',dtype=torch.float16):loss=loss_fn(model(x),y)
            if not torch.isfinite(loss):raise RuntimeError('non-finite training loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),12)
            scaler.step(optimizer);scaler.update();total+=loss.item()
        scheduler.step()
        row={'epoch':epoch+1,'loss':total/config['steps_per_epoch'],'train_seconds':time.time()-t,'lr':optimizer.param_groups[0]['lr']}
        if (epoch+1)%config['validate_every']==0 or epoch==0 or epoch+1==config['max_epochs']:
            vt=time.time();evaluation=validate(model,config['direction'],device,config['baseline']);score=evaluation['mean_patient_macro_dice']
            row.update(dev_dice=score,dev_class_dice=evaluation['class_dice'],validation_seconds=time.time()-vt)
            save_json(folder/f'dev_epoch{epoch+1:03}.json',evaluation)
            if score>best+1e-4:
                best=score;bad=0
                torch_save(folder/'best.pt',{'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'config':config,
                    'epoch':epoch+1,'dev_dice':score,'split_sha256':split_sha,'architecture':ARCHITECTURE,'code_identity':ident})
            else:bad+=1
        history.append(row)
        torch_save(checkpoint,{'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
             'scaler':scaler.state_dict(),'rng':get_rng(),'epoch':epoch+1,'best':best,'bad_validations':bad,
             'config':config,'split_sha256':split_sha,'code_identity':ident,'history':history,
             'resume_boundary':'end_of_epoch; no partial epoch continuation','sampler':'Python random state stored; no prefetch workers'})
        save_json(folder/'HISTORY.json',history)
        save_json(folder/'STATUS.json',{'state':'TRAINING','epoch':epoch+1,'max_epochs':config['max_epochs'],
             'best_dev_dice':best,'last':row,'elapsed_seconds':time.time()-t0,'peak_gpu_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else 0,'time':utc_now()})
        print(json.dumps({'name':config['name'],**row,'best':best}),flush=True)
        if bad>=config['patience_validations']:break
    result={'state':'COMPLETED','epoch':epoch+1,'best_dev_dice':best,'best_sha256':sha256(folder/'best.pt'),
            'last_sha256':sha256(checkpoint),'seconds':time.time()-t0,'time':utc_now()}
    save_json(folder/'STATUS.json',result);return result

def standard_config(name):
    baseline=name=='baseline2d';direction='axial' if baseline else name
    if direction not in AXES:raise ValueError(name)
    return {'name':name,'direction':direction,'baseline':baseline,'device':'cuda','seed':20261017,
            'batch_size':8,'steps_per_epoch':128,'max_epochs':150,'validate_every':5,'patience_validations':6,'lr':.001,
            'size':256,'positive_slice_sampling':.75,'class_weights':[.2,1,1.5,1]}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',choices=['baseline2d','axial','coronal','sagittal'],required=True)
    p.add_argument('--smoke',action='store_true');a=p.parse_args();c=standard_config(a.name)
    if a.smoke:c.update(name=a.name+'_smoke',max_epochs=2,steps_per_epoch=4,validate_every=2)
    result=train(c);print(json.dumps(result));raise SystemExit(0 if result['state']=='COMPLETED' else 2)
