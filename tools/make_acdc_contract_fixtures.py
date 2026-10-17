"""Generate small, unmistakably synthetic ACDC interface fixtures without a model.

Only Python standard library is needed. No training, downloads, overlay or DMA.
The output directory must not already exist. Never load this pack as a model.
"""
from pathlib import Path
import argparse,hashlib,json,struct

PAIRS=((3,16),(16,16),(16,32),(32,32),(32,64),(64,64),(64,128),(128,128),(192,64),(64,64),(96,32),(32,32),(48,16),(16,16),(16,4))
BIAS={'axial':[9,-7,17,-5],'coronal':[-11,21,5,1],'sagittal':[-15,-5,3,25]}

def rne(value,multiplier):
    p=value*multiplier;a=abs(p);q,r=divmod(a,1<<24)
    q+=r>(1<<23) or (r==(1<<23) and q%2==1)
    return max(-127,min(127,-q if p<0 else q))

def build(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);members=[]
    def write(relative,data):
        path=output/relative;path.parent.mkdir(parents=True,exist_ok=True)
        if isinstance(data,dict):data=(json.dumps(data,ensure_ascii=False,indent=2)+'\n').encode('utf-8')
        path.write_bytes(data);members.append({'path':relative,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    write('SYNTHETIC_DO_NOT_DEPLOY.json',{'synthetic':True,'trained_model':False,'quality_evidence':False,
          'purpose':'DMA/layout/negative-head/CAT/parameter-switch interface tests only',
          'production_loader_must_reject':True})
    write('input_nchw.bin',bytes([129])*65536+bytes(65536)+bytes([127])*65536)
    for direction,bias in BIAS.items():
        layers=[]
        for i,(ci,co) in enumerate(PAIRS):
            kernel=1 if i==14 else 3;n=co*ci*kernel*kernel
            b=bias if i==14 else [0]*co
            parameter=bytes(n)+struct.pack('<'+'i'*co,*b)+struct.pack('<'+'i'*co,*([1<<24]*co))
            name=f'{direction}/C{i+1:02}_parameters.bin';write(name,parameter)
            layers.append({'id':f'C{i+1:02}','input_channels':ci,'output_channels':co,'kernel':kernel,'padding':0 if i==14 else 1,
                         'relu':i<14,'parameters':name,'bytes':len(parameter)})
        write(f'{direction}/expected_head_nchw.bin',b''.join(bytes([b&255])*65536 for b in bias))
        write(f'{direction}/fixture_spec.json',{'schema':'acdc-contract-fixture-v1','synthetic':True,'source_kind':'synthetic',
            'profile_id':f'fixture_{direction}_NOT_A_TRAINED_MODEL','architecture':'ACDC25D-B16-P3-v1',
            'input_shape':[1,3,256,256],'head_shape':[1,4,256,256],'head_bias_int8':bias,
            'parameters_layout':'OIHW int8 weights, then int32LE bias, then int32LE positive Q24',
            'layers':layers,'all_weights_zero':True,'hidden_bias_zero':True,'all_multipliers_q24':1<<24})
    cs,cd,h,w=64,128,4,4
    skip=[(c%31)+1 for c in range(cs) for _ in range(h*w)]
    deep=[(c*4+y*2+x)%64 for c in range(cd) for y in range(2) for x in range(2)]
    expected=[rne(v,1<<23) for v in skip]+[deep[c*4+(y//2)*2+x//2] for c in range(cd) for y in range(h) for x in range(w)]
    write('cat4/input_skip_then_deep.bin',bytes(skip+deep))
    write('cat4/expected_output.bin',bytes(expected))
    lut=lambda m:bytes(rne(v if v<128 else v-256,m)&255 for v in range(256))
    write('cat4/lut_skip_then_deep.bin',lut(1<<23)+lut(1<<24))
    write('cat4/case.json',{'mode':4,'h':4,'w':4,'ci':192,'co':0,'kernel':1,'split':64,'relu':False,
         'skip_multiplier_q24':1<<23,'deep_multiplier_q24':1<<24,'input_bytes':1536,'output_bytes':3072,
         'lut_bytes':512,'lut_index':'raw uint8 representation of the signed input byte'})
    write('q24_cases.json',{'multiplier_q24':1<<23,'accumulator':[-5,-3,-1,0,1,3,5],
                          'expected':[-2,-2,0,0,0,2,2],'rounding':'signed nearest ties to even'})
    write('README.md',b'# SYNTHETIC INTERFACE FIXTURES\n\nNo trained weights, no dataset, no segmentation quality evidence.\nDo not load into the production model loader or publish as a trained model.\nThree constant but distinct heads test profile-switch cache invalidation.\nAll first-layer input channels and all head channels are contiguous CHW.\n')
    manifest={'schema':'acdc-contract-fixture-manifest-v1','synthetic':True,'members':members,
              'bytes':sum(m['bytes'] for m in members),'models_and_datasets':False}
    (output/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    m=build(a.out);print(json.dumps({'synthetic':True,'files':len(m['members']),'bytes':m['bytes'],'output':str(a.out)},ensure_ascii=False))
