"""Plan full-width stripes, resident input reuse, and NCHWc8 DDR tensors.

This is a compiler-side ledger, not a running FPGA graph controller.
"""
import hashlib
import json
import math
from budget import ROOT, REPO, MODELS

def main():
    result={'kind':'planned_static_schedule_not_measured','layout':'Cgroup,H,W,8 (NCHWc8)',
            'feature_page_words_per_parity':4096,'models':{}}
    for model,(path,sha,_) in MODELS.items():
        raw=(REPO/path).read_bytes();assert hashlib.sha256(raw).hexdigest()==sha
        spec=json.loads(raw);layers=spec['layers'];plans=[];offset=0
        initial_ci=layers[0]['weight_shape'][1]
        offset=math.ceil(initial_ci/8)*256*256*8
        for li,layer in enumerate(layers):
            co,ci,k,_=layer['weight_shape']
            _,_,h,w=layer.get('output_shape',layer.get('full_output_shape'))
            g=(ci+7)//8;og=(co+7)//8;pad=k//2;iw=w+2*pad
            max_rows=4096//(math.ceil(iw/2)*g)-2*pad
            assert max_rows>=1,(model,li,'no legal full-width stripe')
            stripes=[]
            skip_index={8:5,10:3,12:1}.get(li)
            mode='concat_nearest2' if skip_index is not None else 'maxpool2' if li in (2,4,6) else 'map' if li else 'input'
            for y in range(0,h,max_rows):
                height=min(max_rows,h-y)
                valid_y0=max(0,y-pad);valid_y1=min(h,y+height+pad)
                rows=valid_y1-valid_y0
                words=math.ceil(iw/2)*(height+2*pad)*g
                assert words<=4096
                if skip_index is not None:
                    sc=layers[skip_index]['weight_shape'][0]//8
                    # Planned source-row cache eliminates nearest2 duplicate reads.
                    deep_rows=(valid_y1-1)//2-valid_y0//2+1
                    read_bytes=rows*w*sc*8+deep_rows*(w//2)*(g-sc)*8
                elif mode=='maxpool2':read_bytes=rows*w*g*8*4
                else:read_bytes=rows*w*g*8
                weights=og*8*g*8*k*k
                stripes.append({'y':y,'height':height,'feature_words_per_parity':words,
                                'input_ddr_bytes':read_bytes,'weight_ddr_bytes':weights,
                                'parameter_bytes':og*64,'output_ddr_bytes':height*w*og*8,
                                'conv_tile_starts':og})
            tensor_bytes=og*h*w*8
            plans.append({'index':li,'name':layer['name'],'mode':mode,'skip_source_layer':skip_index,
                'source_layer':li-1 if li else None,'ci':ci,'co':co,'h':h,'w':w,'kernel':k,
                'output_offset':offset,'output_bytes':tensor_bytes,'stripes':stripes,
                'relu_in_quantizer':model=='camus' and li<14,
                'mapping_contract':'skin float32-generated ReLU+rescale LUT' if model=='skin' and li else
                    'CAMUS frozen concat Q24 LUT; identity otherwise' if model=='camus' else 'identity'})
            offset+=tensor_bytes
        sums={key:sum(s[key] for p in plans for s in p['stripes']) for key in
              ('input_ddr_bytes','weight_ddr_bytes','parameter_bytes','output_ddr_bytes','conv_tile_starts')}
        sums['total_ddr_bytes']=sum(sums[k] for k in ('input_ddr_bytes','weight_ddr_bytes','parameter_bytes','output_ddr_bytes'))
        result['models'][model]={'spec_sha256':sha,'arena_bytes_without_reuse':offset,'layers':plans,'totals':sums,
            'assumptions':['weights reload per output block per stripe', 'input stays resident across output blocks',
                           'nearest2 source-row cache', 'pool uses two source rows', 'padding creates no DDR reads',
                           'DDR latency and contention not included', 'head stored with padded eight channels']}
        print(model,sums,'arena_bytes',offset)
    (ROOT/'evidence/stripe_schedule.json').write_text(json.dumps(result,indent=2),encoding='utf-8')

if __name__=='__main__':main()
