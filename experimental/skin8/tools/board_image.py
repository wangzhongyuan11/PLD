"""Relocate experimental frame descriptors into explicitly supplied DDR regions.

Pure host function; does not connect to, reconfigure or allocate on a board.
"""
import hashlib
import json
import struct
from pathlib import Path


def build_image(directory, regions):
    directory=Path(directory)
    manifest=json.loads((directory/'manifest.json').read_text())
    schedule=json.loads((directory/'descriptors.json').read_text())
    descriptors=schedule['descriptors']
    artifacts={}
    def artifact(name):
        # A parameter block is shared by many stripes. Validate each file once
        # per image build, without retaining stale model bytes across sessions.
        if name in artifacts:return artifacts[name]
        data=(directory/name).read_bytes()
        record=manifest['artifacts'][name]
        if len(data)!=record['bytes'] or hashlib.sha256(data).hexdigest()!=record['sha256']:
            raise ValueError('artifact integrity failure: '+name)
        artifacts[name]=data
        return data
    weights=bytearray();weight_offsets=[]
    tables=bytearray()
    for layer in range(15):
        weight_offsets.append(len(weights));weights.extend(artifact(f'{layer:02d}_weights.bin'))
        weights.extend(bytes((-len(weights))%64));tables.extend(artifact(f'{layer:02d}_luts.bin'))
    first=artifact('first_input_nchwc8.bin')
    head_size=manifest['artifacts']['golden_head_nchwc8.bin']['bytes']
    sizes={'input':len(first),'arena':schedule['arena_bytes'],'weights':len(weights),
           'luts':len(tables),'descriptors':len(descriptors)*128,'head':head_size}
    if set(regions)!=set(sizes):raise ValueError('exact named region set required')
    for name,(base,size) in regions.items():
        alignment=128 if name=='descriptors' else 64
        if not isinstance(base,int) or not isinstance(size,int) or base<0 or base%alignment or size<sizes[name] or base+size>0x20000000:
            raise ValueError('invalid physical region: '+name)
    ordered=sorted((base,base+size,name) for name,(base,size) in regions.items())
    if any(a[1]>b[0] for a,b in zip(ordered,ordered[1:])):raise ValueError('overlapping physical regions')
    arena=regions['arena'][0];binary=bytearray()
    final_base=min(d['destination_offset'] for d in descriptors if d['layer']==14)
    for d in descriptors:
        li=d['layer'];a=d['branches'][0];b=d['branches'][1] if len(d['branches'])==2 else None
        if not 0<=li<15:raise ValueError('invalid layer')
        layer=manifest['layers'][li]
        if (d['groups'],d['kernel'],d['width'],d['height'])!=(layer['input_groups'],layer['kernel'],layer['input_shape'][2],layer['input_shape'][1]):
            raise ValueError('descriptor shape differs from frozen model')
        if d['y']<0 or d['rows']<1 or d['y']+d['rows']>d['height']:raise ValueError('invalid stripe geometry')
        if len(d['branches']) not in (1,2) or sum(v['groups'] for v in d['branches'])!=d['groups']:raise ValueError('invalid branch groups')
        for branch in d['branches']:
            sw,sh=d['width'],d['height']
            if branch['mode']==1:sw*=2;sh*=2
            elif branch['mode']==2:
                if sw%2 or sh%2:raise ValueError('odd nearest shape')
                sw//=2;sh//=2
            elif branch['mode']!=0:raise ValueError('invalid spatial mode')
            offset=branch['source_offset'];span=sw*sh*branch['groups']*8
            if offset<0 or offset%8 or branch['groups']<1 or offset+span>sizes['arena']:raise ValueError('source outside arena')
            if li==0 and (offset!=0 or span!=sizes['input']):raise ValueError('invalid first input span')
        weight_size=manifest['artifacts'][d['weights_file']]['bytes']
        if d['weights_offset']<0 or d['weights_offset']%layer['weight_block_bytes'] or d['weights_offset']+layer['weight_block_bytes']>weight_size:
            raise ValueError('weight block outside allocation')
        if d['params_offset']<0 or d['params_offset']%64 or d['params_offset']+64>manifest['artifacts'][d['params_file']]['bytes']:
            raise ValueError('parameter block outside allocation')
        dest_offset=d['destination_offset']-final_base if li==14 else d['destination_offset']
        if dest_offset<0 or dest_offset%8 or dest_offset+d['width']*d['rows']*8>sizes['head' if li==14 else 'arena']:
            raise ValueError('destination outside allocation')
        flags=(d['kernel']==3)|(int(d['relu'])<<1)|(a['table']<<2)|((b['table'] if b else 1)<<3)
        flags|=(d['stripe_epoch_within_frame']%2)<<4
        flags|=(int(b is not None)<<5)|(int(d['reuse_features'])<<6)|(int(d['load_luts'])<<7)
        source=regions['input'][0] if li==0 else arena+a['source_offset']
        source2=arena+b['source_offset'] if b else 0
        destination=regions['head'][0]+d['destination_offset']-final_base if li==14 else arena+d['destination_offset']
        row=bytearray(128)
        struct.pack_into('<4I4H4B2I',row,0,source,source2,regions['weights'][0]+weight_offsets[li]+d['weights_offset'],destination,
                         d['width'],d['height'],d['y'],d['rows'],d['groups'],a['groups'],a['mode'],flags,
                         d['stripe_epoch_within_frame'],regions['luts'][0]+li*512)
        row[48:112]=artifact(d['params_file'])[d['params_offset']:d['params_offset']+64]
        if len(row)!=128:raise ValueError('short parameter block')
        binary.extend(row)
    return {'weights':bytes(weights),'luts':bytes(tables),'descriptors':bytes(binary),
            'first_input':first,'required_sizes':sizes,'count':len(descriptors),'manifest':manifest}
