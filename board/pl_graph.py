"""Frozen TinyUNet scheduling: host gathers bytes; PL computes every operator."""
import math
import hashlib
from pathlib import Path
import sys
import time
from functools import lru_cache

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reference.tinyunet import rescale_lut,require_equal
from reference.tiling import choose_tile,choose_fused_decoder_tile,tiles,input_tile,deep_input_tile
from tensor_accel import shape


@lru_cache(maxsize=64)
def tensor_tile_shape(mode,channels,height,width,split=0):
    """Choose an output tile; concat boundaries must align to nearest-2 cells."""
    candidates=[]
    for h in {height}|{1<<p for p in range(height.bit_length())}:
        for w in {width}|{1<<p for p in range(width.bit_length())}:
            if h>height or w>width or (mode==4 and (h%2 or w%2)):
                continue
            ih,iw=(h*2,w*2) if mode==2 else (h//2,w//2) if mode==3 else (h,w)
            try: shape(mode,ih,iw,channels,split=split)
            except ValueError: continue
            candidates.append((math.ceil(height/h)*math.ceil(width/w),-h*w,-w,h,w))
    if not candidates:
        raise ValueError("no tensor tile fits PL capacities")
    return min(candidates)[-2:]


class PLGraph:
    def __init__(self,model,accelerator,fuse_maps=False,fuse_decoder=False):
        self.model,self.accelerator=model,accelerator
        self.fuse_maps=bool(fuse_maps)
        self.fuse_decoder=bool(fuse_decoder)
        if self.fuse_maps and getattr(accelerator,"abi_version",0)<0x00010001:
            raise ValueError("fused input mapping requires ABI 1.1")
        if self.fuse_decoder and getattr(accelerator,"abi_version",0)<0x00010002:
            raise ValueError("fused decoder input requires ABI 1.2")

    def transition(self,index,previous,skip,frame):
        layers=self.model.layers
        layer=layers[index]
        main=rescale_lut(layers[index-1]["output_scale"],layer["input_scale"])
        shape_out=layer["input_shape"]
        _,channels,height,width=shape_out
        mode=4 if skip is not None else 2 if index in (2,4,6) else 1
        split=skip.shape[1] if skip is not None else 0
        th,tw=tensor_tile_shape(mode,channels,height,width,split)
        result=np.empty(shape_out,dtype=np.int8)
        first=True
        for tile in tiles(height,width,th,tw):
            y,x,h,w=tile.y,tile.x,tile.height,tile.width
            ih,iw=(h*2,w*2) if mode==2 else (h,w)
            self.accelerator.configure(mode,ih,iw,channels,split=split,frame=frame)
            if first:
                if skip is not None:
                    skip_index={8:5,10:3,12:1}[index]
                    skip_lut=rescale_lut(layers[skip_index]["output_scale"],layer["input_scale"])
                    self.accelerator.load_lut(skip_lut,main)
                else: self.accelerator.load_lut(main)
                first=False
            if skip is not None:
                self.accelerator.load_input(skip[:,:,y:y+h,x:x+w],previous[:,:,y//2:(y+h)//2,x//2:(x+w)//2])
            elif mode==2:
                self.accelerator.load_input(previous[:,:,y*2:(y+h)*2,x*2:(x+w)*2])
            else: self.accelerator.load_input(previous[:,:,y:y+h,x:x+w])
            result[:,:,y:y+h,x:x+w]=self.accelerator.run()
        return result

    def convolution(self,index,image,frame,fused=False):
        layer=self.model.layers[index]
        weights,bias,mult=self.model.parameters[index]
        co,ci,k,_=weights.shape
        _,_,height,width=layer["output_shape"]
        th,tw=choose_tile(ci,co,height,width,k)
        result=np.empty(layer["output_shape"],dtype=np.int8)
        first=True
        if fused:
            lut=rescale_lut(self.model.layers[index-1]["output_scale"],layer["input_scale"])
            if lut[0]!=0:
                raise ValueError("fused halo mapping must preserve zero padding")
        for tile in tiles(height,width,th,tw):
            data=input_tile(image,tile,k,layer["padding"])
            self.accelerator.configure(5 if fused else 0,*data.shape[2:],ci,co,k,frame=frame)
            if first:
                self.accelerator.load_parameters(weights,bias,mult)
                if fused: self.accelerator.load_lut(lut)
                first=False
            self.accelerator.load_input(data)
            result[:,:,tile.y:tile.y+tile.height,tile.x:tile.x+tile.width]=self.accelerator.run()
        return result

    def convolution_fused(self,index,skip,deep,frame):
        """Decoder-entry conv: PL maps both branches, upsamples and concatenates.

        The host only gathers bytes: the full-resolution skip halo patch and a
        compact half-resolution deep patch.  Mode 6 loads LUT0/LUT1, applies
        nearest-2 upsampling while the conv engine reads deep channels, keeps
        skip channels first, and computes the convolution exactly as mode 0.
        """
        layers=self.model.layers
        layer=layers[index]
        weights,bias,mult=self.model.parameters[index]
        co,ci,k,_=weights.shape
        if k!=3:
            raise ValueError("fused decoder input requires a 3x3 kernel")
        _,_,height,width=layer["output_shape"]
        split=skip.shape[1]
        if not 0<split<ci or split%4 or deep.shape[1]%4:
            raise ValueError("fused decoder split must align input groups to one branch")
        skip_index={8:5,10:3,12:1}[index]
        lut0=rescale_lut(layers[skip_index]["output_scale"],layer["input_scale"])
        lut1=rescale_lut(layers[index-1]["output_scale"],layer["input_scale"])
        if lut0[0]!=0 or lut1[0]!=0:
            raise ValueError("fused halo mapping must preserve zero padding")
        th,tw=choose_fused_decoder_tile(ci,split,co,height,width)
        tile_list=list(tiles(height,width,th,tw))
        if any(tile.y%2 or tile.x%2 or tile.height%2 or tile.width%2 for tile in tile_list):
            raise ValueError("fused decoder tiling requires even origins and dimensions")
        result=np.empty(layer["output_shape"],dtype=np.int8)
        first=True
        for tile in tile_list:
            skip_patch=input_tile(skip,tile,k,layer["padding"])
            deep_patch=deep_input_tile(deep,tile,k)
            self.accelerator.configure(6,*skip_patch.shape[2:],ci,co,k,split=split,frame=frame)
            if first:
                self.accelerator.load_parameters(weights,bias,mult)
                self.accelerator.load_lut(lut0,lut1)
                first=False
            self.accelerator.load_input(skip_patch,deep_patch)
            result[:,:,tile.y:tile.y+tile.height,tile.x:tile.x+tile.width]=self.accelerator.run()
        return result

    def run(self,image,frame=1,verify=False,callback=None,input_metadata=None):
        image=np.asarray(image)
        if image.dtype!=np.int8 or list(image.shape)!=self.model.layers[0]["input_shape"] or np.any(image==-128):
            raise ValueError("first input must match the frozen model NCHW INT8 contract")
        metadata=dict(input_metadata or {})
        if metadata:
            if metadata.get("model_spec_sha256")!=self.model.spec_hash or metadata.get("input_sha256")!=hashlib.sha256(image.tobytes()).hexdigest():
                raise ValueError("input metadata does not identify these pixels/model")
            rect=metadata.get("valid_rect")
            if not isinstance(rect,list) or len(rect)!=4 or any(not isinstance(v,int) for v in rect):
                raise ValueError("valid_rect must be integer [left,top,width,height]")
            left,top,w,h=rect
            if min(left,top)<0 or min(w,h)<1 or left+w>image.shape[3] or top+h>image.shape[2]:
                raise ValueError("valid rectangle outside the input")
        skips={}; previous=None; records=[]
        start=time.perf_counter()
        for index,layer in enumerate(self.model.layers):
            layer_start=time.perf_counter()
            fused=self.fuse_maps and index in (1,3,5,7,9,11,13,14)
            decoder_entry=self.fuse_decoder and index in (8,10,12)
            if decoder_entry:
                # Previous raw PL output is the deep branch; the retained
                # encoder feature is the skip. No host transition runs here.
                skip_index={8:5,10:3,12:1}[index]
                skip=skips.pop(skip_index)
                previous=self.convolution_fused(index,skip,previous,frame)
                x=None
            else:
                if index==0: x=image
                elif fused: x=previous
                else:
                    skip_index={8:5,10:3,12:1}.get(index)
                    skip=skips.get(skip_index)
                    x=self.transition(index,previous,skip,frame)
                    if skip_index is not None:
                        del skips[skip_index]
                if verify:
                    # Fused input is still the previous raw PL result. The LUT
                    # executes inside the loader; no host arithmetic substitutes it.
                    expected=self.model.golden(index-1,"output") if fused else self.model.golden(index,"input")
                    require_equal(x,expected,layer["directory"]+"/PL source")
                previous=self.convolution(index,x,frame,fused)
            if verify: require_equal(previous,self.model.golden(index,"output"),layer["directory"]+"/PL output")
            if index in (1,3,5): skips[index]=previous
            record={"layer":index,"name":layer["directory"],"seconds":time.perf_counter()-layer_start,
                    "input_shape":list(x.shape) if x is not None else list(layer["input_shape"]),
                    "output_shape":list(previous.shape),
                    "input_mapping":"PL fused decoder input" if decoder_entry else
                                    ("PL load LUT" if fused else "separate PL transition")}
            records.append(record)
            if callback: callback(record,previous)
        if skips:
            raise RuntimeError("unconsumed skip feature")
        result={"frame_id":frame,"model_spec_sha256":self.model.spec_hash,
                "shape":list(previous.shape),"dtype":"int8","layout":"NCHW",
                **self.model.threshold,"model_name":self.model.model_name,
                "model_manifest_sha256":self.model.manifest_hash,
                "valid_rect":metadata.get("valid_rect"),"input_metadata":metadata,
                "output_sha256":hashlib.sha256(previous.tobytes()).hexdigest(),"status":"success"}
        return previous,{"frame":frame,"seconds":time.perf_counter()-start,"layers":records,"result":result,
                         "fused_maps":self.fuse_maps,
                         "fused_decoder":self.fuse_decoder,
                         "timing_scope":"PS tile gathering + PL commands/DMA/compute + stitch; includes golden comparison when verify=True"}
