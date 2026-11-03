"""Exact GPU int8 x int8 -> int32 accumulation, bounded workspace per image.

unfold only copies integer-valued float32 values; no floating multiplication or
accumulation is used. The signed127 values are exactly representable. Backend
availability is explicit; a failed GPU call raises, rather than changing maths.
"""
import torch
import torch.nn.functional as F

def pack_weight(weight):
    co,ci,k,kw=weight.shape
    if weight.dtype!=torch.int8 or k!=kw:raise ValueError('int8 OIHW weight required')
    count=ci*k*k;kp=(count+31)//32*32;cp=(co+7)//8*8
    return F.pad(weight.reshape(co,count),(0,kp-count,0,cp-co)).T.contiguous(),kp

def conv_int32(x,packed,kernel,padding,bias):
    matrix,kp=packed;n,ci,h,w=x.shape;co=bias.numel();oh=h+2*padding-kernel+1;ow=w+2*padding-kernel+1
    if x.dtype!=torch.int8 or x.device.type!='cuda':raise ValueError('CUDA int8 NCHW input required')
    result=[]
    for sample in x.split(1):
        unfolded=F.unfold(sample.float(),kernel_size=kernel,padding=padding)[0].T.contiguous().to(torch.int8)
        rows,count=unfolded.shape;mp=(rows+31)//32*32
        operand=F.pad(unfolded,(0,kp-count,0,mp-rows))
        accumulator=torch._int_mm(operand,matrix)[:rows,:co]
        accumulator=accumulator+bias.reshape(1,co)
        result.append(accumulator.T.reshape(co,oh,ow).contiguous())
    return torch.stack(result)
