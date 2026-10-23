"""DDR arena and 64-byte jobs for the spatial executor (separate from ACD4).

Addresses are physical byte addresses. Epoch changes whenever source data or
parameters at an existing address change. Buffers stay owned until checked
completion, or until the vendor DMA engines have been quiesced and reset.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import struct

from acdc.contract import SHAPES
from .layout import BankGeometry
from .planner import PROFILES, plan

KERNEL3, RELU, CAT, POOL, BARRIER = 1, 2, 4, 8, 16
JOB_BYTES = 64


@dataclass(frozen=True)
class Job:
    source0: int
    source1: int
    parameters: int
    lut: int
    output: int
    output_stride: int
    pool_output: int
    pool_stride: int
    image_h: int
    image_w: int
    y: int
    x: int
    height: int
    width: int
    ci: int
    co: int
    split: int
    flags: int
    tag: int
    producer0: int
    producer1: int
    epoch: int

    def validate(self, profile=None):
        values = vars(self)
        if any(type(v) is not int or not 0 <= v <= 0xffffffff for v in values.values()):
            raise ValueError('job fields must be unsigned Python integers')
        for n in ('source0','source1','parameters','lut','output','output_stride','pool_output','pool_stride'):
            if values[n] % 8:
                raise ValueError('unaligned job field: '+n)
        if (self.image_h not in (1,2,4,8,16,32,64,128,256)
                or self.image_w not in (1,2,4,8,16,32,64,128,256)
                or not 1 <= self.height <= self.image_h or not 1 <= self.width <= self.image_w
                or self.y+self.height > self.image_h or self.x+self.width > self.image_w
                or not 1 <= self.ci <= 192 or not 1 <= self.co <= 8 or self.flags & ~31
                or self.output_stride < self.width*8):
            raise ValueError('unsupported job geometry/flags/stride')
        cat = bool(self.flags & CAT)
        if cat:
            if not (0 < self.split < self.ci and self.split % 8 == self.ci % 8 == 0
                    and min(self.image_h,self.image_w) >= 2):
                raise ValueError('CAT needs two complete Ci8 branches')
        elif self.split:
            raise ValueError('non-CAT split must be zero')
        if self.flags & POOL and (any(v % 2 for v in (self.y,self.x,self.height,self.width))
                                 or self.pool_stride < self.width//2*8):
            raise ValueError('unaligned pool geometry/stride')
        extents = [(self.source0,math.ceil((self.split if cat else self.ci)/8)*self.image_h*self.image_w*8),
                   (self.parameters,64+math.ceil(self.ci/8)*(9 if self.flags & KERNEL3 else 1)*64),
                   (self.output,(self.height-1)*self.output_stride+self.width*8)]
        if cat:
            extents += [(self.source1,(self.ci-self.split)//8*(self.image_h//2)*(self.image_w//2)*8),
                        (self.lut,512)]
        if self.flags & POOL:
            extents += [(self.pool_output,(self.height//2-1)*self.pool_stride+self.width//2*8)]
        if any(a+n > 1 << 32 for a,n in extents):
            raise ValueError('32-bit DMA address range wraps')
        if profile is not None:
            prof = PROFILES[profile]
            pad = 2 if self.flags & KERNEL3 else 0
            BankGeometry(prof.p,self.ci,self.height+pad,self.width+pad,prof.input_words)
        return self

    def words(self):
        self.validate()
        return (self.source0,self.source1,self.parameters,self.lut,self.output,self.output_stride,
                self.pool_output,self.pool_stride,(self.image_h<<16)|self.image_w,
                (self.y<<16)|self.x,(self.height<<16)|self.width,
                self.ci|(self.co<<8)|(self.split<<16)|(self.flags<<24),
                self.tag,self.producer0,self.producer1,self.epoch)

    def pack(self):
        return struct.pack('<16I',*self.words())

    @classmethod
    def unpack(cls, packet):
        if len(packet) != JOB_BYTES:
            raise ValueError('job packet must contain 64 bytes')
        w = struct.unpack('<16I',packet)
        return cls(*w[:8],w[8]>>16,w[8]&65535,w[9]>>16,w[9]&65535,w[10]>>16,w[10]&65535,
                   w[11]&255,(w[11]>>8)&255,(w[11]>>16)&255,w[11]>>24,*w[12:]).validate()


class Arena:
    def __init__(self, base, capacity=16*1024*1024):
        if (type(base) is not int or type(capacity) is not int or base < 0 or base % 4096
                or capacity < 4096 or base+capacity > 1 << 32):
            raise ValueError('arena must be page aligned and fit 32-bit physical DDR')
        self.base,self.capacity,self.used,self.buffers = base,capacity,0,{}

    def allocate(self, name, size):
        offset = (self.used+4095)//4096*4096
        if name in self.buffers or size < 1 or offset+size > self.capacity:
            raise ValueError('duplicate or overflowing arena allocation')
        address = self.base+offset
        self.buffers[name] = dict(address=address,bytes=size,offset=offset)
        self.used = offset+size
        return address


def compile_program(profile='P256', base=0x10000000, epoch=1):
    """Compile all 15 frozen convolutions, implicit pools/CAT, layer barriers.

    No model values are needed to compile addresses; this does not certify an
    A bundle. Parameter groups and 3 exact CAT tables must be loaded separately.
    """
    if type(epoch) is not int or not 0 <= epoch <= 0xffffffff:
        raise ValueError('invalid frame epoch')
    schedule = plan(profile)
    arena = Arena(base)
    size = {'input':8*256*256,'head':8*256*256,'work_a':16*256*256,'work_b':16*256*256,
            'skip0':16*256*256,'skip1':32*128*128,'skip2':64*64*64,
            'pool0':16*128*128,'pool1':32*64*64,'pool2':64*32*32,
            'parameters':schedule['model_storage_bytes'],'luts':3*512}
    addr = {name:arena.allocate(name,n) for name,n in size.items()}
    source = ['input','work_a','pool0','work_a','pool1','work_a','pool2','work_a',
              'skip2','work_a','skip1','work_a','skip0','work_a','work_b']
    target = ['work_a','skip0','work_a','skip1','work_a','skip2','work_a','work_b',
              'work_a','work_b','work_a','work_b','work_a','work_b','head']
    producers = [0,1,3,4,6,7,9,10,8,13,5,16,2,19,20]
    jobs,layers = [],[]
    param_offset = 0
    for i,((ci,co,h,k),lp) in enumerate(zip(SHAPES,schedule['layers'])):
        packet_size = 64+math.ceil(ci/8)*k*k*64
        cat = i in (8,10,12); pooled = i in (1,3,5)
        split = {8:64,10:32,12:16}.get(i,0)
        first = len(jobs)
        for tile in lp['tiles']:
            y,x,th,tw = (tile[n] for n in ('y','x','height','width'))
            for group in range(math.ceil(co/8)):
                out = addr[target[i]]+(group*h*h+y*h+x)*8
                pool_address = (addr['pool'+str((i-1)//2)]+(group*(h//2)**2+(y//2)*(h//2)+x//2)*8) if pooled else 0
                flags = (KERNEL3 if k==3 else 0)|(RELU if i<14 else 0)|(CAT if cat else 0)|(POOL if pooled else 0)
                if len(jobs)==first:
                    flags |= BARRIER
                job = Job(addr[source[i]],addr['work_b'] if cat else 0,
                          addr['parameters']+param_offset+group*packet_size,
                          addr['luts']+(i-8)//2*512 if cat else 0,
                          out,h*8,pool_address,h//2*8 if pooled else 0,h,h,y,x,th,tw,ci,
                          min(8,co-group*8),split,flags,len(jobs),producers[i],
                          {8:11,10:14,12:17}.get(i,0),epoch).validate(profile)
                jobs.append(job)
        layers.append(dict(name=f'C{i+1:02}',first_job=first,jobs=len(jobs)-first,
                           parameter_offset=param_offset,parameter_group_bytes=packet_size))
        param_offset += math.ceil(co/8)*packet_size
    addr['program'] = arena.allocate('program',len(jobs)*JOB_BYTES)
    return dict(schema='acdc-spatial-program-v1',profile=profile,epoch=epoch,arena=arena,
                addresses=addr,jobs=jobs,layers=layers,program=b''.join(job.pack() for job in jobs))
