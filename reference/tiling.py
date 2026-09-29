"""NCHW halo tiling within the planned 4-input/8-output bank capacities."""
from dataclasses import dataclass
import math
from functools import lru_cache

import numpy as np

from .tinyunet import conv2d

INPUT_BANK_BYTES = 32768  # four banks, one for each channel modulo four
OUTPUT_BANK_BYTES = 8192  # eight banks, one for each output channel modulo eight


@dataclass(frozen=True)
class Tile:
    y: int
    x: int
    height: int
    width: int


@lru_cache(maxsize=64)
def choose_tile(ci, co, height, width, kernel):
    if min(ci, co, height, width) <= 0 or kernel not in (1,3):
        raise ValueError("invalid layer dimensions")
    choices = []
    heights = {height} | {1 << p for p in range(height.bit_length())}
    widths = {width} | {1 << p for p in range(width.bit_length())}
    for h in heights:
        for w in widths:
            if h > height or w > width:
                continue
            input_size = math.ceil(ci/4) * (h+kernel-1) * (w+kernel-1)
            output_size = math.ceil(co/8) * h * w
            if input_size <= INPUT_BANK_BYTES and output_size <= OUTPUT_BANK_BYTES:
                count = math.ceil(height/h) * math.ceil(width/w)
                choices.append((count, -h*w, -w, h, w))
    if not choices:
        raise ValueError("no tile fits bank capacities")
    return min(choices)[-2:]


@lru_cache(maxsize=16)
def choose_fused_decoder_tile(ci, split, co, height, width):
    """Largest useful even decoder tile under mode-6 bank capacities."""
    if (min(ci, split, co, height, width) <= 0 or split >= ci or
            split % 4 or (ci-split) % 4 or height % 2 or width % 2):
        raise ValueError("invalid fused decoder dimensions")
    choices=[]
    heights={height} | {1 << p for p in range(1,height.bit_length())}
    widths={width} | {1 << p for p in range(1,width.bit_length())}
    for h in heights:
        for w in widths:
            if h>height or w>width or h%2 or w%2:
                continue
            ih,iw=h+2,w+2
            input_size=(split//4)*ih*iw+((ci-split)//4)*(ih//2+1)*(iw//2+1)
            output_size=math.ceil(co/8)*h*w
            if input_size<=INPUT_BANK_BYTES and output_size<=OUTPUT_BANK_BYTES:
                count=math.ceil(height/h)*math.ceil(width/w)
                choices.append((count,-h*w,-w,h,w))
    if not choices:
        raise ValueError("no fused decoder tile fits bank capacities")
    return min(choices)[-2:]


def tiles(height, width, tile_height, tile_width):
    for y in range(0,height,tile_height):
        for x in range(0,width,tile_width):
            yield Tile(y,x,min(tile_height,height-y),min(tile_width,width-x))


def input_tile(image, tile, kernel, padding):
    """Copy the exact halo; outside the full image is zero, not a neighbor tile.

    Host work here is byte gathering only. Conv hardware receives a contiguous
    NCHW tile with halo and runs with local padding=0.
    """
    y, x = tile.y-padding, tile.x-padding
    ih, iw = tile.height+kernel-1, tile.width+kernel-1
    result = np.zeros((image.shape[0],image.shape[1],ih,iw),dtype=np.int8)
    y0, y1 = max(y,0), min(y+ih,image.shape[2])
    x0, x1 = max(x,0), min(x+iw,image.shape[3])
    result[:,:,y0-y:y1-y,x0-x:x1-x] = image[:,:,y0:y1,x0:x1]
    return result


def deep_input_tile(image, tile, kernel):
    """Compact half-resolution source patch for a fused decoder-entry conv.

    The conv input patch covers full-resolution rows/cols
    [tile.y-1, tile.y+tile.height+1) for kernel 3 (one halo cell per side).
    After nearest-2 upsample each needed source sample is floor(coord/2);
    for an even tile origin and kernel 3 the patch-local index is (i+1)//2
    and the required source rectangle is (input_h//2+1)x(input_w//2+1)
    starting at tile.origin//2-1, i.e. source rows/cols
    [origin//2-1, origin//2+extent//2+1).  Samples
    outside the source feature stay zero, matching the conv zero halo.  This
    is byte gathering/zero padding only; PL still does upsample/scale/conv.
    """
    if kernel != 3:
        raise ValueError("fused decoder input requires a 3x3 kernel")
    if tile.y % 2 or tile.x % 2:
        raise ValueError("fused decoder tiles require even origins")
    ih, iw = tile.height+kernel-1, tile.width+kernel-1
    dh, dw = ih//2+1, iw//2+1
    dy0, dx0 = tile.y//2-1, tile.x//2-1
    result = np.zeros((image.shape[0],image.shape[1],dh,dw),dtype=np.int8)
    y0, y1 = max(dy0,0), min(dy0+dh,image.shape[2])
    x0, x1 = max(dx0,0), min(dx0+dw,image.shape[3])
    if y0 < y1 and x0 < x1:
        result[:,:,y0-dy0:y1-dy0,x0-dx0:x1-dx0] = image[:,:,y0:y1,x0:x1]
    return result


def conv2d_tiled(image, weights, bias, multipliers, padding, tile_shape=None):
    co, ci, kernel, _ = weights.shape
    height = image.shape[2] + 2*padding-kernel+1
    width = image.shape[3] + 2*padding-kernel+1
    th, tw = tile_shape or choose_tile(ci,co,height,width,kernel)
    output = np.empty((image.shape[0],co,height,width),dtype=np.int8)
    for tile in tiles(height,width,th,tw):
        data = input_tile(image,tile,kernel,padding)
        computed = conv2d(data,weights,bias,multipliers,0)
        output[:,:,tile.y:tile.y+tile.height,tile.x:tile.x+tile.width] = computed
    return output
