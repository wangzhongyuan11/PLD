"""Command/AXIS regression vectors: 15 real CAMUS tile goldens plus boundaries."""
from pathlib import Path
import json
import numpy as np
from .model import load_model, load_golden
from .device import output_channel_chunks
from reference.tiling import Tile, input_tile
from .integer import conv2d_int32, requantize_int8, rescale_int8, maxpool2_int8
from .integer import q24_lut as lut


class Writer:
    def __init__(self,stream):
        self.stream,self.job,self.frame,self.runs,self.loads,self.errors=stream,0,0x12345678,0,0,0

    def line(self,op,*args):
        fields=[op,*args]
        fields += [0]*(8-len(fields))
        self.stream.write(" ".join(f"{int(value):x}" for value in fields)+"\n")

    def write(self,addr,value,strobe=15,response=0):
        self.line(0,addr,value,strobe,response)

    def configure(self,mode,h,w,ci,co=0,k=1,split=0):
        for addr,value in ((0x18,mode),(0x1c,h),(0x20,w),(0x24,ci),(0x28,co),(0x2c,k),(0x30,split),(0x34,self.frame)):
            self.write(addr,value)

    def start(self,command):
        self.job+=1
        self.write(0x14,command)
        self.write(0x38,self.job)
        self.write(8,1)

    def packet(self,op,data):
        data=bytes(data)
        for pos in range(0,len(data),8):
            part=data[pos:pos+8]
            self.line(op,int.from_bytes(part,"little"),(1<<len(part))-1,int(pos+len(part)==len(data)))

    def wait(self,error=0,received=0,sent=0):
        if error: self.errors+=1
        self.line(2,error,self.job,self.frame,received,sent)

    def load(self,command,data):
        self.start(command)
        self.packet(1,data)
        self.wait(received=len(data))
        self.loads+=1

    def run(self,expected):
        self.packet(3,expected.tobytes())
        self.start(4)
        self.wait(sent=expected.size)
        self.runs+=1

    def reset(self):
        self.write(8,2)
        self.line(5,0x0c,0,0x7f)


def main():
    model = load_model()
    root = Path('out/camus/vectors'); root.mkdir(parents=True, exist_ok=True)
    descriptions = []
    with (root / 'accel.txt').open('w', encoding='ascii') as stream:
        writer = Writer(stream)
        writer.line(5, 0, 0x43414d32, 0xffffffff)
        writer.line(5, 4, 0x00020000, 0xffffffff)

        def configure(mode, h, w, ci, co=0, k=1, split=0, relu=False):
            writer.configure(mode, h, w, ci, co, k, split)
            writer.write(0x5c, int(relu))

        def conv(data, weight, bias, mult, relu, expected=None, repeat=False):
            co, ci, k, _ = weight.shape
            configure(0, *data.shape[2:], ci, co, k, relu=relu)
            writer.load(1, weight.tobytes() + bias.astype('<i4').tobytes() + mult.astype('<i4').tobytes())
            writer.load(2, data.tobytes())
            if expected is None:
                expected = requantize_int8(conv2d_int32(data, weight, bias, padding=0), mult)
                if relu: expected = np.maximum(expected, np.int8(0))
            writer.run(expected)
            if repeat: writer.run(expected)

        for layer, row in zip(model.layers, model.spec['layers']):
            folder = model.root / layer.directory / 'tile'
            data = np.fromfile(folder / 'input.bin', np.int8).reshape(row['tile']['input']['shape'])
            expected = np.fromfile(folder / 'output_golden.bin', np.int8).reshape(row['tile']['output_golden']['shape'])
            p = layer.padding
            data = np.pad(data, ((0, 0), (0, 0), (p, p), (p, p)))
            offset = 0
            for co in output_channel_chunks(data.shape[1], layer.weight.shape[0], layer.kernel):
                conv(data, layer.weight[offset:offset+co], layer.bias[offset:offset+co],
                     layer.multipliers[offset:offset+co], layer.relu,
                     expected[:, offset:offset+co], layer.index == 0)
                offset += co
            descriptions.append(layer.name)
            if layer.index == 7:
                writer.start(4)
                writer.write(0x5c, 0, response=2)  # rejected while compute is active
                writer.line(6, 20)
                writer.reset()
        rng = np.random.default_rng(20261015)
        for ci, co, k in ((191, 32, 3), (128, 56, 3), (5, 9, 3), (1, 1, 1)):
            data = rng.integers(-127, 128, (1, ci, 4+k-1, 5+k-1), dtype=np.int8)
            weight = rng.integers(-3, 4, (co, ci, k, k), dtype=np.int8)
            bias = rng.integers(-100, 101, co, dtype=np.int32)
            mult = np.full(co, 1 << 20, dtype=np.int32)
            conv(data, weight, bias, mult, False)
            conv(data, weight, bias, mult, True)
        # Exact positive/negative half ties, symmetric saturation, negative head.
        data = np.array([-127, -5, -3, -1, 1, 3, 5, 127], np.int8).reshape(1, 1, 1, 8)
        for m in (1 << 23, (1 << 31)-1):
            conv(data, np.ones((1, 1, 1, 1), np.int8), np.zeros(1, np.int32),
                 np.array([m], np.int32), False)
        # Real spatial tiles, including exact 32768-byte input-bank and
        # 8192-byte output-bank limits, full width 256, and bottom/right halo.
        for index, co, tile in ((0,16,Tile(0,0,16,256)), (7,56,Tile(0,0,16,32)),
                                (8,32,Tile(48,32,16,32)), (14,4,Tile(0,0,32,256))):
            layer=model.layers[index]
            data=input_tile(load_golden(model,layer,'input_full'),tile,layer.kernel,layer.padding)
            expected=load_golden(model,layer,'output_full_golden')[:,:co,tile.y:tile.y+tile.height,tile.x:tile.x+tile.width]
            conv(data,layer.weight[:co],layer.bias[:co],layer.multipliers[:co],layer.relu,expected)
        # Complete 192-channel tensor and odd split expose counters and branch banks.
        for mode, c, split in ((1, 192, 0), (2, 128, 0), (3, 191, 0), (4, 192, 64), (4, 13, 5)):
            first_c = split if mode == 4 else c
            data = rng.integers(-127, 128, (1, first_c, 4, 6), dtype=np.int8)
            table = lut(11743909)
            configure(mode, 4, 6, c, split=split)
            if mode == 4:
                second = rng.integers(-127, 128, (1, c-split, 2, 3), dtype=np.int8)
                expected = np.concatenate((rescale_int8(data, 11743909), np.repeat(np.repeat(second, 2, 2), 2, 3)), axis=1)
                payload = data.tobytes() + second.tobytes(); tables = table.tobytes() + lut(1 << 24).tobytes()
            else:
                transformed = maxpool2_int8(data) if mode == 2 else np.repeat(np.repeat(data, 2, 2), 2, 3) if mode == 3 else data
                expected = rescale_int8(transformed, 11743909)
                payload, tables = data.tobytes(), table.tobytes()
            writer.load(3, tables); writer.load(2, payload); writer.run(expected)
        data=load_golden(model,model.layers[1],'output_full_golden')[:,:,:32,:256]
        expected=load_golden(model,model.layers[2],'input_full')[:,:,:16,:128]
        configure(2,32,256,16)
        writer.load(3,lut(1<<24).tobytes());writer.load(2,data.tobytes());writer.run(expected)
        skip=load_golden(model,model.layers[5],'output_full_golden')[:,:,:4,:64]
        deep=load_golden(model,model.layers[7],'output_full_golden')[:,:,:2,:32]
        configure(4,4,64,192,split=64)
        writer.load(3,lut(11743909).tobytes()+lut(1<<24).tobytes())
        writer.load(2,skip.tobytes()+deep.tobytes())
        writer.run(load_golden(model,model.layers[8],'input_full')[:,:,:4,:64])
        # A reset in a long dot product must flush all ready tags and results.
        configure(0, 10, 10, 128, 56, 3)
        writer.start(4); writer.wait(error=4); writer.reset()
        for mode, ci, co, k in ((0, 191, 33, 3), (0, 193, 1, 1), (5, 1, 1, 1), (6, 1, 1, 3)):
            configure(mode, 8, 8, ci, co, k)
            writer.start(1); writer.wait(error=1); writer.reset()
        # Short/overlong packets and bad keep are never accepted as complete input.
        configure(1, 2, 3, 1)
        writer.start(2); writer.packet(1, bytes(5)); writer.wait(error=2, received=5); writer.reset()
        configure(1, 2, 3, 1)
        writer.start(2); writer.line(1, 0, 0x55, 1); writer.wait(error=3); writer.reset()
        # ReLU register is byte-strobed; higher bytes cannot change bit 0.
        writer.write(0x5c, 1); writer.write(0x5c, 0, strobe=2)
        writer.line(5, 0x5c, 1, 0xffffffff); writer.write(0x5c, 0)
        writer.line(10, writer.runs, writer.loads, writer.errors)
        (root / 'coverage.json').write_text(json.dumps(dict(real_layers=descriptions, runs=writer.runs,
            loads=writer.loads, expected_errors=writer.errors, seed=20261015), indent=2)+'\n')
    print(f'CAMUS_VECTORS layers=15 runs={writer.runs} loads={writer.loads} errors={writer.errors}')


if __name__ == '__main__':
    main()
