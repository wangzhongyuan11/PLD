"""Portable interface-fixture regression; no NumPy, model or dataset required."""
from pathlib import Path
import decimal,hashlib,json,struct,tempfile,unittest
from make_acdc_contract_fixtures import build

class FixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name)/'pack';cls.manifest=build(cls.root)
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_manifest_and_no_overwrite(self):
        for row in self.manifest['members']:
            p=self.root/row['path'];self.assertEqual(p.stat().st_size,row['bytes']);self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),row['sha256'])
        with self.assertRaises(FileExistsError):build(self.root)

    def test_three_distinct_signed_heads(self):
        heads=[]
        for direction in ('axial','coronal','sagittal'):
            d=self.root/direction;spec=json.loads((d/'fixture_spec.json').read_text());self.assertTrue(spec['synthetic'])
            raw=(d/'expected_head_nchw.bin').read_bytes();heads.append(raw)
            self.assertEqual(len(raw),262144)
            for c,value in enumerate(spec['head_bias_int8']):self.assertEqual(raw[c*65536:(c+1)*65536],bytes([value&255])*65536)
        self.assertEqual(len(set(heads)),3)

    def test_q24_with_independent_decimal_rounding(self):
        c=json.loads((self.root/'q24_cases.json').read_text())
        actual=[int((decimal.Decimal(a)*c['multiplier_q24']/decimal.Decimal(1<<24)).to_integral_value(rounding=decimal.ROUND_HALF_EVEN)) for a in c['accumulator']]
        self.assertEqual(actual,c['expected'])

    def test_wide_convolution_and_chunk_mutants(self):
        for layer in ('C07','C08','C09'):
            d=self.root/'wide'/layer;c=json.loads((d/'case.json').read_text());ci=c['input_shape'][1];co=c['output_shape'][1]
            raw=(d/'parameters.bin').read_bytes();n=co*ci*9;weights=struct.unpack(f'{n}b',raw[:n]);bias=struct.unpack(f'<{co}i',raw[n:n+co*4]);mult=struct.unpack(f'<{co}i',raw[n+co*4:])
            inp=struct.unpack(f'{ci*36}b',(d/'input_chw.bin').read_bytes());oracle=[]
            for out in range(co):
                # Generic sparse integer convolution oracle, unrelated to the
                # fixture generator's explicit channel/ramp formula.
                terms=[(i//9,(i%9)//3,i%3,v) for i,v in enumerate(weights[out*ci*9:(out+1)*ci*9]) if v]
                for y in range(4):
                    for x in range(4):
                        acc=bias[out]+sum(v*inp[channel*36+(y+dy)*6+x+dx] for channel,dy,dx,v in terms)
                        q=int((decimal.Decimal(acc)*mult[out]/decimal.Decimal(1<<24)).to_integral_value(rounding=decimal.ROUND_HALF_EVEN))
                        oracle.append(max(0,min(127,q)))
            expected=(d/'expected_output_chw.bin').read_bytes();self.assertEqual(bytes(oracle),expected)
            self.assertEqual(len(set(expected[i*16:(i+1)*16] for i in range(co))),co)
            rebuilt=bytearray(len(expected))
            for j,p in enumerate(c['chunks']):
                start=p['start_output_channel'];count=p['output_channels'];golden=(d/f'chunk{j}_expected.bin').read_bytes()
                self.assertEqual(golden,expected[start*16:(start+count)*16]);rebuilt[start*16:(start+count)*16]=golden
            self.assertEqual(bytes(rebuilt),expected)
            a=c['chunks'][0]['output_channels']*16;b=c['chunks'][1]['start_output_channel']*16
            omitted=bytearray(rebuilt);omitted[:a]=bytes(a)
            swapped=bytearray(rebuilt);swapped[:a],swapped[b:b+a]=rebuilt[b:b+a],rebuilt[:a]
            shifted=rebuilt[-16:]+rebuilt[:-16]
            for bad in (omitted,swapped,shifted):self.assertNotEqual(bytes(bad),expected)

if __name__=='__main__':unittest.main()
