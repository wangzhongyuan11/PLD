import math

import numpy as np
import pytest

from acdc.contract import SHAPES
from acdc.optimization.layout import (BankGeometry, pack_feature, pack_weights,
                                      unpack_feature, unpack_weights, pack_parameter_groups)
from acdc.optimization.planner import PROFILES, plan


def test_external_model_layout_roundtrip_all_layers():
    rng = np.random.default_rng(2516)
    for ci, co, h, k in SHAPES:
        w = rng.integers(-127, 128, (co, ci, k, k), dtype=np.int16).astype(np.int8)
        assert np.array_equal(unpack_weights(pack_weights(w), co, ci), w)
        # Include complete production spatial sizes, first/head tails.
        a = rng.integers(-127, 128, (co, h, h), dtype=np.int16).astype(np.int8)
        packed = pack_feature(a)
        assert packed.flags.c_contiguous
        assert np.array_equal(unpack_feature(packed, co), a)
        if co % 8:
            assert np.all(packed[-1, :, :, co % 8:] == 0)


@pytest.mark.parametrize('p,words', [(4, 4608), (6, 3584)])
def test_physical_bank_coordinates_and_dense_payload(p, words):
    rng = np.random.default_rng(8412)
    geometry = BankGeometry(p, 192, 26, 26, words)
    a = rng.integers(-127, 128, (192, 26, 26), dtype=np.int16).astype(np.int8)
    banks = geometry.encode(a)
    seen = set()
    for block in range(math.ceil(geometry.ci/8)):
        for y in range(geometry.height):
            for x in range(geometry.width):
                address = geometry.address(block, y, x)
                assert address not in seen
                seen.add(address)
                assert np.array_equal(banks[address], a[block*8:block*8+8, y, x])
    # Full32 input for the C08 special path, no stored out-of-frame padding.
    BankGeometry(p, 128, 32, 32, words)
    with pytest.raises(ValueError, match='capacity'):
        BankGeometry(p, 192, 64, 64, words)


def test_layout_rejects_bad_dtype_and_forbidden_values():
    with pytest.raises(ValueError):
        pack_feature(np.zeros((3, 4, 4), dtype=np.float32))
    with pytest.raises(ValueError):
        pack_weights(np.full((8, 8, 3, 3), -128, dtype=np.int8))


def test_parameter_dma_packets_preserve_original_weights_bias_and_scale():
    rng=np.random.default_rng(1608)
    for ci,co,h,k in SHAPES:
        w=rng.integers(-127,128,(co,ci,k,k),dtype=np.int16).astype(np.int8)
        b=rng.integers(-100000,100000,co,dtype=np.int32)
        m=rng.integers(1,1<<30,co,dtype=np.int32)
        packets=pack_parameter_groups(w,b,m)
        assert len(packets)==math.ceil(co/8)
        blocks=[]
        for group,packet in enumerate(packets):
            assert len(packet)==64+math.ceil(ci/8)*k*k*64
            bp=np.frombuffer(packet[:32],dtype='<i4');mp=np.frombuffer(packet[32:64],dtype='<i4')
            count=min(8,co-group*8)
            assert np.array_equal(bp[:count],b[group*8:group*8+count])
            assert np.array_equal(mp[:count],m[group*8:group*8+count])
            assert np.all(bp[count:]==0) and np.all(mp[count:]==1)
            blocks.append(np.frombuffer(packet[64:],dtype=np.int8).reshape(math.ceil(ci/8),k,k,8,8))
        assert np.array_equal(unpack_weights(np.stack(blocks),co,ci),w)


@pytest.mark.parametrize('name', ['P256', 'P384'])
def test_schedule_covers_every_output_once_and_fits_physical_banks(name):
    result = plan(name)
    profile = PROFILES[name]
    assert result['board_measured'] is False
    for layer, (_, _, h, _) in zip(result['layers'], SHAPES):
        coverage = np.zeros((h, h), dtype=np.uint8)
        for tile in layer['tiles']:
            y, x, oh, ow = (tile[n] for n in ('y', 'x', 'height', 'width'))
            coverage[y:y+oh, x:x+ow] += 1
            assert oh*ow <= profile.output_words
            BankGeometry(profile.p, layer['ci'], oh+layer['kernel']-1,
                         ow+layer['kernel']-1, profile.input_words)
        assert np.all(coverage == 1)
        # Physical head output holds eight lanes, not four external channels.
        expected_output = math.ceil(layer['co']/8)*8*h*h
        assert layer['payload_bytes']['output'] == expected_output
