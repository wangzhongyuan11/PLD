import dataclasses
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np

from acdc.contract import ContractError,NODE_NAMES
from acdc.integer import round_q24,conv_full
from acdc.model import create_synthetic_bundle,load_bundle,validate_acceptance
from acdc.graph import Graph,emulator
from acdc.planning import workload


class NumericsTest(unittest.TestCase):
    def test_signed_ties(self):
        np.testing.assert_array_equal(round_q24(np.array([-5,-3,3,5],np.int64)*(1<<23)),[-2,-2,2,2])

    def test_synthetic_guard_and_tamper(self):
        with tempfile.TemporaryDirectory() as d:
            root=create_synthetic_bundle(Path(d)/'model')
            with self.assertRaises(ContractError):load_bundle(root)
            model=load_bundle(root,allow_synthetic=True)
            self.assertEqual(model.layers[0].weight.shape,(16,3,3,3))
            self.assertEqual(model.layers[8].weight.shape,(64,192,3,3))
            with (root/'C01.w.bin').open('r+b') as f:f.write(b'\x08')
            with self.assertRaises(ContractError):load_bundle(root,allow_synthetic=True)

    def test_different_profile_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=create_synthetic_bundle(Path(d)/'m')
            p=root/'model_spec.json';s=json.loads(p.read_text());s['orientation']='coronal';p.write_text(json.dumps(s))
            with self.assertRaises(ContractError):load_bundle(root,allow_synthetic=True)

    def test_accumulator_overflow(self):
        x=np.full((1,1,1,1),127,np.int8);w=np.full((1,1,1,1),127,np.int8)
        with self.assertRaises(ContractError):conv_full(x,w,np.array([2**31-1],np.int32),np.array([1],np.int32),0,False)

    def test_entire_graph_matches_independent_reference(self):
        with tempfile.TemporaryDirectory() as d:
            model=load_bundle(create_synthetic_bundle(Path(d)/'m'),allow_synthetic=True)
            # Same complete topology and Co chunks, reduced spatial size for a fast regression.
            # A separate full-256 synthetic verification is exposed by the verification script.
            model=dataclasses.replace(model,input_shape=(1,3,16,16),head_shape=(1,4,16,16))
            x=np.random.default_rng(21).integers(-127,128,size=model.input_shape,dtype=np.int8)
            a=Graph(model).forward(x,capture=True);b=Graph(model,emulator()).forward(x,capture=True)
            self.assertEqual(tuple(a['captures']),NODE_NAMES)
            for node in NODE_NAMES:np.testing.assert_array_equal(a['captures'][node],b['captures'][node],err_msg=node)
            self.assertLess(int(a['head'].min()),0)

    def test_static_budget(self):
        p=workload()
        self.assertEqual(p['compatibility']['total_mac_count'],2674917376)
        self.assertEqual(p['compatibility']['total_packet_bytes'],32293264)
        self.assertEqual(p['engine_cycles'],110396848)
        self.assertEqual(p['candidate_128_issue_cycles'],21512192)

    def test_acceptance_is_not_just_a_pass_boolean(self):
        r=dict(status='passed',orientation='axial',integer_oracle_mismatches=0,verified_conv_layers=15,
               volumes=[dict(patient=str(i),phase=t,fp32_dice=[.9,.9,.9],int8_dice=[.895,.895,.895],
                             new_empty_nonempty_gt=0) for i in range(20) for t in ('ED','ES')])
        validate_acceptance(r)
        r['volumes'][0]['int8_dice']=[0.,0.,0.]
        with self.assertRaises(ContractError):validate_acceptance(r)
