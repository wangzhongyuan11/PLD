import unittest
from acdc.contract import ContractError
from acdc.exchange import translate


class ExchangeTest(unittest.TestCase):
    def request(self):
        return dict(schema='acdc-board-request-v1',architecture_id='ACDC25D-B16-P3-v1',model_id='acdc_axial_v1',
            model_spec_sha256='a'*64,head_scale=.125,input_shape=[1,3,256,256],output_shape=[1,4,256,256],
            dtype='int8',layout='NCHW',input_scale=1/127,zero_point=0,required_backend='pynq_pl',synthetic=False,
            request_id='id',volume_id='volume',run_id='run',phase='ED',orientation='axial',center_index=0,
            neighbor_indices=[0,0,1],slice_count=3,geometry_sha256='b'*64)

    def test_mapping_preserves_identity_and_geometry_hash_kind(self):
        a=self.request();b=translate(a,'a'*64,.125,'c'*64)
        self.assertEqual(b['model_sha256'],a['model_spec_sha256'])
        self.assertEqual(b['total_slices'],3);self.assertEqual(b['request_id'],'id')
        self.assertEqual(b['geometry_sha256'],a['geometry_sha256'])
        self.assertEqual(b['geometry_digest_kind'],'A sidecar file SHA256')

    def test_other_direction_or_scale_is_not_silently_adapted(self):
        for change in (dict(orientation='coronal'),dict(head_scale=.2),dict(zero_point=128),dict(synthetic=True)):
            with self.assertRaises(ContractError):translate(dict(self.request(),**change),'a'*64,.125,'c'*64)
