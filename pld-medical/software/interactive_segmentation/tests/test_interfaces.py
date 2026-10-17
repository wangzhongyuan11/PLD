import base64
import gzip
import http.client
import io
import json
import threading
import unittest
import numpy as np
from PIL import Image
from software.interactive_segmentation.contracts import Request,InputError
from software.interactive_segmentation.refinement import seeded_cut
from software.interactive_segmentation.service import SegmentationService,png_data
from software.interactive_segmentation.images import decode_image
from software.interactive_segmentation.skin_api import SkinAPI
from software.interactive_segmentation.camus_api import CamusAPI
from software.interactive_segmentation.server import make_server


def pixels(url):
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(url.split(',',1)[1]))))


class FixturePredictor:
    """Deterministic image-derived probabilities; no real weight loading."""
    def __init__(self): self.seen=[]
    def predict(self,image,model,variant,backend):
        self.seen.append(image.shape)
        p=image[...,0].astype(np.float32)/255
        if model=='skin': result=np.stack((1-p,p))
        else: result=np.stack((p*0+.1,p*0+.1,p*.8,(1-p)*.8))
        return result,{'model':model,'variant':variant,'backend':backend,'test_fixture':True}


class ContractTests(unittest.TestCase):
    def test_default_skin_and_camus_are_different(self):
        self.assertEqual(Request.parse('skin',{},40,30).threshold,.45)
        self.assertIsNone(Request.parse('camus',{},40,30).threshold)

    def test_invalid_and_ambiguous_parameters_are_rejected(self):
        bodies=[{'threshold':True},{'threshold':float('nan')},{'threshold':0},{'threshold':1},
                {'backend':'board'},{'target_class':True},{'roi':{'x0':10,'y0':0,'x1':3,'y1':3}},
                {'points':[{'x':40,'y':3,'label':'foreground'}]}, {'ground_truth':[]},
                {'points':[{'x':3,'y':2,'label':'foreground'},{'x':3,'y':2,'label':'background'}]}]
        for body in bodies:
            with self.subTest(body=body),self.assertRaises(InputError): Request.parse('skin',body,40,30)

    def test_roi_and_box_coordinates_are_native_and_half_open(self):
        body={'roi':{'x0':3,'y0':4,'x1':20,'y1':25},'box':{'x0':5,'y0':5,'x1':18,'y1':23},
              'points':[{'x':12,'y':10,'label':'foreground'}]}
        self.assertEqual(Request.parse('camus',body,40,30).points[0].x,12)
        body['points'][0]['x']=18
        with self.assertRaises(InputError): Request.parse('camus',body,40,30)

    def test_skin_cannot_select_camus_structure_or_candidate(self):
        for body in ({'variant':'candidate'},{'target_class':3}):
            with self.assertRaises(InputError): Request.parse('skin',body,40,30)


class CutTests(unittest.TestCase):
    def setUp(self):
        y,x=np.ogrid[:48,:72]
        self.truth=((x-34)/17)**2+((y-24)/11)**2<1
        self.image=np.full((48,72,3),35,np.uint8);self.image[self.truth]=190

    def test_box_extracts_object_rather_than_filling_rectangle(self):
        prior=np.where(self.truth,.85,.1)
        mask,meta=seeded_cut(self.image,prior,[],(7,4,65,44))
        self.assertGreater((mask&self.truth).sum()/(mask|self.truth).sum(),.95)
        self.assertLess(mask.sum(),.6*(58*40))
        self.assertFalse(mask[0,0]);self.assertTrue(meta['box_used_as_background_constraint'])

    def test_points_change_regions_and_preserve_foreground_background(self):
        # Flat model prior cannot identify the object; prompt appearance must.
        mask,meta=seeded_cut(self.image,np.full(self.truth.shape,.4),[(34,24,True),(2,2,False)])
        self.assertTrue(mask[24,34]);self.assertFalse(mask[2,2])
        self.assertGreater((mask&self.truth).sum()/(mask|self.truth).sum(),.9)
        self.assertGreater(mask.sum(),100)  # Not just a painted seed pixel.
        self.assertTrue(meta['appearance_used']);self.assertFalse(meta['network_prompt_conditioned'])

    def test_coarse_grid_seed_conflict_is_not_silently_overwritten(self):
        with self.assertRaises(InputError):
            seeded_cut(np.zeros((400,800,3),np.uint8),np.full((400,800),.5),[(0,0,True),(1,0,False)])

    def test_sparse_colour_samples_do_not_overturn_confident_model_regions(self):
        # A valid point can sample a local dark texture inside the object;
        # black padding is a valid background point but not all background
        # has that colour. Those two samples must not replace the whole mask.
        image=self.image.copy();image[22:27,32:37]=35;image[:5,:5]=0
        prior=np.where(self.truth,.98,.02)
        mask,_=seeded_cut(image,prior,[(34,24,True),(2,2,False)])
        self.assertTrue(mask[24,34]);self.assertFalse(mask[2,2])
        self.assertGreater((mask&self.truth).sum()/(mask|self.truth).sum(),.95)

    def test_cut_does_not_mutate_inputs(self):
        p=np.where(self.truth,.9,.1);before=p.copy();image=self.image.copy()
        seeded_cut(self.image,p,[(34,24,True)])
        np.testing.assert_array_equal(p,before);np.testing.assert_array_equal(self.image,image)


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.fake=FixturePredictor();self.service=SegmentationService(self.fake)
        self.image=np.repeat(np.tile(np.linspace(0,255,50,dtype=np.uint8),(32,1))[...,None],3,axis=2)

    def test_skin_threshold_changes_mask_monotonically(self):
        low=self.service.process('skin',self.image,{'threshold':.3})
        high=self.service.process('skin',self.image,{'threshold':.8})
        a=pixels(low['outputs']['target_mask_png'])>0;b=pixels(high['outputs']['target_mask_png'])>0
        self.assertGreater(a.sum(),b.sum());self.assertTrue(np.all(~b|a))
        self.assertIsNone(low['accuracy_metrics']);self.assertFalse(low['reference_available'])

    def test_camus_default_argmax_and_target_threshold(self):
        base=self.service.process('camus',self.image,{'target_class':2})
        changed=self.service.process('camus',self.image,{'target_class':2,'threshold':.9})
        self.assertIsNone(base['parameters']['threshold'])
        self.assertGreater(base['statistics']['target_pixels'],0)
        self.assertEqual(changed['statistics']['target_pixels'],0)
        self.assertEqual(set(np.unique(pixels(base['outputs']['labels_png']))),{2,3})

    def test_roi_is_cropped_before_inference_and_mapped_back(self):
        r={'x0':12,'y0':4,'x1':40,'y1':24}
        result=self.service.process('skin',self.image,{'roi':r})
        labels=pixels(result['outputs']['labels_png'])
        self.assertEqual(self.fake.seen,[(20,28,3)])
        self.assertEqual(labels.shape,(32,50));self.assertFalse(labels[:4].any());self.assertFalse(labels[:,:12].any())
        self.assertEqual(result['statistics']['processed_pixels'],560)

    def test_local_points_and_box_map_to_original_roi_pixels(self):
        result=self.service.process('skin',self.image,{'roi':{'x0':10,'y0':4,'x1':45,'y1':29},
            'box':{'x0':18,'y0':7,'x1':42,'y1':27},'points':[{'x':25,'y':14,'label':'foreground'},{'x':20,'y':18,'label':'background'}]})
        mask=pixels(result['outputs']['target_mask_png'])>0
        self.assertTrue(mask[14,25]);self.assertFalse(mask[18,20]);self.assertFalse(mask[:7].any());self.assertFalse(mask[:,42:].any())
        self.assertEqual(result['operation'],'prompt_refinement')


class ImageTests(unittest.TestCase):
    def test_png_roundtrip_and_invalid_base64(self):
        image=np.full((7,11,3),120,np.uint8)
        np.testing.assert_array_equal(decode_image('skin',png_data(image)),image)
        with self.assertRaises(InputError): decode_image('skin','data:image/png;base64,!!!')

    def test_nifti_transpose_uses_same_display_convention(self):
        import nibabel as nib
        array=np.arange(35,dtype=np.uint8).reshape(5,7)
        raw=gzip.compress(nib.Nifti1Image(array,np.eye(4)).to_bytes())
        encoded=base64.b64encode(raw).decode()
        np.testing.assert_array_equal(decode_image('camus',encoded,'frame.nii.gz')[...,0],array.T)
        with self.assertRaises(InputError): decode_image('skin',encoded,'frame.nii.gz')


class HTTPTests(unittest.TestCase):
    def call(self,server,path,body=None,origin=None):
        connection=http.client.HTTPConnection(*server.server_address,timeout=15)
        headers={'Content-Type':'application/json'}
        if origin: headers['Origin']=origin
        connection.request('POST' if body is not None else 'GET',path,
            body=None if body is None else json.dumps(body),headers=headers)
        response=connection.getresponse();data=json.loads(response.read());connection.close()
        return response.status,data

    def test_two_services_reject_each_others_routes_and_keep_own_defaults(self):
        for model,api in [('skin',SkinAPI),('camus',CamusAPI)]:
            with self.subTest(model=model),make_server(model,0,api(SegmentationService(FixturePredictor()))) as server:
                thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
                try:
                    status,data=self.call(server,f'/api/{model}/capabilities')
                    self.assertEqual(status,200);self.assertEqual(data['model'],model)
                    self.assertEqual(self.call(server,'/api/'+('camus' if model=='skin' else 'skin')+'/capabilities')[0],404)
                    self.assertEqual(self.call(server,f'/api/{model}/capabilities',origin='https://unrelated.example')[0],403)
                    image=png_data(np.full((20,30,3),180,np.uint8))
                    status,data=self.call(server,f'/api/{model}/predict',{'image':image,'parameters':{}})
                    self.assertEqual(status,200);self.assertEqual(data['model'],model)
                    self.assertEqual(data['parameters']['threshold'],.45 if model=='skin' else None)
                    status,data=self.call(server,f'/api/{model}/predict',{'image':image,'parameters':{'threshold':2}})
                    self.assertEqual(status,400);self.assertIn('error',data)
                finally: server.shutdown();thread.join(timeout=3)


if __name__=='__main__': unittest.main()
