"""Numerical and failure contracts for the PS-only acceleration paths."""
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from camus.result import overlay_rgb,segmentation_rgb,label_display_png,CLASS_COLORS,native_labels_from_head
from camus.png import save_png,save_views
from camus.postprocess import apply_island5_hole1
from camus.contract import tensor_tile_shape


class OptimizationTests(unittest.TestCase):
    def test_nonconv_tiles_reduce_transactions_within_bank_limits(self):
        from camus.contract import shape
        for mode, channels, side, split, expected in (
            (2,8,128,0,(32,128)), (2,16,64,0,(32,64)),
            (2,32,32,0,(32,32)), (4,192,64,64,(10,34)),
            (4,96,128,32,(26,26)), (4,48,256,16,(26,52)),
        ):
            th,tw=tensor_tile_shape(mode,channels,side,side,split)
            self.assertEqual((th,tw),expected)
            ih,iw=(th*2,tw*2) if mode==2 else (th,tw)
            self.assertLessEqual(shape(mode,ih,iw,channels,split=split)['output_occupancy'],8192)

    def test_overlay_exhaustive_gray_class_lut_matches_float_formula(self):
        gray=np.tile(np.arange(256,dtype=np.uint8),(4,1))
        labels=np.repeat(np.arange(4,dtype=np.uint8)[:,None],256,axis=1)
        rgb=np.repeat(gray[:,:,None],3,axis=2)
        expected=np.rint(.55*rgb.astype(np.float32)+.45*CLASS_COLORS[labels].astype(np.float32)).astype(np.uint8)
        np.testing.assert_array_equal(np.array(overlay_rgb(gray,labels)),expected)

    def test_png_roundtrip_l_rgb_and_one_pixel(self):
        rng=np.random.default_rng(24)
        with tempfile.TemporaryDirectory() as tmp:
            for shape in ((1,1),(1,1,3),(37,43),(37,43,3)):
                a=rng.integers(0,256,shape,dtype=np.uint8);path=Path(tmp)/'x.png'
                save_png(a,path)
                with Image.open(path) as image:np.testing.assert_array_equal(np.array(image),a)

    def test_png_writer_exception_is_propagated_before_zip(self):
        with tempfile.TemporaryDirectory() as tmp,patch('camus.png.save_png',side_effect=OSError('full')):
            a=np.zeros((2,2),np.uint8)
            with self.assertRaisesRegex(OSError,'full'):save_views(Path(tmp),a,a,a)
            self.assertFalse((Path(tmp)/'result.zip').exists())

    def test_light_preview_precedes_full_views_and_preserves_export_pixels(self):
        rng=np.random.default_rng(42)
        gray=rng.integers(0,256,(650,900),dtype=np.uint8)
        labels=rng.integers(0,4,gray.shape,dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);called=[]
            def ready():
                called.append(True)
                self.assertFalse((root/'overlay.png').exists())
                self.assertFalse((root/'label.png').exists())
                for name in ('preview_original.jpg','preview_overlay.jpg'):
                    with Image.open(root/name) as image:self.assertLessEqual(max(image.size),512)
            save_views(root,gray,labels,labels,on_preview=ready)
            self.assertEqual(called,[True])
            for name,expected in [('original.png',gray),('overlay.png',overlay_rgb(gray,labels)),
                                  ('label.png',label_display_png(labels)),('segmentation.png',segmentation_rgb(labels))]:
                with Image.open(root/name) as image:np.testing.assert_array_equal(np.array(image),expected)

    def test_holes_with_foreign_classes_remain_and_global_ratio_is_preserved(self):
        a=np.ones((100,100),np.uint8)
        a[10:12,10:12]=0;a[20:22,20:22]=2;a[30:32,30:32]=3
        result=apply_island5_hole1(a)
        self.assertEqual(result.hole_pixels_filled,4)
        self.assertTrue(np.all(result.labels[10:12,10:12]==1))
        np.testing.assert_array_equal(result.labels[20:22,20:22],a[20:22,20:22])
        b=np.ones((30,30),np.uint8);b[3:6,3:6]=0;b[20:23,20:23]=0
        result=apply_island5_hole1(b)
        self.assertFalse(result.hole_applied)
        np.testing.assert_array_equal(result.labels,b)


class NativeOptimizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sys
        if sys.platform!='linux':raise unittest.SkipTest('native library is built on the PYNQ Linux board')
        from camus.native import load_native
        cls.native=load_native()

    def test_native_restore_random_shapes_ties_and_extremes(self):
        rng=np.random.default_rng(842)
        for h,w in ((1,1),(1,513),(513,1),(100,300),(487,590),(616,748),(257,259)):
            head=rng.integers(-127,128,(1,4,256,256),dtype=np.int8)
            head[:,2]=head[:,1] # Exact ties still keep the lower class.
            expected=native_labels_from_head(head,h,w,.19834566491795338)
            np.testing.assert_array_equal(self.native.restore(head,h,w,.19834566491795338),expected)

    def test_native_four_view_pixels_match_existing_renderers(self):
        rng=np.random.default_rng(914)
        gray=rng.integers(0,4096,(67,89),dtype=np.uint16)
        labels=rng.integers(0,4,gray.shape,dtype=np.uint8)
        processed=rng.integers(0,4,gray.shape,dtype=np.uint8)
        rendered=self.native.render_views(gray,labels,processed)
        expected={'label_raw.png':label_display_png(labels),
                  'label.png':label_display_png(processed),
                  'segmentation.png':segmentation_rgb(processed),
                  'overlay.png':overlay_rgb(gray,processed)}
        for name,image in expected.items():
            np.testing.assert_array_equal(rendered[name],np.asarray(image),err_msg=name)

    def test_native_preprocess_matches_frozen_float_pipeline(self):
        from camus.preprocess import preprocess
        rng=np.random.default_rng(451)
        with tempfile.TemporaryDirectory() as tmp:
            for shape,dtype in (((487,590),np.float32),((616,748),np.uint16),((1,257),np.float64)):
                values=rng.normal(120,100,shape).astype(dtype)
                path=Path(tmp)/'frame.npy';np.save(path,values)
                expected=preprocess(path);actual=self.native.preprocess(path)
                np.testing.assert_array_equal(actual.normalized_float32,expected.normalized_float32)
                np.testing.assert_array_equal(actual.canvas_float32,expected.canvas_float32)
                np.testing.assert_array_equal(actual.input_int8,expected.input_int8)
                self.assertEqual(actual.letterbox_meta,expected.letterbox_meta)
                self.assertEqual((actual.clipped_low,actual.clipped_high),
                                 (expected.clipped_low,expected.clipped_high))

    def test_native_copy_strides_reject_invalid_and_gather_halo_scatter(self):
        from reference.tiling import Tile,input_tile
        source=np.arange(60,dtype=np.int8).reshape(1,3,4,5)
        view=source[:,:,::-1,::2];dst=np.empty(view.shape,np.int8)
        self.native.copy(dst,view);np.testing.assert_array_equal(dst,view)
        source[0,0,0,0]=-128
        with self.assertRaises(ValueError):self.native.copy(np.empty_like(source),source)
        source[0,0,0,0]=0
        tile=Tile(0,0,2,3);expected=input_tile(source,tile,3,1);dst=np.empty(expected.size,np.int8)
        self.native.gather(dst,source,tile,3,1);np.testing.assert_array_equal(dst.reshape(expected.shape),expected)
        output=np.zeros_like(source);part=source[:,:,:2,:3].copy()
        self.native.scatter(output,part,tile);np.testing.assert_array_equal(output[:,:,:2,:3],part)
        guard=np.full(64,np.uint8(0xa5));guard[:part.size]=part.view(np.uint8).ravel()
        output.fill(0)
        self.assertEqual(self.native.scatter_checked(output,guard,tile,part.size),0)
        np.testing.assert_array_equal(output[:,:,:2,:3],part)
        guard[-1]=0
        output.fill(0)
        self.assertEqual(self.native.scatter_checked(output,guard,tile,part.size),1)
        self.assertEqual(np.count_nonzero(output),0)

    def test_native_cam2_config_registers_and_bounded_timeout(self):
        import ctypes as ct
        from types import SimpleNamespace
        from camus.contract import shape
        core=np.zeros(1024,np.uint32);dma=np.zeros(1024,np.uint32)
        self.native.bind(SimpleNamespace(array=core),SimpleNamespace(mmio=SimpleNamespace(array=dma)))
        self.native.configure(shape(0,10,10,192,32,3,relu=True),13)
        self.assertEqual(int(core[0x24//4]),192);self.assertEqual(int(core[0x5c//4]),1)
        self.assertEqual(int(core[0x34//4]),13)
        result=(ct.c_uint32*8)()
        code=self.native.wait_fn(core.ctypes.data,core.nbytes,2,13,1,16,1_000_000,result)
        self.assertEqual(code,9)

    def test_native_postprocess_matches_scipy_pixels_and_all_metadata(self):
        from dataclasses import asdict
        rng=np.random.default_rng(43)
        samples=[rng.integers(0,4,(h,w),dtype=np.uint8) for h,w in ((1,1),(1,50),(50,1),(79,103))]
        a=np.ones((100,100),np.uint8);a[10:12,10:12]=0;a[20:22,20:22]=2;a[30:32,30:32]=3;samples.append(a)
        a=np.zeros((100,100),np.uint8);a[1:75,1:75]=1;a[90:92,90:92]=1;samples.append(a)
        samples.extend([np.full((10,10),v,np.uint8) for v in range(4)])
        for labels in samples:
            reference=apply_island5_hole1(labels);actual=self.native.postprocess(labels)
            np.testing.assert_array_equal(actual.labels,reference.labels)
            a,b=asdict(actual),asdict(reference);a.pop('labels');b.pop('labels');self.assertEqual(a,b)
