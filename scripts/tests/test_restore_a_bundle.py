import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('restore_a_bundle',HERE.parent/'RESTORE_A_Models_and_Datasets.py')
restore=importlib.util.module_from_spec(spec);spec.loader.exec_module(restore)


class RestoreTests(unittest.TestCase):
    def setUp(self):
        parent=HERE/'test_tmp';parent.mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=parent)
        self.root=Path(self.temp.name)
        stream=io.BytesIO()
        with zipfile.ZipFile(stream,'w') as z:z.writestr('models/example.txt',b'synthetic fixture only')
        self.raw=stream.getvalue()
        pieces=[self.raw[:40],self.raw[40:100],self.raw[100:]]
        self.parts=[]
        for index,raw in enumerate(pieces,1):
            name=restore.ARCHIVE+'.%03d'%index
            (self.root/name).write_bytes(raw)
            self.parts.append(dict(name=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
        self.manifest=dict(schema=restore.SCHEMA,archive=dict(name=restore.ARCHIVE,bytes=len(self.raw),
            sha256=hashlib.sha256(self.raw).hexdigest()),parts=self.parts)
        self.write_manifest()

    def tearDown(self):self.temp.cleanup()
    def write_manifest(self):(self.root/restore.MANIFEST).write_text(json.dumps(self.manifest),encoding='utf-8')
    def run_restore(self,**kwargs):
        with contextlib.redirect_stdout(io.StringIO()):return restore.restore(self.root,**kwargs)

    def test_restores_exact_single_zip_without_extracting(self):
        output=self.run_restore()
        self.assertEqual(output.read_bytes(),self.raw)
        self.assertFalse((self.root/'models').exists())
        with zipfile.ZipFile(output) as z:self.assertIsNone(z.testzip())
    def test_verify_only_does_not_create_output(self):
        self.run_restore(verify_only=True)
        self.assertFalse((self.root/restore.ARCHIVE).exists())
    def test_missing_part_fails_before_writing(self):
        (self.root/self.parts[1]['name']).unlink()
        with self.assertRaises(ValueError):self.run_restore()
        self.assertFalse((self.root/restore.ARCHIVE).exists())
    def test_corrupted_part_leaves_no_output_or_temporary_file(self):
        part=self.root/self.parts[1]['name'];raw=part.read_bytes();part.write_bytes(bytes([raw[0]^1])+raw[1:])
        with self.assertRaisesRegex(ValueError,'SHA256'):self.run_restore()
        self.assertFalse((self.root/restore.ARCHIVE).exists())
        self.assertEqual(list(self.root.glob('*.assembling')),[])
    def test_combined_hash_checked_even_when_individual_parts_match(self):
        self.manifest['archive']['sha256']='f'*64;self.write_manifest()
        with self.assertRaisesRegex(ValueError,'Combined'):self.run_restore()
        self.assertFalse((self.root/restore.ARCHIVE).exists())
    def test_existing_output_is_preserved(self):
        output=self.root/restore.ARCHIVE;output.write_bytes(b'keep existing file')
        with self.assertRaises(FileExistsError):self.run_restore()
        self.assertEqual(output.read_bytes(),b'keep existing file')
    def test_path_traversal_and_reordered_parts_rejected(self):
        self.manifest['parts'][0]['name']='../escape';self.write_manifest()
        with self.assertRaises(ValueError):self.run_restore()
        self.assertFalse((self.root.parent/'escape').exists())
    def test_inconsistent_total_size_rejected(self):
        self.manifest['archive']['bytes']+=1;self.write_manifest()
        with self.assertRaises(ValueError):self.run_restore()

    def test_posix_publish_without_replacement(self):
        source=self.root/'source';destination=self.root/'destination'
        source.write_bytes(b'complete archive')
        with mock.patch.object(restore.os,'name','posix'):
            restore.publish_without_replacement(source,destination)
        self.assertEqual(destination.read_bytes(),b'complete archive')
        self.assertFalse(source.exists())

    def test_posix_concurrent_output_preserved_and_temporary_cleaned(self):
        publish=restore.publish_without_replacement
        real_link=restore.os.link
        def racing_publish(source,destination):
            def racing_link(src,dst):
                destination.write_bytes(b'concurrently created file')
                real_link(src,dst)
            with mock.patch.object(restore.os,'name','posix'), mock.patch.object(restore.os,'link',side_effect=racing_link):
                publish(source,destination)
        with mock.patch.object(restore,'publish_without_replacement',side_effect=racing_publish):
            with self.assertRaises(FileExistsError):self.run_restore()
        self.assertEqual((self.root/restore.ARCHIVE).read_bytes(),b'concurrently created file')
        self.assertEqual(list(self.root.glob('*.assembling')),[])


if __name__=='__main__':unittest.main(verbosity=2)
