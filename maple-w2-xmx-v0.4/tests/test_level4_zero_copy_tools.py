import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'tools'/f'{name}.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
app=load('apply_level4_zero_copy');audit=load('analyze_level4_zero_copy')

class ApplyTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.base=Path(self.tmp.name);self.root=self.base/'module';self.overlay=self.base/'overlay'
        for p in (self.root,self.overlay):p.mkdir()
        for n in ('include/maple_moe.hpp','src/maple_moe.cpp','src/maple_level4.cpp'):
            p=self.root/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'untouched')
        self.manifest={'files':[]}
    def item(self,name,old,new):
        p=self.overlay/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(new)
        if old is not None:
            p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(old)
        self.manifest['files'].append({'path':name,'old_sha256':app.digest(old) if old is not None else None,'new_sha256':app.digest(new)})
    def test_dry_apply_and_idempotence(self):
        self.item('new.hpp',None,b'new');self.item('src/maple_level4.cpp',b'untouched',b'updated')
        self.assertEqual(app.apply(self.root,self.overlay,self.manifest)['count'],2)
        self.assertFalse((self.root/'new.hpp').exists())
        result=app.apply(self.root,self.overlay,self.manifest,True)
        self.assertEqual((self.root/'src/maple_moe.cpp').read_bytes(),b'untouched')
        self.assertEqual((Path(result['backup'])/'src/maple_level4.cpp').read_bytes(),b'untouched')
        self.assertEqual(app.apply(self.root,self.overlay,self.manifest,True)['count'],0)
    def test_unknown_edit_no_partial_write(self):
        self.item('new.hpp',None,b'new');self.item('src/maple_level4.cpp',b'untouched',b'updated')
        (self.root/'src/maple_level4.cpp').write_bytes(b'local change')
        with self.assertRaises(ValueError):app.apply(self.root,self.overlay,self.manifest,True)
        self.assertFalse((self.root/'new.hpp').exists())
    def test_cmake_preserves_user_changes(self):
        self.item('cmake/level4.cmake',b'old',b'new')
        (self.root/'cmake/level4.cmake').write_bytes(b'custom CMake\n')
        app.apply(self.root,self.overlay,self.manifest,True)
        p=(self.root/'cmake/level4.cmake').read_text();self.assertTrue(p.startswith('custom CMake\n'));self.assertIn('level4_zerocopy.cmake',p)
        self.assertEqual(app.apply(self.root,self.overlay,self.manifest,True)['count'],0)
    def test_corrupt_overlay(self):
        self.item('new.hpp',None,b'new');(self.overlay/'new.hpp').write_bytes(b'corrupt')
        with self.assertRaises(ValueError):app.plan(self.root,self.overlay,self.manifest)
    def test_traversal_and_symlink(self):
        for n in ('../outside','/absolute','C:/x','a\\b'):
            with self.assertRaises(ValueError):app.inside(self.root,n)
        try:(self.root/'link').symlink_to(self.overlay,target_is_directory=True)
        except OSError:return
        with self.assertRaises(ValueError):app.inside(self.root,'link/x')

class AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        (self.root/'SCOPE.txt').write_text('Real Vulkan->LevelZero/SYCL Level4->Vulkan\nentry=advance\n')
        self.samples=[{'repeat':0,'inbound_host_us':10,'sycl_host_us':20,'acquire_submit_us':5,'total_host_us':35,'execute_wall_us':36,'gpu_sycl_span_us':18,'boundary_d2d_bytes':0,'imports':1,'exports':1}]
        self.nums=[{'repeat':i,'output':n,'max_abs':0,'nmse':0,'max_abs_over_rms':0,'finite':1,'bitwise_equal':1} for i in (0,1) for n in ('hidden','q','k','v')]
    def save(self):
        for name,rows in [('samples.csv',self.samples),('numerical.csv',self.nums)]:
            with (self.root/name).open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    def test_valid_synthetic_parser_input(self):
        self.save();r=audit.analyze(self.root);self.assertEqual(r['sample_count'],1);self.assertEqual(r['checked_outputs'],8)
    def test_copy_rejected(self):
        self.samples[0]['boundary_d2d_bytes']=256;self.save()
        with self.assertRaises(ValueError):audit.analyze(self.root)
    def test_nan_rejected(self):
        self.nums[1]['nmse']='nan';self.save()
        with self.assertRaises(ValueError):audit.analyze(self.root)
    def test_missing_numerical(self):
        self.nums.pop();self.save()
        with self.assertRaises(ValueError):audit.analyze(self.root)
    def test_reimport_rejected(self):
        self.samples.append(dict(self.samples[0],repeat=1,imports=2));self.save()
        with self.assertRaises(ValueError):audit.analyze(self.root)
    def test_timer_mismatch_rejected(self):
        self.samples[0]['total_host_us']=100;self.save()
        with self.assertRaises(ValueError):audit.analyze(self.root)
if __name__=='__main__':unittest.main()
