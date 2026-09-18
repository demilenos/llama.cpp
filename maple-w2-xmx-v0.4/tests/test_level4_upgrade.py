import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

class UpgradeTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.base=Path(self.tmp.name)
        self.payload=self.base/'payload';self.root=self.base/'module'
        (self.payload/'tools').mkdir(parents=True);(self.root/'include').mkdir(parents=True);(self.root/'src').mkdir()
        original=Path(__file__).resolve().parents[1]/'tools/apply_level4_upgrade.py'
        shutil.copy(original,self.payload/'tools'/original.name)
        (self.payload/'include').mkdir();(self.payload/'include/new.hpp').write_text('new')
        self.metadata={'files':{'include/new.hpp':hashlib.sha256(b'new').hexdigest()}}
        (self.payload/'LEVEL4_FILES.json').write_text(json.dumps(self.metadata))
        (self.root/'include/maple_moe.hpp').write_text('gate_tokens_per_tile MoeWorkspace enqueue_moe')
        (self.root/'include/maple_w2a8.hpp').write_text('enqueue_a8_quant A8Mode')
        (self.root/'src/maple_moe.cpp').write_text('USER MERGED enqueue_moe DO NOT REPLACE')
        (self.root/'CMakeLists.txt').write_text('add_library(maple-w2-xmx STATIC src/maple_moe.cpp)\n')
    def tearDown(self):self.tmp.cleanup()
    def run_apply(self,*args):
        return subprocess.run([sys.executable,str(self.payload/'tools/apply_level4_upgrade.py'),'--root',str(self.root),*args],capture_output=True,text=True)
    def test_dry_run(self):
        self.assertEqual(self.run_apply().returncode,0);self.assertFalse((self.root/'include/new.hpp').exists())
    def test_additive_idempotent(self):
        self.assertEqual(self.run_apply('--apply').returncode,0);self.assertEqual(self.run_apply('--apply').returncode,0)
        self.assertEqual((self.root/'CMakeLists.txt').read_text().count('include(cmake/level4.cmake)'),1)
        self.assertEqual((self.root/'src/maple_moe.cpp').read_text(),'USER MERGED enqueue_moe DO NOT REPLACE')
    def test_conflict(self):
        (self.root/'include/new.hpp').write_text('USER EDIT');self.assertNotEqual(self.run_apply('--apply').returncode,0)
        self.assertNotIn('level4.cmake',(self.root/'CMakeLists.txt').read_text())
    def test_bad_hash(self):
        (self.payload/'include/new.hpp').write_text('changed');self.assertNotEqual(self.run_apply('--apply').returncode,0)
        self.assertFalse((self.root/'include/new.hpp').exists())
    def test_missing_interface(self):
        (self.root/'include/maple_moe.hpp').write_text('v0.3');self.assertNotEqual(self.run_apply('--apply').returncode,0)
    def test_cmake_refusal(self):
        (self.root/'CMakeLists.txt').write_text('project(llama)');self.assertNotEqual(self.run_apply('--apply').returncode,0)
        self.assertEqual(self.run_apply('--apply','--no-cmake').returncode,0)
if __name__=='__main__':unittest.main()
