import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import run_gemm_suite as g

class GemmSuiteTests(unittest.TestCase):
    def test_full_count(self):
        self.assertEqual(len(g.cases_for('full')),14)
        self.assertEqual([c['tokens'] for c in g.cases_for('full') if c.get('model_shape')],list(g.QS))
    def test_quick_count(self):self.assertEqual(len(g.cases_for('quick')),7)
    def test_ooo_and_tail_smoke(self):
        c=g.cases_for('smoke')[1];self.assertEqual(c['tokens'],13);self.assertEqual(c['order'],0)
    def test_fixed_contract(self):
        for r in (4,8):
            for c in g.cases_for('full'):
                a=g.case_args(c,{},28,0,r);d=dict(zip(a[::2],a[1::2]))
                self.assertEqual(d['--token-tile'],str(r))
                self.assertEqual(d['--gate-split'],'1');self.assertEqual(d['--down-split'],'1')
                self.assertEqual(d['--local'],'4');self.assertEqual(d['--id-mode'],'rotate')
                self.assertNotIn('--groups',d)
    def test_capsule_selection(self):
        caps={k:Path(k+'.mw2') for k in ('gate','up','down')}
        self.assertNotIn('--gate',g.case_args(g.cases_for('full')[0],caps,28,0))
        self.assertIn('--gate',g.case_args(g.cases_for('full')[-1],caps,28,0))
    def stats(self,text):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'s.csv';p.write_text('round,phase,variant,wall_us\n'+text)
            return g.paired_stats(p)
    def test_paired_delta(self):
        r=self.stats('1,repeat,x-grouped-rc1,100\n1,repeat,x-gemm-rc4,80\n2,repeat,x-grouped-rc1,90\n2,repeat,x-gemm-rc4,110\n')
        self.assertEqual(r['gemm_wins'],1);self.assertEqual(r['paired_rounds'],2);self.assertEqual(r['paired_saved_median_us'],0)
    def test_missing_arm(self):
        with self.assertRaises(ValueError):self.stats('1,repeat,x-grouped-rc1,100\n')
    def test_unknown_arm(self):
        with self.assertRaises(ValueError):self.stats('1,repeat,x,100\n')
    def test_nan_time(self):
        with self.assertRaises(ValueError):self.stats('1,repeat,x-gemm-rc4,nan\n')
    def test_duplicate_arm(self):
        with self.assertRaises(ValueError):self.stats('1,repeat,x-grouped-rc1,100\n1,repeat,x-grouped-rc1,90\n')
    def test_sources(self):
        s=g.sources()
        for name in ('src/maple_w2a8_gemm.cpp','src/expert_tiles.cpp','tools/gemm_compare.cpp','include/expert_tiles_reference.hpp'):
            self.assertIn(name,s)
    def test_no_implicit_stale_build(self):
        with tempfile.TemporaryDirectory() as td:
            e=Path(td)/'fake.exe';e.write_bytes(b'x')
            with self.assertRaises(ValueError):g.verify_build(e)
            p=e.parent/'gemm-build.json'
            for v,sh,eh in [('0.4',g.sources(),g.sha256(e)),('0.5',{},g.sha256(e)),('0.5',g.sources(),'bad')]:
                p.write_text(json.dumps({'version':v,'source_sha256':sh,'executable_sha256':eh}))
                with self.assertRaises(ValueError):g.verify_build(e)
    def test_accept_matching_build(self):
        with tempfile.TemporaryDirectory() as td:
            e=Path(td)/'fake.exe';e.write_bytes(b'x')
            v={'version':'0.5','source_sha256':g.sources(),'executable_sha256':g.sha256(e)}
            (e.parent/'gemm-build.json').write_text(json.dumps(v));self.assertEqual(g.verify_build(e),v)
    def test_planner_has_no_host_wait(self):
        s=(ROOT/'src/expert_tiles.cpp').read_text()
        for name in ('.wait(','.wait_and_throw(','malloc_','memcpy('):self.assertNotIn(name,s)
    def test_actual_multitoken_dpas(self):
        s=(ROOT/'src/maple_w2a8_gemm.cpp').read_text()
        self.assertIn('xmx::dpas<8,R,int32_t,int32_t,uint32_t,int8_t',s)
        self.assertIn('es::simd<int8_t,R*32>',s);self.assertIn('es::simd<int32_t,R*8>',s)
        self.assertIn('da[r]=a.scales[ax/32]',s)
        self.assertIn('if(active[r])',s);self.assertIn('if(original[r]>=0)',s)
    def test_baseline_is_grouped_not_token_major(self):
        s=(ROOT/'tools/gemm_compare.cpp').read_text()
        self.assertIn('for(uint32_t repeat:{1u,token_tile})',s)
        self.assertIn('o.expert_grouping=true;o.token_tile=repeat;',s)
        self.assertIn('vars[(r+j)%2]',s)
    def test_raw_inputs_are_saved(self):
        s=(ROOT/'tools/gemm_compare.cpp').read_text()
        for name in ('x.f32','routes.f32','ids_frames.i32','--ids-frames-file'):self.assertIn(name,s)
    def test_quant_planes_exact(self):
        s=(ROOT/'tools/gemm_compare.cpp').read_text()
        self.assertIn('multi-token path altered quantized q/scale planes',s)
    def test_no_memory_expansion_hidden_in_kernel(self):
        s=(ROOT/'src/maple_w2a8_gemm.cpp').read_text().split('template<int R>static void probe_repeat')[0]
        self.assertNotIn('malloc_',s);self.assertNotIn('.wait(',s);self.assertNotIn('memcpy(',s)
        self.assertIn('p.y0+oi',s)
if __name__=='__main__':unittest.main()
