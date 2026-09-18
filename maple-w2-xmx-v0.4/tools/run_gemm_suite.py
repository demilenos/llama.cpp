#!/usr/bin/env python3
"""v0.5: paired grouped RC1 / multi-token RC4(8) Q sweep. G32/H32 unchanged."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import shutil
from pathlib import Path
import statistics
import sys
from typing import Any
from run_suite import ROOT, run_logged

QS = (1,2,4,8,16,32,64,128,256,512,1024,2048)

def source_files(root: Path=ROOT) -> list[Path]:
    return sorted([*root.glob('include/*.hpp'), *root.glob('src/*.cpp'),root/'tools/gemm_compare.cpp'])

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        while chunk:=f.read(4*1024*1024): h.update(chunk)
    return h.hexdigest()

def sources(root: Path=ROOT) -> dict[str,str]:
    return {p.relative_to(root).as_posix():sha256(p) for p in source_files(root)}

def cases_for(suite: str) -> list[dict[str,Any]]:
    cases=[dict(name='gemm-smoke-q1',tokens=1,k=256,hidden=256,experts=8,order=1),
           dict(name='gemm-smoke-q13-ooo',tokens=13,k=256,hidden=256,experts=8,order=0)]
    qs=() if suite=='smoke' else (1,4,64,256,2048) if suite=='quick' else QS
    return cases+[dict(name=f'q{q:04d}',tokens=q,k=2048,hidden=512,experts=256,order=1,model_shape=True) for q in qs]

def case_args(c:dict[str,Any], caps:dict[str,Path], repeats:int, scrub:int, token_tile:int=4) -> list[str]:
    args=['--tokens',str(c['tokens']),'--k',str(c['k']),'--hidden',str(c['hidden']),'--experts',str(c['experts']),
          '--topk','8','--token-tile',str(token_tile),'--gate-split','1','--down-split','1','--local','4',
          '--in-order',str(c['order']),'--repeats',str(min(4,repeats) if not c.get('model_shape') else repeats),
          '--warmup','3','--pattern','normal','--id-mode','rotate','--scrub-mib',str(scrub),'--skip-probe','1']
    if c.get('model_shape') and caps:
        for key,path in caps.items(): args.extend([f'--{key}',str(path)])
    return args

def verify_build(exe: Path) -> dict[str,Any]:
    path=exe.parent/'gemm-build.json'
    if not path.is_file(): raise ValueError(f'missing build manifest {path}; rebuild v0.5 first')
    built=json.loads(path.read_text(encoding='utf-8-sig'))
    if built.get('version')!='0.5': raise ValueError('not a v0.5 build')
    if built.get('source_sha256')!=sources(): raise ValueError('source changed after build; do not use -NoBuild')
    if built.get('executable_sha256')!=sha256(exe): raise ValueError('executable changed after build')
    return built

def paired_stats(path:Path) -> dict[str,Any]:
    with path.open(newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
    rounds:dict[str,dict[str,float]]={}
    for row in rows:
        if row['phase']!='repeat': continue
        arm='gemm' if '-gemm-rc' in row['variant'] else 'baseline' if row['variant'].endswith('grouped-rc1') else None
        if arm is None: raise ValueError('unknown benchmark arm')
        if not 0 < float(row['wall_us']) < float('inf'): raise ValueError('non-positive/nonfinite wall sample')
        if arm in rounds.setdefault(row['round'],{}): raise ValueError('duplicate arm in round')
        rounds[row['round']][arm]=float(row['wall_us'])
    if not rounds or any(set(r)!= {'baseline','gemm'} for r in rounds.values()): raise ValueError('unpaired samples')
    delta=[r['baseline']-r['gemm'] for r in rounds.values()]
    return dict(paired_rounds=len(delta),gemm_wins=sum(x>0 for x in delta),paired_saved_median_us=statistics.median(delta),
        paired_speedup_median=statistics.median(r['baseline']/r['gemm'] for r in rounds.values()))

def write_csv(path:Path,rows:list[dict[str,Any]]) -> None:
    if not rows: return
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def capture_provenance(out:Path,exe:Path) -> dict[str,Any]:
    result={'platform':platform.platform(),'python':sys.version,'argv':sys.argv,
            'executable':str(exe.resolve()),'executable_sha256':sha256(exe),
            'harness_sha256':sha256(Path(__file__))}
    # No full environment dump: capture only relevant, non-secret runtime switches.
    keys=('ONEAPI_DEVICE_SELECTOR','SYCL_DEVICE_FILTER','SYCL_CACHE_PERSISTENT','SYCL_CACHE_DIR',
          'SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS','SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE',
          'UR_L0_USE_IMMEDIATE_COMMANDLISTS','UR_L0_USE_COPY_ENGINE','MAPLE_W2_DEVICE_CHECK_EVERY_CALL')
    result['runtime_environment']={k:os.environ[k] for k in keys if k in os.environ}
    pd=out/'provenance';pd.mkdir(exist_ok=True)
    for name in ('compiler-version.txt','gemm-build.json','maple_w2a8_gemm-build.log',
                 'expert_tiles-build.log','maple_moe-build.log','maple-gemm-compare-link.log'):
        f=exe.parent/name
        if f.is_file():shutil.copy2(f,pd/name)
    git={}
    for label,args in (('head',['rev-parse','HEAD']),('status',['status','--porcelain']),
                       ('diff',['diff','--binary']),('cached_diff',['diff','--cached','--binary'])):
        try:
            r=subprocess.run(['git','-C',str(ROOT),*args],capture_output=True,timeout=20,check=False)
            git[label]={'exit':r.returncode}
            if r.returncode==0:
                filename=f'git-{label}.txt';(pd/filename).write_bytes(r.stdout)
                git[label].update(file=filename,sha256=sha256(pd/filename))
                if label=='head':git[label]['value']=r.stdout.decode(errors='replace').strip()
            else:git[label]['unavailable']=r.stderr.decode(errors='replace').strip()
        except (OSError,subprocess.TimeoutExpired) as e:git[label]={'unavailable':str(e)}
    result['git']=git
    result['git_note']='A parent repository HEAD alone does not describe untracked lab files; use source/build hashes too.'
    return result

def main(argv:list[str]|None=None)->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--exe',type=Path,default=ROOT/'build'/('maple-gemm-compare.exe' if os.name=='nt' else 'maple-gemm-compare'))
    ap.add_argument('--suite',choices=('smoke','quick','full'),default='full');ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--token-tile',type=int,choices=(4,8),default=4);ap.add_argument('--device',default='A750');ap.add_argument('--repeats',type=int,default=28);ap.add_argument('--scrub-mib',type=int,default=0)
    ap.add_argument('--layer',type=int,default=0);ap.add_argument('--dry-run',action='store_true')
    group=ap.add_mutually_exclusive_group();group.add_argument('--model',type=Path);group.add_argument('--capsule-dir',type=Path);group.add_argument('--synthetic',action='store_true')
    ns=ap.parse_args(argv)
    if ns.layer<0 or not 2<=ns.repeats<=10000 or not 0<=ns.scrub_mib<=128: ap.error('invalid layer/repeats/scrub')
    plan=cases_for(ns.suite)
    if ns.dry_run:
        print(json.dumps({'version':'0.5','only_change':'multi-token DPAS + per-call tile descriptors','token_tile':ns.token_tile,'arms':2,'fixed':'s2tile8 G32/H32 gluquant async split1/1 local4','cases':plan},indent=2));return 0
    if ns.suite!='smoke' and not(ns.model or ns.capsule_dir or ns.synthetic): ap.error('provide --model/--capsule-dir, or explicitly --synthetic')
    if not ns.exe.is_file(): ap.error(f'missing executable: {ns.exe}')
    if (ns.out/'suite_status.json').exists(): ap.error('choose fresh output directory')
    ns.out.mkdir(parents=True,exist_ok=True)
    rows=[];speeds=[]
    status:dict[str,Any]={'version':'0.5','complete':False,'passed':False,'cases':[],
        'scope':'one layer; grouping EVERY call in both arms, tile plan EVERY GEMM call; no Vulkan or server TG',
        'source_note':'based on attached v0.4 source zip SHA256 606d2937ff6cf267050112650913561a74532a7e990f55736eea06a2341eb008; not an assertion of the current local merged tree'}
    try:
        built=verify_build(ns.exe);status['build']=built;status['provenance']=capture_provenance(ns.out,ns.exe)
        run_logged([str(ns.exe.resolve()),'--device',ns.device,'--probe-only','1'],ns.out/'probe.log')
        caps={}
        if ns.capsule_dir:
            for key in ('gate','up','down'):
                path=(ns.capsule_dir/f'{key}.mw2').resolve()
                if not path.is_file(): raise ValueError(f'missing capsule {path}')
                caps[key]=path
        elif ns.model and ns.suite!='smoke':
            if not ns.model.is_file(): raise ValueError(f'missing model {ns.model}')
            for key in ('gate','up','down'):
                path=(ns.out/f'weights/{key}.mw2').resolve();path.parent.mkdir(parents=True,exist_ok=True)
                run_logged([sys.executable,str(ROOT/'tools/extract_tq2.py'),'--model',str(ns.model.resolve()),'--tensor',f'blk.{ns.layer}.ffn_{key}_exps.weight','--output',str(path)],ns.out/f'extract-{key}.log')
                caps[key]=path
        status['capsules']={k:dict(path=str(p),sha256=sha256(p)) for k,p in caps.items()}
        (ns.out/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
        for c in plan:
            directory=(ns.out/c['name']).resolve()
            cmd=[str(ns.exe.resolve()),'--device',ns.device,'--out',str(directory)]+case_args(c,caps,ns.repeats,ns.scrub_mib,ns.token_tile)
            run_logged(cmd,ns.out/f'{c["name"]}.log')
            with (directory/'summary.csv').open(newline='',encoding='utf-8') as f: new=list(csv.DictReader(f))
            if len(new)!=2 or {r['token_tile'] for r in new}!={'1',str(ns.token_tile)} or any(r['expert_grouping']!='1' for r in new) or any(r['kernel_pass']!='1' or r['pair_pass']!='1' for r in new):
                raise ValueError(f'invalid/failed two-arm summary: {c["name"]}')
            for r in new:r['case']=c['name']
            rows.extend(new);write_csv(ns.out/'all_gemm_summary.csv',rows)
            b=next(r for r in new if r['token_tile']=='1');g=next(r for r in new if r['token_tile']==str(ns.token_tile))
            speeds.append(dict(case=c['name'],Q=c['tokens'],source=g['source'],token_tile=ns.token_tile,grouped_rc1_us=b['wall_median_us'],gemm_us=g['wall_median_us'],
                speedup=g['speedup_vs_grouped_rc1_same_build'],group_kernel_us=g['group_kernel_median_us'],tile_plan_us=g['tile_plan_median_us'],
                gate_speedup=float(b['gate_up_median_us'])/float(g['gate_up_median_us']),down_speedup=float(b['down_median_us'])/float(g['down_median_us']),
                grouped_rc1_p95_us=b['wall_p95_us'],gemm_p95_us=g['wall_p95_us'],**paired_stats(directory/'samples.csv')))
            write_csv(ns.out/'q_speedup.csv',speeds)
            status['cases'].append({'name':c['name'],'passed':True,'argv':cmd,'file_sha256':{f.name:sha256(f) for f in sorted(directory.iterdir()) if f.is_file()}})
        status['passed']=True;status['complete']=True
        print(f'PASS {len(plan)} cases/{len(rows)} variants. Read {ns.out / "q_speedup.csv"}.')
        print('No performance winner or automatic cutover is preselected. Group construction is in BOTH arms; tile planning is included in the GEMM arm. Not server TG.')
        return 0
    except (OSError,ValueError,RuntimeError) as exc:
        status['error']=str(exc);print(f'FAIL {exc}',file=sys.stderr);return 1
    finally:
        (ns.out/'suite_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
if __name__=='__main__':raise SystemExit(main())
