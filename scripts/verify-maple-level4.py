import json, os, subprocess, time, urllib.request
from pathlib import Path
import argparse
parser=argparse.ArgumentParser(description='Run actual Maple server baseline/RC1/RC4/RC8 smoke comparisons')
parser.add_argument('--server',required=True)
parser.add_argument('--model',required=True)
parser.add_argument('--out',required=True,type=Path)
parser.add_argument('--port',type=int,default=9954)
args=parser.parse_args()
root=args.out.resolve();root.mkdir(parents=True,exist_ok=True)
exe=str(Path(args.server).resolve());model=str(Path(args.model).resolve())
base_url=f'http://127.0.0.1:{args.port}'
summary=[]
for tile in ('baseline','1','4','8'):
 env=os.environ.copy()
 if tile=='baseline': env.pop('LLAMA_MAPLE_LEVEL4',None)
 else: env.update(LLAMA_MAPLE_LEVEL4='1', LLAMA_MAPLE_TOKEN_TILE=tile)
 log=root/f'server-{tile}.log'
 with log.open('w',encoding='utf-8') as f:
  proc=subprocess.Popen([exe,'-m',model,'-ngl','99','-c','512','-b','16','-ub','16','--no-warmup','--host','127.0.0.1','--port',str(args.port),'-np','1','-t','4'],env=env,stdout=f,stderr=subprocess.STDOUT)
  try:
   deadline=time.time()+180
   while True:
    if proc.poll() is not None: raise RuntimeError(f'{tile} server exited {proc.returncode}: {log}')
    try:
     with urllib.request.urlopen(base_url+'/health',timeout=2) as r:
      if r.status==200: break
    except Exception: pass
    if time.time()>deadline: raise TimeoutError(f'{tile} server startup')
    time.sleep(1)
   payload={'prompt':'The capital of France is','n_predict':2,'temperature':0,'seed':1,'n_probs':20,'cache_prompt':False}
   request=urllib.request.Request(base_url+'/completion',json.dumps(payload).encode(),{'Content-Type':'application/json'})
   with urllib.request.urlopen(request,timeout=480) as response: data=json.load(response)
   (root/f'response-{tile}.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
   summary.append({'tile':tile,'content':data.get('content'),'tokens_predicted':data.get('tokens_predicted'),'timings':data.get('timings'),'status':'PASS'})
   print(json.dumps(summary[-1]),flush=True)
  finally:
   proc.terminate()
   try: proc.wait(timeout=20)
   except subprocess.TimeoutExpired: proc.kill();proc.wait()
(root/'server-summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')

reference=json.loads((root/'response-1.json').read_text(encoding='utf-8'))
for tile in ('4','8'):
 candidate=json.loads((root/f'response-{tile}.json').read_text(encoding='utf-8'))
 if candidate['content']!=reference['content']: raise AssertionError(f'RC{tile} token text differs')
 deltas=[]
 for a,b in zip(reference['completion_probabilities'],candidate['completion_probabilities']):
  aa={x['id']:x['logprob'] for x in a['top_logprobs']}
  bb={x['id']:x['logprob'] for x in b['top_logprobs']}
  if aa.keys()!=bb.keys(): raise AssertionError(f'RC{tile} top-20 IDs differ')
  deltas.extend(abs(aa[k]-bb[k]) for k in aa)
 if not deltas or max(deltas)>1e-5: raise AssertionError(f'RC{tile} top-20 logprobs differ: {max(deltas,default=float("inf"))}')
 print(f'RC{tile} vs RC1 top-20 logprob max delta: {max(deltas)}',flush=True)
