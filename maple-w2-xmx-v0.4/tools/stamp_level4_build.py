#!/usr/bin/env python3
import argparse,hashlib,json,platform,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--exe',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
files={str(p.relative_to(root)).replace('\\','/'):sha(p) for sub in ('src','include','integration') for p in (root/sub).rglob('*') if p.is_file() and p.suffix in ('.hpp','.cpp')}
def git(*args):
 try:
  r=subprocess.run(['git','-C',str(root),*args],text=True,capture_output=True);return r.stdout.strip() if r.returncode==0 else None
 except OSError:return None
out={'scope':'Level4 compiled-source identity; not GPU/model verification','platform':platform.platform(),'executable':str(a.exe),'exe_sha256':sha(a.exe),'sources':files,'git_head':git('rev-parse','HEAD'),'git_status':git('status','--porcelain')}
dest=a.exe.resolve().parent/'level4-build.json';dest.write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf-8');print(dest)
