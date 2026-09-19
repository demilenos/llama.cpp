"""Archive Bonsai2 measurements and verify hashes. This script never deletes originals."""
import hashlib, json, pathlib, subprocess, zipfile, datetime, os
ROOT = pathlib.Path(__file__).resolve().parents[3]
REPO = ROOT / 'prismml-llama.cpp'
OUT = ROOT / 'archives'
OUT.mkdir(exist_ok=True)
stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
archive = OUT / ('alchemist-bonsai2-measurements-' + stamp + '.zip')
meta = json.loads((ROOT/'pp-fix/server-current-launch.json').read_text(encoding='utf-8-sig'))
protected = {pathlib.Path(meta[k]).resolve() for k in ('StdoutLog','StderrLog')}
protected.add((ROOT/'pp-fix/server-current-launch.json').resolve())
items = {}
def add(path, delete=False):
    if not path.is_file(): return
    resolved = path.resolve()
    if not resolved.is_relative_to(ROOT) or path.is_symlink():
        raise RuntimeError('Unsafe path: '+str(path))
    name = resolved.relative_to(ROOT).as_posix()
    items[name] = {'path':resolved, 'delete':delete and resolved not in protected}
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()
for p in (ROOT/'pp-fix').iterdir():
    if p.suffix.lower() in ('.log','.json','.kld'): add(p, True)
    elif p.suffix.lower() in ('.md','.ps1','.cmd','.patch','.txt'): add(p)
for p in (ROOT/'logs').rglob('*.log'): add(p,True)
for p in ROOT.glob('llama-server-9931*.log'): add(p,True)
for p in (ROOT/'probe-four').rglob('*'):
    if p.suffix.lower()=='.zip': continue
    measurement=(p.suffix.lower() in ('.log','.bin','.spv') or 'intel-shader-cache-after-isa' in p.parts or p.name=='done.txt')
    add(p,measurement)
for p in (REPO/'docs/a750-bonsai2').rglob('*'): add(p,'evidence' in p.parts)
for folder in (ROOT/'pp-fix/pre-signs',ROOT/'pp-fix/baseline-bin',REPO/'scripts/a750-bonsai2'):
    for p in folder.rglob('*'): add(p)
for p in (REPO/'build-a750-vulkan/bin').iterdir():
    if p.suffix.lower() in ('.exe','.dll'): add(p)
add(REPO/'build-a750-vulkan/CMakeCache.txt')
def git(*args): return subprocess.check_output(['git','-C',str(REPO),*args])
model=ROOT/'Ternary-Bonsai-2-27B-PTQ1_0.gguf'
manifest={'created':stamp,'workspace':str(ROOT),'source_head':git('rev-parse','HEAD').decode().strip(),
 'external_model':{'path':model.name,'bytes':model.stat().st_size,'sha256':digest(model)},
 'notes':['ISA extraction logs include failures; no recovered Alchemist ISA is claimed.',
 'Current server logs are snapshots and retained live. Source, binaries, fixtures and current launch metadata are retained.',
 'PP XMX and final TG configurations differ; consult decision and memory-budget reports.'], 'files':[]}
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
    for name,info in sorted(items.items()):
        h=hashlib.sha256(); size=0
        with info['path'].open('rb') as source, z.open(name,'w',force_zip64=True) as dest:
            for chunk in iter(lambda:source.read(1024*1024),b''):
                h.update(chunk); size+=len(chunk); dest.write(chunk)
        manifest['files'].append({'path':name,'bytes':size,'sha256':h.hexdigest(),'delete_original':info['delete']})
    z.writestr('provenance/source-changes.patch',git('diff','07c232edf','HEAD','--','ggml','src','tests','scripts'))
    z.writestr('provenance/commits.txt',git('log','--format=fuller','07c232edf..HEAD'))
    z.writestr('provenance/status.txt',git('status','--short'))
    z.writestr('MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    z.writestr('RESTORE.txt','Extract the ZIP into C:/AI/bonsai2_27b to restore original relative paths. Avoid overwriting live files. To rerun quality validation, restore pp-fix/baseline.kld. The GGUF is external and identified by SHA256 in MANIFEST.json. Older source versions remain in Git; this ZIP also contains source changes and baseline/current binaries.\n')
with zipfile.ZipFile(archive) as z:
    bad=z.testzip()
    if bad: raise RuntimeError('CRC failed: '+bad)
    for item in manifest['files']:
        h=hashlib.sha256(); size=0
        with z.open(item['path']) as f:
            for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk); size+=len(chunk)
        if h.hexdigest()!=item['sha256'] or size!=item['bytes']: raise RuntimeError('Archive hash mismatch: '+item['path'])
# Originals remain untouched. A separate explicit scope approval is required for deletion.
receipt={'archive':str(archive),'archive_sha256':digest(archive),'archive_bytes':archive.stat().st_size,
 'verified_files':len(manifest['files']),'deleted':[],
 'proposed_deletions':[i for i in manifest['files'] if i['delete_original']],
 'retained':[i['path'] for i in manifest['files'] if not i['delete_original']]}
receipt_path=archive.with_suffix('.receipt.json')
receipt_path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'archive':str(archive),'sha256':receipt['archive_sha256'],'MiB':round(receipt['archive_bytes']/1048576,2),'verified':receipt['verified_files'],'deleted':0,'proposed_deletions':len(receipt['proposed_deletions'])}))
