"""Verified completed-control inventories and source-only preparations."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sha=lambda b:hashlib.sha256(b).hexdigest()
paths=set();inventories=[]
for name,filename in [('p2-streaming-graded64-v01','publication-inventory.json'),('p2-graded-coarse-v01','publication-inventory.json'),('p2-graded-coarse-benchmark-v01','publication-inventory.json'),('p2-warp-trajectory-v01','publication-inventory.json'),('p2-warp-trajectory-v02','publication-inventory.json'),('p2-warp-trajectory-v03','publication-inventory.json'),('c38-static-prequalification-v01','preparation-inventory.json'),('c38-static-data-interface-v01','publication-inventory.json')]:
 p=ROOT/'runs/operator-dev'/name/filename
 inventories.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p.read_bytes())))
 for row in json.loads(p.read_text())['files']:
  if sha((ROOT/row['path']).read_bytes())!=row['sha256']:raise ValueError('Changed component '+row['path'])
  paths.add(row['path'])
 paths.add(str(p.relative_to(ROOT)))
paths.update(['docs/GAUSSIAN_C38_MECHANICS_CONTROLS.md','docs/GAUSSIAN_C38_FORWARD_PROGRESS.md','runs/operator-dev/c38f-mechanics-controls-v01/build.py'])
files=[]
for name in sorted(paths):
 p=ROOT/name;b=p.read_bytes();b.decode('utf-8')
 if p.is_symlink() or p.suffix not in ['.py','.md','.json','.txt','.log']:raise ValueError('UTF8 source/text only')
 files.append(dict(path=name,bytes=len(b),sha256=sha(b),git_blob_sha1=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()))
m=dict(schema='C38F-mechanics-controls-v01',base_commit='f5594b5f7cb1233edb1fc30ce0e7c0981398f304',base_tree='bf29708087f26be985aa41e307dbbc830c4f0cc1',component_inventories=inventories,files=files,file_count=len(files),total_bytes=sum(r['bytes'] for r in files),raw_datasets_or_media_or_model_checkpoints=False,campaign_complete=False)
p=ROOT/'runs/campaign-c38f-paths.json';p.write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(dict(files=len(files),bytes=m['total_bytes'],manifest_sha256=sha(p.read_bytes()))))
