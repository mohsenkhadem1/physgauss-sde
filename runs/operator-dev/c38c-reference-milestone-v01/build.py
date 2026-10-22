"""Completed small-reference and forward-method milestone; raw data stay local."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sha=lambda b:hashlib.sha256(b).hexdigest();paths=set();inventories=[]
for name in ['c38-bar-active-reference-v01','p2-transverse-spatial-v01','p2-two-level-static-v01','p2-contact-dop853-v01']:
 p=ROOT/'runs/operator-dev'/name/'publication-inventory.json';inventories.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p.read_bytes())))
 for r in json.loads(p.read_text())['files']:
  if sha((ROOT/r['path']).read_bytes())!=r['sha256']:raise ValueError('Changed frozen component '+r['path'])
  paths.add(r['path'])
 paths.add(str(p.relative_to(ROOT)))
def compact(x):
 if isinstance(x,dict):
  r={k:compact(v) for k,v in x.items() if k not in ['history','linear_records','traceback']}
  if x.get('linear_records'):r['maximum_linear_iterations']=max(v['iterations'] for v in x['linear_records'])
  return r
 if isinstance(x,list):return [compact(v) for v in x]
 return x
name='p2-graded-static-v01';op=ROOT/'runs/operator-dev'/name;source=ROOT/'runs/gaussian-campaign-v01'/name/'summary.json';execution=json.loads((op/'execution.json').read_text())
if execution.get('terminal') is not True:raise ValueError('Terminal execution required')
result=compact(json.loads(source.read_text()));result['full_local_result']=dict(path=str(source.relative_to(ROOT)),sha256=sha(source.read_bytes()),omitted='Iteration histories only; metrics and capture hashes retained');result['execution']=execution
target=source.with_name('publication-summary.json');target.write_text(json.dumps(result,indent=2)+'\n');paths.add(str(target.relative_to(ROOT)))
for n in ['protocol.json','execution.json']:paths.add(str((op/n).relative_to(ROOT)))
paths.update(['README.md','docs/GAUSSIAN_C38_FORWARD_PROGRESS.md','experiments/campaign_p2_geometry.py','experiments/campaign_p2_exact_quadrature.py','experiments/run_p2_graded_static.py','tests/test_campaign_p2_geometry.py','tests/test_campaign_p2_exact_quadrature.py','runs/operator-dev/c38c-reference-milestone-v01/build.py','runs/operator-dev/c38c-reference-milestone-v01/validation.json'])
files=[]
for name in sorted(paths):
 p=ROOT/name;b=p.read_bytes();b.decode('utf-8')
 if p.suffix not in ['.py','.md','.json','.txt','.log'] or p.is_symlink():raise ValueError('Source/text only '+name)
 files.append(dict(path=name,bytes=len(b),sha256=sha(b),git_blob_sha1=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()))
d=dict(schema='c38c-reference-milestone-v01',base_commit='b114b54e20410e1465bf97f9d2e5008822470f4a',base_tree='60feb0b1ceae5e9fb3e621dec82ba60b39b45f26',component_inventories=inventories,files=files,file_count=len(files),total_bytes=sum(r['bytes'] for r in files),raw_arrays_or_observations_or_media=False,campaign_complete=False)
p=ROOT/'runs/campaign-c38c-paths.json';p.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(dict(files=len(files),bytes=d['total_bytes'],manifest_sha256=sha(p.read_bytes()))))
