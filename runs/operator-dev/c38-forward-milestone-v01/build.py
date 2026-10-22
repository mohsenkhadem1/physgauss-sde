"""Build text-only milestone manifest; frozen component bytes are verified."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sha=lambda b:hashlib.sha256(b).hexdigest()
invs=['newton-stable-p2-spatial-v01','newton-gravity-reference-v01','newton-gravity-static-v01','newton-contact-validation-v01','block-heldout-suite-v01','p2-consistent-transient-v01','p2-plane-contact-v01']
paths=set();inventories=[]
for name in invs:
 p=ROOT/'runs/operator-dev'/name/'publication-inventory.json';d=json.loads(p.read_text());inventories.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p.read_bytes())))
 for r in d['files']:
  if sha((ROOT/r['path']).read_bytes())!=r['sha256']:raise ValueError('Changed frozen component '+r['path'])
  paths.add(r['path'])
 paths.add(str(p.relative_to(ROOT)))
def compact(x):
 if isinstance(x,dict):return {k:compact(v) for k,v in x.items() if k not in ['ledger','history','traceback']}
 if isinstance(x,list):return [compact(v) for v in x]
 return x
for name in ['p2-torch-benchmark-v01','p2-torch-spatial-v01','p2-torch-time-v01','p2-gpu-refinement-v01']:
 op=ROOT/'runs/operator-dev'/name;source=ROOT/'runs/gaussian-campaign-v01'/name/'summary.json';d=json.loads(source.read_text())
 if d.get('completed') is not True:raise ValueError('Terminal complete case required '+name)
 result=compact(d);result['full_local_result']=dict(path=str(source.relative_to(ROOT)),sha256=sha(source.read_bytes()),omitted='Only iteration histories and per-step ledgers; all aggregate metrics/capture hashes retained')
 target=source.with_name('publication-summary.json');target.write_text(json.dumps(result,indent=2)+'\n');paths.add(str(target.relative_to(ROOT)))
 for n in ['protocol.json','execution.json']:paths.add(str((op/n).relative_to(ROOT)))
paths.update(['README.md','docs/GAUSSIAN_C38_FORWARD_PROGRESS.md','experiments/campaign_p2_torch.py','experiments/campaign_p2_transient.py','experiments/run_p2_torch_benchmark.py','experiments/run_p2_torch_spatial.py','experiments/run_p2_torch_time.py','experiments/run_p2_gpu_refinement.py','tests/test_campaign_p2_torch.py','tests/test_campaign_p2_transient.py','runs/operator-dev/p2-torch-independent-review-v01/review.json','runs/operator-dev/c38-full-campaign-v01/intent.json','runs/operator-dev/c38-forward-milestone-v01/build.py','runs/operator-dev/c38-forward-milestone-v01/validation.json'])
files=[]
for name in sorted(paths):
 p=ROOT/name;b=p.read_bytes();b.decode('utf-8')
 if p.suffix not in ['.py','.md','.json','.txt','.log'] or p.is_symlink():raise ValueError('Source/text only '+name)
 files.append(dict(path=name,bytes=len(b),sha256=sha(b),git_blob_sha1=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()))
manifest=dict(schema='c38-forward-milestone-v01',base_commit='7a5ab7a74a77e1006eb8ec32fdefa78dde5e8ef1',base_tree='1f2c483f495b182cb1568bf46f38f3d04b7f4da5',component_inventories=inventories,files=files,file_count=len(files),total_bytes=sum(r['bytes'] for r in files),raw_arrays_or_media=False,campaign_complete=False)
p=ROOT/'runs/campaign-c38a-paths.json';p.write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(dict(files=len(files),bytes=manifest['total_bytes'],manifest_sha256=sha(p.read_bytes()))))
