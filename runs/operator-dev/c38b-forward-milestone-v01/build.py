"""Verified compact milestone; no raw arrays, observations or media."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sha=lambda b:hashlib.sha256(b).hexdigest()
invs=['p2-contact-refinement-v02','p2-contact-validation-v01','p2-consistent-temporal-refinement-v02','p2-cpu-gpu-reference-comparison-v01','p2-block-preconditioner-v01','p2-contact-cpu-oracle-v01','p2-coarse-preconditioner-v01','c38-prediction-controls-v01','c38-data-controls-v01'];paths=set();inventories=[]
for name in invs:
 p=ROOT/'runs/operator-dev'/name/'publication-inventory.json';inventories.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p.read_bytes())))
 for r in json.loads(p.read_text())['files']:
  if sha((ROOT/r['path']).read_bytes())!=r['sha256']:raise ValueError('Changed frozen component '+r['path'])
  paths.add(r['path'])
 paths.add(str(p.relative_to(ROOT)))
def compact(x):
 if isinstance(x,dict):
  result={k:compact(v) for k,v in x.items() if k not in ['ledger','history','Newton_history','mixed_records','traceback']}
  if 'mixed_records' in x:result['mixed_linear_summary']=dict(solves=len(x['mixed_records']),FP64_fallbacks=sum(r['fallback'] for r in x['mixed_records']),maximum_original_FP64_linear_residual_N=max(r['original_FP64_residual_N'] for r in x['mixed_records']),maximum_CG_iteration_upper_bound=max(r['total_CG_iterations_upper_bound'] for r in x['mixed_records']))
  return result
 if isinstance(x,list):return [compact(v) for v in x]
 return x
for name in ['p2-gpu-refinement-v02','p2-gpu-refinement-v03','p2-varied-static-v01','p2-mixed-benchmark-v01']:
 op=ROOT/'runs/operator-dev'/name;source=ROOT/'runs/gaussian-campaign-v01'/name/'summary.json';d=json.loads(source.read_text());execution=json.loads((op/'execution.json').read_text())
 if execution.get('terminal') is not True:raise ValueError('Terminal execution required '+name)
 result=compact(d);result['full_local_result']=dict(path=str(source.relative_to(ROOT)),sha256=sha(source.read_bytes()),omitted='Iteration/step ledgers only; aggregate metrics and capture hashes retained')
 result['execution']=execution
 target=source.with_name('publication-summary.json');target.write_text(json.dumps(result,indent=2)+'\n');paths.add(str(target.relative_to(ROOT)))
 for n in ['protocol.json','execution.json']:paths.add(str((op/n).relative_to(ROOT)))
paths.update(['README.md','docs/GAUSSIAN_C38_FORWARD_PROGRESS.md','experiments/run_p2_gpu_refinement_v02.py','experiments/run_p2_gpu_refinement_v03.py','experiments/run_p2_varied_static.py','experiments/campaign_p2_mixed_cg.py','experiments/run_p2_mixed_benchmark.py','experiments/diagnose_p2_trajectory_geometry.py','tests/test_campaign_p2_mixed_cg.py','tests/test_p2_trajectory_geometry.py','runs/operator-dev/p2-gpu-refinement-v02/operator-resource-preemption.json','runs/operator-dev/p2-gpu-refinement-v02/captured-geometry.json','runs/operator-dev/p2-gpu-refinement-v03/captured-geometry.json','runs/operator-dev/c38b-forward-milestone-v01/build.py','runs/operator-dev/c38b-forward-milestone-v01/validation.json'])
files=[]
for name in sorted(paths):
 p=ROOT/name;b=p.read_bytes();b.decode('utf-8')
 if p.suffix not in ['.py','.md','.json','.txt','.log'] or p.is_symlink():raise ValueError('Source/text only '+name)
 files.append(dict(path=name,bytes=len(b),sha256=sha(b),git_blob_sha1=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()))
manifest=dict(schema='c38b-forward-milestone-v01',base_commit='9e2f2fed4c0daea74b04055817d3185f4a7c1a92',base_tree='ea35da9a86b72ac9690cc1dad6e1852b06573925',component_inventories=inventories,files=files,file_count=len(files),total_bytes=sum(r['bytes'] for r in files),raw_arrays_or_media=False,campaign_complete=False)
p=ROOT/'runs/campaign-c38b-paths.json';p.write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(dict(files=len(files),bytes=manifest['total_bytes'],manifest_sha256=sha(p.read_bytes()))))
