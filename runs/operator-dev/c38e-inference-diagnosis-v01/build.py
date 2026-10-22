"""Completed inference negatives and forward qualification, UTF8 artifacts only."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sha=lambda b:hashlib.sha256(b).hexdigest()
paths=set();inventories=[]
for name in ['c38-beta-control-v01','c38-beta-importance-v02','p2-contact-dop853-v03',
             'p2-exact80-operator-v01','p2-warm-exact80-48-v01','p2-checkpointed-static-v01',
             'p2-warp-rhs-v01','p2-keast-operator-parity-v01','c38-static-cohort-plan-v01']:
    p=ROOT/'runs/operator-dev'/name/'publication-inventory.json'
    inventories.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p.read_bytes())))
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
for name in ['p2-supported-gravity-v01','p2-gravity-refinement-v01']:
    source=ROOT/'runs/gaussian-campaign-v01'/name/'summary.json';result=json.loads(source.read_text())
    if result.get('completed') is not True:raise ValueError('Completed controls required')
    small=compact(result);small['full_local_result']=dict(path=str(source.relative_to(ROOT)),sha256=sha(source.read_bytes()),omitted='Iteration histories only; diagnostics/capture hashes retained')
    target=source.with_name('publication-summary.json');target.write_text(json.dumps(small,indent=2)+'\n');paths.add(str(target.relative_to(ROOT)))
    paths.add(str((ROOT/'runs/operator-dev'/name/'protocol.json').relative_to(ROOT)))
paths.add('runs/operator-dev/p2-gravity-refinement-v01/execution.json')
paths.update(['README.md','docs/GAUSSIAN_C38_FORWARD_PROGRESS.md',
 'experiments/campaign_p2_supported_preconditioner.py','tests/test_campaign_p2_supported_preconditioner.py',
 'experiments/campaign_p2_support_trace.py','tests/test_campaign_p2_support_trace.py',
 'experiments/run_p2_supported_gravity_controls.py','experiments/run_p2_gravity_refinement.py',
 'experiments/campaign_p2_keast_quadrature.py','tests/test_campaign_p2_keast_quadrature.py',
 'experiments/campaign_p2_warp_rhs.py','tests/test_campaign_p2_warp_rhs.py',
 'runs/operator-dev/p2-warp-rhs-v01/prototype-receipt.json',
 'runs/operator-dev/c38e-inference-diagnosis-v01/build.py',
 'runs/operator-dev/c38e-inference-diagnosis-v01/validation.json'])
files=[]
for n in sorted(paths):
    p=ROOT/n;b=p.read_bytes();b.decode('utf-8')
    if p.is_symlink() or p.suffix not in ['.py','.md','.json','.txt','.log']:raise ValueError('Source/compact text only')
    files.append(dict(path=n,bytes=len(b),sha256=sha(b),git_blob_sha1=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()))
m=dict(schema='c38e-inference-diagnosis-v01',base_commit='52aa207758f7a9bcc11a08676ed5a68391c95a61',base_tree='a4e29df8c54fc191024efbf59b0650fe5c106ea9',component_inventories=inventories,files=files,file_count=len(files),total_bytes=sum(r['bytes'] for r in files),raw_datasets_or_media_or_model_checkpoints=False,campaign_complete=False)
p=ROOT/'runs/campaign-c38e-paths.json';p.write_text(json.dumps(m,indent=2)+'\n')
print(json.dumps(dict(files=len(files),bytes=m['total_bytes'],manifest_sha256=sha(p.read_bytes()))))
