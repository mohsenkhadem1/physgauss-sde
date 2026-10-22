"""Publish frozen source and completed diagnostics; all raw arrays stay local."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sha=lambda b:hashlib.sha256(b).hexdigest()
paths=set();inventories=[]
for name in ['p2-streaming-v01','p2-streaming-gpu-parity-v01','p2-contact-dop853-v02']:
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

name='p2-graded-static-v02';op=ROOT/'runs/operator-dev'/name
source=ROOT/'runs/gaussian-campaign-v01'/name/'summary.json'
execution=json.loads((op/'execution.json').read_text())
if execution.get('terminal') is not True:raise ValueError('Terminal graded40/48 required')
result=compact(json.loads(source.read_text()));result['execution']=execution
result['full_local_result']=dict(path=str(source.relative_to(ROOT)),sha256=sha(source.read_bytes()),omitted='Iteration histories only; metrics and capture hashes retained')
target=source.with_name('publication-summary.json');target.write_text(json.dumps(result,indent=2)+'\n')
paths.update([str(target.relative_to(ROOT)),str((op/'protocol.json').relative_to(ROOT)),str((op/'execution.json').relative_to(ROOT))])

paths.update(['README.md','docs/GAUSSIAN_C38_FORWARD_PROGRESS.md',
 'experiments/campaign_p2_jacobian_certificate.py','tests/test_campaign_p2_jacobian_certificate.py',
 'experiments/run_p2_saved_jacobian_certificate.py',
 'runs/operator-dev/p2-bernstein-saved-v01/protocol.json',
 'runs/gaussian-campaign-v01/p2-bernstein-saved-v01/summary.json',
 'experiments/run_p2_graded_static_v02.py',
 'experiments/run_p2_streaming_graded64.py',
 'runs/operator-dev/p2-streaming-graded64-v01/protocol.json',
 'experiments/continue_p2_contact_dop853_v03.py',
 'tests/test_continue_p2_contact_dop853_v03.py',
 'runs/operator-dev/p2-contact-dop853-v03/protocol.json',
 'experiments/run_p2_exact80_operator_assessment.py',
 'runs/operator-dev/p2-exact80-operator-v01/protocol.json',
 'experiments/campaign_c38_beta_posterior.py',
 'experiments/campaign_c38_beta_data.py','experiments/campaign_c38_beta_metrics.py',
 'experiments/run_campaign_c38_beta_control.py',
 'tests/test_campaign_c38_beta_posterior.py','tests/test_campaign_c38_beta_data.py',
 'tests/test_campaign_c38_beta_metrics.py','tests/test_run_campaign_c38_beta_control.py',
 'runs/operator-dev/c38-beta-control-v01/protocol-v01.json',
 'runs/operator-dev/c38d-numerical-control-v01/build.py',
 'runs/operator-dev/c38d-numerical-control-v01/validation.json'])
files=[]
for n in sorted(paths):
    p=ROOT/n;b=p.read_bytes();b.decode('utf-8')
    if p.is_symlink() or p.suffix not in ['.py','.md','.json','.txt','.log']:raise ValueError('UTF8 source/report only')
    files.append(dict(path=n,bytes=len(b),sha256=sha(b),git_blob_sha1=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()))
m=dict(schema='c38d-numerical-control-v01',base_commit='ec6312bf25f053d40457f318aaa304b05f909fb3',base_tree='aaf425d580c9bb3d8d017037d074d2522f329ca3',component_inventories=inventories,files=files,file_count=len(files),total_bytes=sum(r['bytes'] for r in files),raw_arrays_or_observations_or_media=False,campaign_complete=False,ongoing_protocols_are_not_results=True)
p=ROOT/'runs/campaign-c38d-paths.json';p.write_text(json.dumps(m,indent=2)+'\n')
print(json.dumps(dict(files=len(files),bytes=m['total_bytes'],manifest_sha256=sha(p.read_bytes()))))
