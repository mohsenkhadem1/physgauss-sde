"""Read-only post-stage process/provenance boundary, preserving every failure."""
import datetime,hashlib,json
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
STAGE=ROOT/'runs/gaussian-campaign-v01/authored-force-envelope-v01/stage1'
OUT=ROOT/'runs/operator-dev/authored-stage1-monitor-v01'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ledger=read(STAGE/'execution.json');manifest=read(STAGE/'resolved-manifest-v01.json');seal=read(STAGE/'prelaunch-seal-v01.json')
 assert ledger['terminal'] and len(ledger['jobs'])==len(manifest['cells'])==51
 assert not Path(f'/proc/{ledger["driver_pid"]}').exists(),'Parent must reap original supervisor session before terminal boundary'
 ids={c['id']:c for c in manifest['cells']};rows=[]
 for job in ledger['jobs']:
  assert job['id'] in ids and job['output']==ids[job['id']]['output']
  pid=job.get('pid')
  if pid is not None:
   assert not Path(f'/proc/{pid}').exists(),'Worker PID still present'
   execution=read(ROOT/job['output']/'execution.json')
   assert execution['terminal'] and execution['pid']==pid
   if job['state']=='completed':
    assert execution['completed'] and execution['completed_steps']==execution['planned_steps']==3*ids[job['id']]['hz']
    summary=read(ROOT/job['output']/'summary.json');assert summary['completed'] and summary['frames']==61 and summary['cell_id']==job['id']
    assert sha(ROOT/job['output']/'trajectory.npz')==summary['trajectory_sha256'] and sha(ROOT/job['output']/'steps.jsonl')==summary['steps_sha256']
   else:assert job['state']=='failed'
  else:assert job['state']=='queued','Never silently replace or retry an unlaunched job'
  rows.append({'id':job['id'],'state':job['state'],'pid':pid,'pid_missing':pid is None or not Path(f'/proc/{pid}').exists()})
 first=ROOT/ledger['jobs'][0]['output'];pins=read(first/'private-provenance/pins.json');pins[seal['stage_driver_source']['path']]=seal['stage_driver_source']['sha256']
 for p,wanted in pins.items():assert sha(ROOT/p)==wanted,p
 result={'verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stage_terminal':True,'all51_completed':ledger['completed'],
  'states':dict(Counter(j['state'] for j in ledger['jobs'])),'all_launched_worker_PIDs_missing':True,'supervisor_pid':ledger['driver_pid'],'supervisor_PID_missing':True,
  'all_frozen_source_proof_manifest_seal_resource_hashes_match':True,'stage_execution_sha256':sha(STAGE/'execution.json'),
  'first_independent_audit_passed':ledger['first_cell_audit_passed'],'first_independent_audit_sha256':sha(STAGE/'first-cell-audit-v01.json'),
  'workers':rows,'actual_force_envelope_qualified':False,'qualification':'Process/completion/provenance audit only; initial51 temporal stage is not completed119-cell qualification.'}
 p=OUT/'terminal-verification-v01.json';assert not p.exists(),'Preserve terminal proof';p.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k!='workers'},indent=2))
if __name__=='__main__':main()
