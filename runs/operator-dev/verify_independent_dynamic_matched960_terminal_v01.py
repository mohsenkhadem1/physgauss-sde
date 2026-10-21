"""Strict terminal boundary for eight matched controls, no launches or edits."""
import datetime,hashlib,json
from pathlib import Path
ROOT=Path('runs/gaussian-campaign-v01/independent-dynamic-gate-v01/matched960-stage-v01')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 d=read(ROOT/'execution.json');prep=read(ROOT/'preparation.json');manifest=read(ROOT/'matched-manifest-v01.json')
 assert d['terminal'] and d['completed'] and len(d['jobs'])==len(manifest['cells'])==8
 assert not Path(f'/proc/{d["driver_pid"]}').exists(),'Poll supervisor session before this terminal verification'
 rows=[]
 for row in d['jobs']:
  assert row['state']=='completed' and row['exit_code']==0
  folder=Path(row['output']);e=read(folder/'execution.json');p=read(folder/'protocol.json')
  assert e['terminal'] and e['completed'] and e['completed_steps']==2880
  assert p['source_input_dependency_and_gate_sha256']==row['expected_protocol_sha256']
  assert not Path(f'/proc/{row["pid"]}').exists(),'Worker remains live'
  rows.append({'id':row['id'],'pid':row['pid'],'pid_missing':True,'completed_steps':e['completed_steps'],'execution_sha256':sha(folder/'execution.json')})
 for path,wanted in prep['source_input_helper_proof_pins'].items():assert sha(path)==wanted,path
 result={'verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'all8cells_completed':True,
  'all8worker_PIDs_missing':True,'supervisor_pid':d['driver_pid'],'supervisor_PID_missing':True,
  'frozen_source_input_helper_proof_hashes_still_match':True,'manifest_sha256':sha(ROOT/'matched-manifest-v01.json'),
  'execution_sha256':sha(ROOT/'execution.json'),'workers':rows,'qualification':'Process/provenance audit only; full60-cell mechanics/comparison audit required separately.'}
 (ROOT/'terminal-verification-v01.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k!='workers'},indent=2))
if __name__=='__main__':main()
