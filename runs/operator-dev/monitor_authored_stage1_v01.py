"""Read-only existing-job snapshot; never launches/restarts/edits a worker."""
import datetime,hashlib,json
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
STAGE=ROOT/'runs/gaussian-campaign-v01/authored-force-envelope-v01/stage1'
OUT=ROOT/'runs/operator-dev/authored-stage1-monitor-v01'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def proc(pid):
 p=Path(f'/proc/{pid}/stat')
 try:return p.read_text().rsplit(')',1)[1].split()[0]
 except FileNotFoundError:return 'missing'
def write(p,d):
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)
def main():
 OUT.mkdir(exist_ok=True)
 ledger=read(STAGE/'execution.json');rows=[]
 for job in ledger['jobs']:
  if job['state']=='running':
   path=ROOT/job['output']/'execution.json';state=read(path) if path.is_file() else {}
   rows.append({'id':job['id'],'pid':job['pid'],'proc_state':proc(job['pid']),'completed_steps':state.get('completed_steps'),
    'planned_steps':state.get('planned_steps'),'last_residual_N':state.get('last_residual_N')})
 first=ROOT/ledger['jobs'][0]['output'];pins=read(first/'private-provenance/pins.json');changed=[p for p,w in pins.items() if sha(ROOT/p)!=w]
 # Runner pins bind seal/resources/manifest/auditor/physics; driver source has its
 # separate explicit record in the immutable seal.
 seal=read(STAGE/'prelaunch-seal-v01.json') if (STAGE/'prelaunch-seal-v01.json').exists() else None
 if seal is not None:
  record=seal['stage_driver_source']
  if sha(ROOT/record['path'])!=record['sha256']:changed.append(record['path'])
 mem=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
 snapshot={'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'driver_pid':ledger['driver_pid'],'driver_proc_state':proc(ledger['driver_pid']),
  'terminal':ledger['terminal'],'completed':ledger['completed'],'first_cell_audit_passed':ledger['first_cell_audit_passed'],
  'counts':dict(Counter(j['state'] for j in ledger['jobs'])),'running':rows,'load_average':Path('/proc/loadavg').read_text().split()[:3],
  'MemAvailable_kB':int(mem['MemAvailable'].split()[0]),'sealed_dependency_hash_changes':changed,'physical_restart_performed':False}
 write(OUT/'progress.json',snapshot)
 with (OUT/'history.jsonl').open('a') as f:f.write(json.dumps(snapshot)+'\n')
 firstfile=OUT/'first-cell-outcome.json'
 if not firstfile.exists() and ledger['first_cell_audit_passed']:
  a=read(STAGE/'first-cell-audit-v01.json');s=read(first/'summary.json')
  write(firstfile,{'id':ledger['jobs'][0]['id'],'audit_passed':a['passed'],'frames':s['frames'],'elapsed_s':s['elapsed_s'],
   'minimum_runtime_J':a['minimum_runtime_J'],'maximum_accepted_residual_N':a['maximum_accepted_residual_N'],
   'maximum_nodal_displacement_m':s['endpoint_metric_ranges']['maximum_nodal_displacement_m'][1],
   'maximum_frame_clock_error_s':s['maximum_frame_clock_error_s'],'maximum_handle_work_error_J':a['maximum_handle_work_error_J'],
   'maximum_fixture_work_error_J':a['maximum_fixture_work_error_J'],'audit_sha256':sha(STAGE/'first-cell-audit-v01.json'),
   'summary_sha256':sha(first/'summary.json'),'actual_force_envelope_qualified':False,'single_cell_is_not_temporal_or_dataset_qualification':True})
 print(json.dumps(snapshot),flush=True)
if __name__=='__main__':main()
