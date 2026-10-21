"""Eight prospective matched960 TRAIN cells. --prepare never launches physics.

Requires explicit parent C14 publication confirmation before --launch. Exactly
four workers maximum, two CPU threads per process, no retries or replacement.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse,hashlib,json,os,subprocess,threading,time
from pathlib import Path
from experiments import campaign_dynamic_960_protocol as guard

BASE=Path('runs/gaussian-campaign-v01/independent-dynamic-gate-v01')
ROOT=BASE/'matched960-stage-v01'
RUNNER=Path('experiments/campaign_nonlinear_dynamics_case.py')
INPUTS=Path('configs/gaussian_independent_diagnostic_inputs_v01.json')
RUNNER_SHA='e7b6b785c262f43d4b1acc503b14feaf1cad659c171024346fc9e70a4dffc342'
HELPER_SHA='98576cc72c8172e6a0766d0fd4f2ddfb8f900d54ca51dd96e270faca8f85edcb'
AUDITOR=Path('experiments/campaign_dynamic_960_audit.py')
AUDITOR_SHA='29cb5e7f9e41822c43a2c2d3aa62a5825e2767b9357b2be04eed0bf7c9abb0ca'


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
 p=Path(p);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)
def validate_sources():
 assert sha(guard.PROTOCOL_PATH)==guard.PROTOCOL_SHA256 and sha(RUNNER)==RUNNER_SHA and sha(guard.__file__)==HELPER_SHA and sha(AUDITOR)==AUDITOR_SHA
 declaration=read(guard.PROTOCOL_PATH)
 pins={str(guard.PROTOCOL_PATH):guard.PROTOCOL_SHA256,str(RUNNER):RUNNER_SHA,str(Path(guard.__file__).resolve()):HELPER_SHA,str(AUDITOR):AUDITOR_SHA,
  **declaration['physical_source_input_sha256'],**declaration['proof_source_sha256'],
  **{p['path']:p['sha256'] for p in declaration['completed_proofs'].values()}}
 for path,wanted in pins.items():assert sha(path)==wanted,'Frozen source/input/proof differs: '+path
 terminal=read(BASE/'conditional-stage-v01/terminal-verification-v01.json')
 assert terminal['all4cells_completed'] and terminal['all4worker_PIDs_missing'] and terminal['supervisor_PID_missing']
 assert not Path(f'/proc/{terminal["supervisor_pid"]}').exists()
 assert all(not Path(f'/proc/{row["pid"]}').exists() for row in terminal['workers'])
 return declaration,pins

def command(row):
 return ['.venv/bin/python','-u',str(RUNNER),'--inputs-json',str(INPUTS),'--output',row['output'],
  '--episode','t0003','--hz','960','--response',row['response'],'--relative-tolerance',str(row['relative_tolerance']),
  '--absolute-tolerance-N',str(row['absolute_tolerance_N']),'--translation',*map(str,row['translation_m']),
  '--matched-960-protocol',str(guard.PROTOCOL_PATH),'--study-cell-id',row['id']]

def request_metadata(row):
 return guard.validate_matched_960(guard.PROTOCOL_PATH,study_cell_id=row['id'],episode='t0003',hz=960,
  response=row['response'],relative_tolerance=row['relative_tolerance'],absolute_tolerance_N=row['absolute_tolerance_N'],
  maximum_iterations=250,translation=row['translation_m'],inputs_json=INPUTS,
  core_path=RUNNER.with_name('campaign_nonlinear_dynamics.py'),gate_path='configs/gaussian_campaign_independent_dynamic_gate_v01.json')

def expected_protocol(row):
 expected=read(BASE/'baseline-matrix-v01.json')['frozen_source_input_and_gate_sha256']
 expected={p:(RUNNER_SHA if Path(p).name==RUNNER.name else wanted) for p,wanted in expected.items()}
 metadata=request_metadata(row)
 expected.update({path:sha(path) for path in metadata['provenance_dependency_paths']})
 expected[str(Path(guard.__file__).resolve())]=HELPER_SHA
 return expected

def prepare():
 declaration,pins=validate_sources();ROOT.mkdir(parents=True,exist_ok=False);(ROOT/'logs').mkdir()
 cells=[dict(row,case='t0003',hz=960,output=str(ROOT/row['id'])) for row in declaration['new_cells']]
 assert len(cells)==8
 manifest={'matched_protocol_sha256':guard.PROTOCOL_SHA256,'approved_runner_source_sha256':[RUNNER_SHA],
  'matched_dependency_sha256':{'campaign_dynamic_960_protocol.py':HELPER_SHA},
  'completed_proof_sha256':{key:item['sha256'] for key,item in declaration['completed_proofs'].items()},'cells':cells}
 write(ROOT/'matched-manifest-v01.json',manifest)
 (ROOT/'driver-source.py').write_bytes(Path(__file__).read_bytes())
 write(ROOT/'preparation.json',{'prepared_unix_s':time.time(),'launched':False,
  'requires':'Explicit parent C14 publication confirmation before --launch','max_workers':4,'cpu_threads_each':2,
  'driver_source_sha256':sha(__file__),'manifest_sha256':sha(ROOT/'matched-manifest-v01.json'),
  'source_input_helper_proof_pins':pins,'cells':[dict(row,command=command(row),expected_protocol_sha256=expected_protocol(row)) for row in cells]})
 print(json.dumps({'prepared':True,'launched':False,'cells':len(cells),'manifest':str(ROOT/'matched-manifest-v01.json')}),flush=True)

def launch(publication):
 if len(publication)!=40 or any(c not in '0123456789abcdef' for c in publication):raise ValueError('Exact confirmed C14 publication commit required')
 prep=read(ROOT/'preparation.json');assert sha(__file__)==prep['driver_source_sha256'];validate_sources()
 assert sha(ROOT/'matched-manifest-v01.json')==prep['manifest_sha256']
 assert not (ROOT/'execution.json').exists(),'Existing execution must be preserved'
 for row in prep['cells']:assert not Path(row['output']).exists(),'Preserve existing output; no retry'
 rows=[dict(row,state='planned') for row in prep['cells']];lock=threading.Lock();stop=threading.Event()
 ledger={'driver_pid':os.getpid(),'publication_commit':publication,'started_unix_s':time.time(),'terminal':False,'completed':False,
  'preparation_sha256':sha(ROOT/'preparation.json'),'manifest_sha256':prep['manifest_sha256'],'max_workers':4,'cpu_threads_each':2,'jobs':rows}
 def save():write(ROOT/'execution.json',ledger)
 save()
 def run(row):
  if stop.is_set():
   with lock:row.update(state='not_launched_after_failure');save()
   return
  try:
   validate_sources();request_metadata(row)
   env={**os.environ,'OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
   with (ROOT/'logs'/(row['id']+'.log')).open('x') as log:
    process=subprocess.Popen(row['command'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env)
    with lock:row.update(state='running',pid=process.pid,started_unix_s=time.time());save();print(json.dumps({'started':row['id'],'pid':process.pid}),flush=True)
    code=process.wait()
   row['exit_code']=code
   if code:raise RuntimeError(f'Declared cell exited {code}; failure preserved, no retry')
   out=Path(row['output']);execution=read(out/'execution.json');protocol=read(out/'protocol.json');summary=read(out/'summary.json')
   assert execution['terminal'] and execution['completed'] and summary['completed'] and execution['completed_steps']==2880
   assert protocol['source_input_dependency_and_gate_sha256']==row['expected_protocol_sha256'],'Provenance mismatch'
   assert protocol['matched_960_protocol']==request_metadata(row),'Exact matched request metadata differs'
   assert protocol['supplemental_accuracy_protocol'] is None and protocol['conditional_temporal_protocol'] is None
   assert (protocol['episode_id'],protocol['physics_hz'],protocol['response'],protocol['relative_nonlinear_residual_tolerance'],protocol['absolute_residual_floor_N'],protocol['translation_m'])==('t0003',960,row['response'],row['relative_tolerance'],row['absolute_tolerance_N'],row['translation_m'])
   with lock:
    row.update(state='completed',finished_unix_s=time.time(),elapsed_s=summary['elapsed_s'],summary_sha256=sha(out/'summary.json'));save()
    print(json.dumps({'finished':row['id'],'elapsed_s':summary['elapsed_s'],'nonlinear_iterations':summary['nonlinear_iterations']}),flush=True)
  except BaseException as exc:
   stop.set()
   with lock:row.update(state='failed',error=repr(exc),finished_unix_s=time.time());save();print(json.dumps({'failed':row['id'],'error':repr(exc)}),flush=True)
 with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(run,rows))
 ledger.update(terminal=True,completed=all(row['state']=='completed' for row in rows),finished_unix_s=time.time());save()
 print(json.dumps({'terminal':True,'completed':ledger['completed']}),flush=True)
 if not ledger['completed']:raise SystemExit(1)

if __name__=='__main__':
 parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group(required=True);mode.add_argument('--prepare',action='store_true');mode.add_argument('--launch',action='store_true');parser.add_argument('--published-commit');args=parser.parse_args()
 if args.prepare:prepare()
 else:launch(args.published_commit or '')
