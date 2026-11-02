"""Prospective secondary reaction replay, fixed primary parameters and policies."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,concurrent.futures,json,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from scripts.compositional_neural_mechanics import unknown_action_campaign as primary
from scripts.compositional_neural_mechanics.full_continuous_inference_pilot import build,pack
from scripts.compositional_neural_mechanics.conditional_inference_pilot import POINTS,predict
from scripts.compositional_neural_mechanics.mechanics import GeometrySpec
from scripts.compositional_neural_mechanics.dataset_runner import write,npz
from scripts.compositional_neural_mechanics_reporting.legacy_dependency_audit import closure,sha
ROOT=primary.ROOT;R=primary.R;BASE=ROOT/'runs/compositional-neural-mechanics-v16';OUT=BASE/'unknown-action-secondary-reactions';PLAN=R/'unknown-reaction-secondary-plan.json';AUDIT=R/'unknown-action-final-independent-audit.json'

def prepare():
 if PLAN.exists()or OUT.exists():raise ValueError('Preserve prospective secondary plan')
 primary.guard()
 plan=dict(scope=__doc__,declared_unix=time.time(),timing='Declared while primary unknown6 fits running, before this agent reads any primary outcomes. Secondary reporting analysis, not frozen primary metric.',parent_plan_sha256=sha(primary.PLAN),source_sha256=closure([Path(__file__).resolve()]),objects=6,models=list(primary.MODELS),states=['initial',*primary.ACTIONS],commands=2,expected_policy_rows=144,prediction_n=2,reference_n=4,workers=4,stage_seconds=1800,per_job_seconds=300,force_observation_count=0,rule='Replay exact initial/aftercandidate primary fitted parameters; no refitting, modelselection or policyselection. Preserve failures. Same settledcommands/geometry.',gate=dict(path=str(AUDIT.relative_to(ROOT)),passed=True,current_evidence_sha256=True),comparison='Forces inN andtorques inNm aboutsame declared toolcenters. No forceuncertaintyband claim. Primaryposition results remainunchanged.')
 write(PLAN,plan)

def guard(require_audit=True):
 p=json.loads(PLAN.read_text());primary.guard()
 if p['parent_plan_sha256']!=sha(primary.PLAN):raise ValueError('Primary freeze changed')
 for path,h in p['source_sha256'].items():
  if sha(ROOT/path)!=h:raise ValueError('Secondary source changed')
 if require_audit:
  a=json.loads(AUDIT.read_text())
  if not a.get('passed')or a.get('errors'):raise ValueError('Independent primary audit required')
  if str(primary.PLAN.relative_to(ROOT))not in a['evidence_sha256']or str((R/'unknown-action-reserved-results.json').relative_to(ROOT))not in a['evidence_sha256']:raise ValueError('Audit mustbindprimaryfreeze/results')
  for path,h in a['evidence_sha256'].items():
   if sha(ROOT/path)!=h:raise ValueError('Independent primary audit stale')
 return p

def worker(world,model):
 torch.set_num_threads(1);p=guard();public=json.loads((primary.B/f'world{world}/public.json').read_text());g=GeometrySpec(**public['geometry']);commands=[np.asarray(x)for x in public['held_poses']];dummy=[dict(pose=np.zeros((2,6)),points=POINTS,displacement=np.zeros_like(POINTS))];folder=OUT/f'world{world}'/model;folder.mkdir(parents=True,exist_ok=False);rows=[]
 if model=='reference':
  committed=json.loads((OUT/'predictions-terminal.json').read_text())
  if committed['plan_sha256']!=sha(PLAN)or len(committed['jobs'])!=12 or {(x['world'],x['model'])for x in committed['jobs']}!={(i,m)for i in range(6)for m in primary.MODELS}:raise ValueError('All secondary predictions terminal before reference replay')
  scene=json.loads((primary.B/f'world{world}/private/scene.json').read_text());parameters={'reference':pack(scene)};fit_status={};e=build(g,public['categories'],dummy,False);e.kw['n']=4
 else:
  base=primary.B/f'world{world}'/model;e=build(g,public['categories'],dummy,model=='shared_laws_fitted');parameters={};fit_status={}
  if (base/'initial-fit.json').exists():fit_status['initial']=json.loads((base/'initial-fit.json').read_text())
  if (base/'fits.json').exists():fit_status.update(json.loads((base/'fits.json').read_text()))
  parameters={k:v.get('parameters')for k,v in fit_status.items()}
 for state in ('reference',)if model=='reference'else p['states']:
  for j,pose in enumerate(commands):
   row=dict(world=world,model=model,state_name=state,command=j,primary_fit_incomplete=fit_status.get(state,{}).get('incomplete'),primary_optimizer_success=fit_status.get(state,{}).get('optimizer_success'))
   try:
    theta=parameters.get(state)
    if theta is None:raise ValueError('Primary fit has no finite parameter estimate')
    theta=np.asarray(theta,float)
    if theta.shape!=(15,)or not np.isfinite(theta).all():raise ValueError('Invalid primary parameter estimate')
    value=predict(e,theta,pose);wrench=np.concatenate([value['forces'],value['torques']],axis=-1)
    if wrench.shape!=(2,6)or not np.isfinite(wrench).all():raise ValueError('Nonfinite reaction')
    path=folder/f'{state}-{j}.npz';npz(path,wrenches=wrench,origins_world_m=g.map(np.array([[0.,.5,.5],[1.,.5,.5]]))+pose[:,:3]);row.update(state='complete',artifact=str(path.relative_to(ROOT)),sha256=sha(path))
   except Exception as error:row.update(state='failed',error=repr(error))
   rows.append(row);write(folder/'progress.json',dict(plan_sha256=sha(PLAN),rows=rows))
 write(folder/'complete.json',dict(plan_sha256=sha(PLAN),audit_sha256=sha(AUDIT),rows=rows))

def validate_completion_rows(rows,world,model):
 expected={(world,model,state,j)for state in (('reference',)if model=='reference'else ('initial',*primary.ACTIONS))for j in (0,1)}
 keys=[(r['world'],r['model'],r['state_name'],r['command'])for r in rows]
 if len(keys)!=len(expected)or set(keys)!=expected or any(r['state']not in ('complete','failed')for r in rows):raise ValueError('Exact unique replay row census required')

def validate_origins(origins,public,command):
 g=GeometrySpec(**public['geometry']);pose=np.asarray(public['held_poses'][command]);expected=g.map(np.array([[0.,.5,.5],[1.,.5,.5]]))+pose[:,:3]
 if np.shape(origins)!=(2,3)or not np.isfinite(origins).all()or not np.allclose(origins,expected,rtol=0,atol=1e-12):raise ValueError('Wrench origins differ from declared tool centers')
 return np.asarray(origins)

def score():
 p=guard();status=json.loads((OUT/'terminal.json').read_text());expected={(i,m)for i in range(6)for m in [*primary.MODELS,'reference']}
 if len(status['jobs'])!=18 or {(x['world'],x['model'])for x in status['jobs']}!=expected or status['plan_sha256']!=sha(PLAN):raise ValueError('Exact18terminal jobs required')
 saved={};origins={};evidence={str(AUDIT.relative_to(ROOT)):sha(AUDIT)}
 for i,m in expected:
  folder=OUT/f'world{i}'/m;path=folder/'complete.json'
  if not path.exists():continue
  c=json.loads(path.read_text())
  if c['plan_sha256']!=sha(PLAN)or c['audit_sha256']!=sha(AUDIT):raise ValueError('Replay commitment changed')
  evidence[str(path.relative_to(ROOT))]=sha(path)
  validate_completion_rows(c['rows'],i,m)
  public=json.loads((primary.B/f'world{i}/public.json').read_text())
  for row in c['rows']:
   if row['state']=='complete':
    if sha(ROOT/row['artifact'])!=row['sha256']:raise ValueError('Reaction artifact changed')
    with np.load(ROOT/row['artifact'],allow_pickle=False)as a:
     if set(a.files)!={'wrenches','origins_world_m'}or a['wrenches'].shape!=(2,6)or not np.isfinite(a['wrenches']).all():raise ValueError('Finite wrench schema required')
     key=(i,m,row['state_name'],row['command']);saved[key]=a['wrenches'].copy();origins[key]=validate_origins(a['origins_world_m'],public,row['command']).copy()
    evidence[row['artifact']]=row['sha256']
 rows=[]
 for i in range(6):
  for m in primary.MODELS:
   path=primary.B/f'world{i}'/m/'selection.json';selection=json.loads(path.read_text())['policies']if path.exists()else{}
   for policy in ['initial',*json.loads(primary.PLAN.read_text())['policies']]:
    state='initial'if policy=='initial'else selection.get(policy)
    for j in (0,1):
     row=dict(world=i,model=m,policy=policy,action=state,command=j);a=saved.get((i,m,state,j));b=saved.get((i,'reference','reference',j))
     if a is None or b is None:row.update(state='failed',error='Missing finite committed reaction/reference')
     else:
      if not np.allclose(origins[i,m,state,j],origins[i,'reference','reference',j],rtol=0,atol=1e-12):raise ValueError('Prediction/reference torque origins differ')
      row.update(state='complete',force_coordinate_rmse_N=float(np.sqrt(np.mean((a[:,:3]-b[:,:3])**2))),torque_coordinate_rmse_Nm=float(np.sqrt(np.mean((a[:,3:]-b[:,3:])**2))))
     rows.append(row)
 if len(rows)!=144:raise ValueError('Fixed144rows required')
 write(R/'unknown-reaction-secondary-results.json',dict(plan_sha256=sha(PLAN),scope=p['scope'],evidence_sha256=evidence,rows=rows))

def run():
 p=guard();OUT.mkdir(parents=True,exist_ok=False);deadline=time.time()+1800;write(OUT/'start.json',dict(plan_sha256=sha(PLAN),audit_sha256=sha(AUDIT),deadline_unix=deadline));jobs=[]
 def one(job):
  i,m=job
  if time.time()>=deadline:return dict(world=i,model=m,exit_code=None,error='Stage deadline')
  with (OUT/f'{i}-{m}.log').open('x')as log:
   child=subprocess.Popen([sys.executable,'-m','scripts.compositional_neural_mechanics_reporting.unknown_reaction_secondary','--worker',str(i),'--model',m],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   try:code=child.wait(timeout=max(.01,min(300,deadline-time.time())))
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait()
  return dict(world=i,model=m,exit_code=code)
 # Commit ALL predicted reactions before secondary private reference replay.
 with concurrent.futures.ThreadPoolExecutor(max_workers=4)as pool:jobs.extend(pool.map(one,[(i,m)for i in range(6)for m in primary.MODELS]))
 write(OUT/'predictions-terminal.json',dict(plan_sha256=sha(PLAN),jobs=jobs))
 with concurrent.futures.ThreadPoolExecutor(max_workers=4)as pool:jobs.extend(pool.map(one,[(i,'reference')for i in range(6)]))
 write(OUT/'terminal.json',dict(plan_sha256=sha(PLAN),jobs=jobs));score()
if __name__=='__main__':
 parser=argparse.ArgumentParser();g=parser.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--run',action='store_true');g.add_argument('--worker',type=int);parser.add_argument('--model',choices=[*primary.MODELS,'reference']);a=parser.parse_args()
 if a.prepare:prepare()
 elif a.run:run()
 else:worker(a.worker,a.model)
