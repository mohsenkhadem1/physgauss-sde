"""Frozen-recipe public-only cohort fitting and forecasts; never reads teacher worlds.

Eight independent family searches, bounded jointly. Catalogue reuse is numerical
response reuse, never reuse of old losses or selected parameters.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse, importlib.util, json, multiprocessing, shutil, time
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
from scripts.compositional_dynamics_v17.prototype import ROOT, sha, write, face_grid
from scripts.compositional_dynamics_v17 import matched_material_fit as base

METADATA_KEYS=('total_mass_kg','gravity_vector_m_s2','public_floor_z_m','frame_dt_s','train_frame_count','material_queries','initial_velocity_m_s')

def guard(bindings):
 pth=Path(bindings['protocol_path']);fth=Path(bindings['freeze_path']);ath=Path(bindings['approval_path'])
 if sha(pth)!=bindings['protocol_sha'] or sha(fth)!=bindings['freeze_sha']:raise ValueError('Protocol/freeze digest mismatch')
 p=json.loads(pth.read_text());f=json.loads(fth.read_text());a=json.loads(ath.read_text())
 if f.get('protocol_sha256')!=sha(pth) or not a.get('passed') or a.get('protocol_sha256')!=sha(pth) or a.get('method_freeze_sha256')!=sha(fth):raise ValueError('Independent approval binding required')
 recipe=p['method_recipe']['shared']
 for key in ('catalogue_plan','catalogue_results'):
  if f.get('input_sha256',{}).get(recipe[key])!=sha(ROOT/recipe[key]):raise ValueError('Catalogue manifest not frozen')
 if recipe['coarse_maxfev']!=24 or recipe['fine_maxfev']!=32 or recipe['coarse_n']!=2 or recipe['fine_n']!=3 or recipe['dt_s']!=50e-6 or recipe['workers']!=8 or recipe['seconds']!=10800:raise ValueError('Undeclared inference recipe')
 for group in ('source_sha256','input_sha256'):
  for name,h in f.get(group,{}).items():
   if sha(ROOT/name)!=h:raise ValueError('Frozen source/input mismatch')
 if f['source_sha256'].get(str(Path(__file__).relative_to(ROOT)))!=sha(__file__):raise ValueError('Fitter not source-frozen')
 return p,recipe

def load_public(folder):
 spec=importlib.util.spec_from_file_location('private_public_reader',Path(base.__file__));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.PUBLIC=folder
 metadata,points,visibility=m.permitted_inputs()
 if metadata['observations_sha256']!=sha(folder/'observations.npz'):raise ValueError('Observation digest mismatch')
 if metadata['initial_velocity_m_s']!=[0,0,0]:raise ValueError('Only declared zero initial velocity supported')
 if points.shape!=(metadata['train_frame_count'],len(metadata['material_queries']),3) or not np.isfinite(points[visibility]).all():raise ValueError('Public observation shape/finiteness')
 if not np.allclose(points[0],np.array(metadata['material_queries'])*np.array([.12,.06,.06])+np.array([0,-.03,-.03]),rtol=0,atol=1e-7):raise ValueError('Known rest geometry differs')
 return metadata,points,visibility

def catalogue(recipe,metadata):
 """Fail closed unless the original entire forward contract is unchanged."""
 planpath=ROOT/recipe['catalogue_plan'];resultpath=ROOT/recipe['catalogue_results'];p=json.loads(planpath.read_text());r=json.loads(resultpath.read_text())
 base.verify(p)
 if r['plan_sha256']!=sha(planpath) or not r['terminal'] or p['n']!=2 or p['dt_s']!=50e-6 or p['geometry_dimensions_m']!=[.12,.06,.06]:raise ValueError('Catalogue simulation identity mismatch')
 oldmetadata=json.loads((base.PUBLIC/'metadata.json').read_text())
 for key in METADATA_KEYS:
  if metadata[key]!=oldmetadata[key]:raise ValueError('Catalogue public physics/query/clock differs: '+key)
 expected={t['id']:t for t in p['tasks']}
 if len(expected)!=54:raise ValueError('Exactly54 fixed catalogue tasks required')
 rows={x['task']['id']:x for x in r['rows'] if x['task']['id'] in expected}
 if set(rows)!=set(expected):raise ValueError('Incomplete catalogue accounting')
 for key,row in rows.items():
  if row['task']!=expected[key] or row['plan_sha256']!=sha(planpath):raise ValueError('Catalogue task changed')
  folder=ROOT/row['path']
  if json.loads((folder/'complete.json').read_text())!=row:raise ValueError('Catalogue status differs')
  if 'prediction_sha256' in row and sha(folder/'prediction.npz')!=row['prediction_sha256']:raise ValueError('Catalogue prediction changed')
  if row['completed'] and 'prediction_sha256' not in row:raise ValueError('Completed catalogue has no prediction')
 return p,list(rows.values())

def verified_inputs(manifest):
 for name,h in manifest['public_input_sha256'].items():
  if sha(ROOT/name)!=h:raise ValueError('Public calibration changed')

def evaluate(task,n,folder,plan,metadata,obs,vis,deadline,binding):
 folder.mkdir(parents=True,exist_ok=False);start=time.monotonic();record=dict(task=task,n=n,completed=False,loss_m2=1e6,protocol_sha256=binding, path=str(folder.relative_to(ROOT)))
 try:
  points,diagnostics=base.simulate(task['parameters'],task['flag'],dict(plan,n=n),metadata,deadline);record.update(diagnostics)
  np.savez_compressed(folder/'prediction.npz',positions_m=points,times_s=diagnostics['saved_times_s']);record['prediction_sha256']=sha(folder/'prediction.npz')
  if record['completed']:record['loss_m2'],record['visible_vector_rms_m'],record['observed_point_count']=base.observed_loss(points,obs,vis)
 except Exception as ex:record.update(completed=False,error=f'{type(ex).__name__}: {ex}')
 record['seconds']=time.monotonic()-start;write(folder/'complete.json',record);return record

def family_worker(args):
 bindings,name,flag,deadline,manifest_path=args;p,recipe=guard(bindings);torch.set_num_threads(1);manifest=json.loads(Path(manifest_path).read_text());verified_inputs(manifest)
 public=ROOT/recipe['public_root']/name;metadata,obs,vis=load_public(public);plan,oldrows=catalogue(recipe,metadata);out=ROOT/recipe['fit_root']/name/f'family-{int(flag):+d}';out.mkdir(parents=True,exist_ok=False);rows=[];summaries=[]
 for old in oldrows:
  if old['task']['flag']!=flag:continue
  rec=dict(task=old['task'],n=2,completed=old['completed'],loss_m2=1e6,protocol_sha256=bindings['protocol_sha'],source_status_sha256=sha(ROOT/old['path']/'complete.json'),source_prediction_sha256=old.get('prediction_sha256'),source_path=old['path'],reused_forward_only=True)
  if old['completed']:
   with np.load(ROOT/old['path']/'prediction.npz')as data:
    if not np.array_equal(data['times_s'],np.arange(metadata['train_frame_count'])*metadata['frame_dt_s']):
     if not np.allclose(data['times_s'],np.arange(metadata['train_frame_count'])*metadata['frame_dt_s'],rtol=0,atol=1e-12):raise ValueError('Catalogue saved clock differs')
    rec['loss_m2'],rec['visible_vector_rms_m'],rec['observed_point_count']=base.observed_loss(data['positions_m'],obs,vis)
  else:rec['error']=old.get('error','Original failed forward')
  rows.append(rec);write(out/'catalogue'/f"{old['task']['id']}.json",rec)
 for phase,n,maxfev,xtol in [('coarse',2,24,.03),('fine',3,32,.02)]:
  eligible=[r for r in rows if r['completed'] and (r['n']==2 if phase=='coarse' else r['n']==2)]
  if not eligible: summaries.append(dict(phase=phase,error='No completed start',terminal=True));break
  start=min(eligible,key=lambda r:r['loss_m2'])['task']['parameters'];count=0;summary=dict(phase=phase,initial_parameters=start,optimizer_success=False,terminal=False)
  def objective(parameters):
   nonlocal count
   if time.monotonic()>=deadline:raise TimeoutError('Joint fit budget')
   task=dict(id=f'{phase}-{count:03d}',flag=flag,parameters=np.asarray(parameters).tolist());count+=1
   row=evaluate(task,n,out/phase/task['id'],plan,metadata,obs,vis,deadline,bindings['protocol_sha']);rows.append(row);return row['loss_m2']
  try:
   options=dict(maxfev=maxfev,xtol=xtol,ftol=.001)
   if phase=='fine':options['direc']=np.eye(3)[[0,2,1]]
   result=minimize(objective,start,method='Powell',bounds=plan['bounds'],options=options)
   summary.update(optimizer_success=bool(result.success),message=str(result.message),evaluations=int(result.nfev))
  except Exception as ex:summary['error']=f'{type(ex).__name__}: {ex}'
  summary['terminal']=True;summaries.append(summary);write(out/f'{phase}-optimizer.json',summary)
  if 'error'in summary:break
 verified_inputs(manifest);guard(bindings)
 result=dict(world_id=name,flag=flag,protocol_sha256=bindings['protocol_sha'],terminal=True,search_complete=len(summaries)==2 and all('error'not in s for s in summaries),rows=rows,optimizers=summaries,public_input_sha256={str(f.relative_to(ROOT)):sha(f)for f in [public/'metadata.json',public/'observations.npz']})
 write(out/'family-complete.json',result);return result

def require_runtime():
 if torch.__version__!='2.14.1+cu130' or np.__version__!='2.5.4':raise ValueError('Use original catalogue .venv Torch2.14.1/NumPy2.5.4 runtime')

def public_status(folder,name,bindings):
 statuspath=folder/'complete.json'
 status=json.loads(statuspath.read_text())
 if status.get('world_id')!=name or status.get('protocol_sha256')!=bindings['protocol_sha']or status.get('method_freeze_sha256')!=bindings['freeze_sha']or type(status.get('completed'))is not bool:raise ValueError('Public export identity mismatch')
 if status['completed']:
  if status.get('public_metadata_sha256')!=sha(folder/'metadata.json')or status.get('public_npz_sha256')!=sha(folder/'observations.npz'):raise ValueError('Public export data hashes mismatch')
 return status

def fit(bindings):
 require_runtime();p,recipe=guard(bindings);out=ROOT/recipe['fit_root']
 if out.exists():raise ValueError('Preserve previous cohort fit')
 ids=p['generator_settings']['world_ids'];inputs={};available=[];unavailable={}
 for name in ids:
  folder=ROOT/recipe['public_root']/name;status=public_status(folder,name,bindings);inputs[str((folder/'complete.json').relative_to(ROOT))]=sha(folder/'complete.json')
  if not status['completed']:unavailable[name]=status;continue
  metadata,_,_=load_public(folder);catalogue(recipe,metadata);available.append(name)
  for f in [folder/'metadata.json',folder/'observations.npz']:inputs[str(f.relative_to(ROOT))]=sha(f)
 out.mkdir(parents=True);manifest=out/'runtime-input-freeze.json';write(manifest,dict(protocol_sha256=bindings['protocol_sha'],method_freeze_sha256=bindings['freeze_sha'],public_input_sha256=inputs,role='Public calibration only; no private or forecast response reads'))
 deadline=time.monotonic()+recipe['seconds']-30;pool=multiprocessing.get_context('spawn').Pool(recipe['workers']);error=None
 try:
  it=pool.imap_unordered(family_worker,[(bindings,name,flag,deadline-15,str(manifest))for name in available for flag in [-1.,1.]])
  for _ in range(2*len(available)):it.next(timeout=max(.01,deadline-time.monotonic()))
 except Exception as ex:error=f'{type(ex).__name__}: {ex}'
 finally:pool.terminate();pool.join()
 allrows=[]
 for name in ids:
  families=[]
  for flag in [-1.,1.]:
   path=out/name/f'family-{int(flag):+d}'/'family-complete.json'
   if path.exists():families.append(json.loads(path.read_text()))
   else:
    partial=[json.loads(x.read_text())for x in sorted(path.parent.glob('*/*/complete.json'))]+[json.loads(x.read_text())for x in sorted((path.parent/'catalogue').glob('*.json'))]
    if any(r['protocol_sha256']!=bindings['protocol_sha']for r in partial):raise ValueError('Foreign partial fit')
    families.append(dict(world_id=name,flag=flag,terminal=False,search_complete=False,error='Missing family terminal record',rows=partial))
  candidates=[r for f in families for r in f['rows']if r['completed']and r['n']==3];complete=all(f['search_complete']for f in families);best=min(candidates,key=lambda r:r['loss_m2'])if candidates else None
  report=dict(world_id=name,terminal=True,search_complete=complete,families=families,best_training_candidate=best,protocol_sha256=bindings['protocol_sha'],public_calibration_unavailable=unavailable.get(name));write(out/name/'fit-results.json',report);allrows.append(report)
  if complete and best:write(out/name/'fitted-parameter-commit.json',dict(world_id=name,task=best['task'],n=3,dt_s=50e-6,loss_m2=best['loss_m2'],search_complete=True,results_sha256=sha(out/name/'fit-results.json'),protocol_sha256=bindings['protocol_sha'],runtime_input_freeze_sha256=sha(manifest),forecast_response_read=False))
 write(out/'complete.json',dict(terminal=True,world_ids=ids,committed=[r['world_id']for r in allrows if r['search_complete']and r['best_training_candidate']],stage_error=error,protocol_sha256=bindings['protocol_sha'],runtime_input_freeze_sha256=sha(manifest)))


def _forecast(bindings,name,role):
 require_runtime();p,recipe=guard(bindings)
 if name not in p['generator_settings']['world_ids']or role not in ('forecast-40mm','forecast-80mm'):raise ValueError('Undeclared forecast')
 fitroot=ROOT/recipe['fit_root'];commitpath=fitroot/name/'fitted-parameter-commit.json';commit=json.loads(commitpath.read_text());manifest=json.loads((fitroot/'runtime-input-freeze.json').read_text());verified_inputs(manifest)
 if commit['protocol_sha256']!=bindings['protocol_sha']or commit['results_sha256']!=sha(fitroot/name/'fit-results.json')or commit['runtime_input_freeze_sha256']!=sha(fitroot/'runtime-input-freeze.json')or not commit['search_complete']:raise ValueError('Invalid fit commit')
 public_status(ROOT/recipe['public_root']/name,name,bindings);metadata,_,_=load_public(ROOT/recipe['public_root']/name);setup=metadata['forecast_public_setups'][role];out=ROOT/recipe['forecast_root']/name/role
 if out.exists():raise ValueError('Preserve forecast')
 out.mkdir(parents=True);torch.set_num_threads(1);law=base.ScaleSeparatedMaterialLaw().double();law.load_state_dict(torch.load(base.BULK,map_location='cpu',weights_only=False)['state_dict']);law.eval()
 for param in law.parameters():param.requires_grad_(False)
 scale,shape,logk=commit['task']['parameters'];flag=commit['task']['flag'];spec=importlib.util.spec_from_file_location('private_cohort_forecast',ROOT/'scripts/compositional_dynamics_v17/prototype.py');impl=importlib.util.module_from_spec(spec);spec.loader.exec_module(impl);impl.PLAN=Path(bindings['protocol_path']);impl.RAW=out/'work';constructor=impl.DynamicContactModel
 def configured(*a,**kw):
  kw.update(gravity=tuple(setup['gravity_vector_m_s2']),bulk_energy=law.energy,material_latent_field=lambda q:q.new_tensor([scale,shape,flag]).expand(*q.shape[:-1],3));return constructor(*a,**kw)
 impl.DynamicContactModel=configured;dims=p['generator_settings']['dimensions_m'];settings=dict(dimensions=dims,young_Pa=12000.,poisson=.35,density=metadata['total_mass_kg']/np.prod(dims),floor_z=setup['floor_z_m'],penalty_N_m3=float(np.exp(logk)),compression=0.,surface_order=8,sample_interval_s=.001,duration=dict(drop=setup['duration_s']))
 row=impl.run_case('drop',3,50e-6,settings,time.monotonic()+recipe['forecast_seconds']-15);src=impl.RAW/'drop-n3-dt0.00005000'/'trajectory.npz';shutil.copyfile(src,out/'trajectory.npz');q=np.asarray(metadata['material_queries']);dense=face_grid(17);indices=[np.flatnonzero(np.all(dense==point,axis=1))[0]for point in q]
 with np.load(out/'trajectory.npz')as d:
  sampled=np.arange(0,len(d['times_s']),10);points=d['surface_frames'][sampled].reshape(len(sampled),-1,3)[:,indices];np.savez_compressed(out/'surface-prediction.npz',times_s=d['times_s'][sampled],positions_m=points,tracked_positions_m=points,material_queries=q)
  clearance=-dims[2]/2-setup['floor_z_m'];row['energy_scale_J']=max(abs(row['initial_energy_J']),metadata['total_mass_kg']*np.linalg.norm(setup['gravity_vector_m_s2'])*clearance,1e-10);row['maximum_relative_energy_drift']=float(np.max(abs(d['ledger'][:,1]-row['initial_energy_J']))/row['energy_scale_J'])
 row.update(world_id=name,role=role,fitted_parameter_commit_sha256=sha(commitpath),forecast_response_read=False);write(out/'complete.json',row);write(out/'prediction-commit.json',dict(world_id=name,role=role,method_freeze_sha256=bindings['freeze_sha'],protocol_sha256=bindings['protocol_sha'],completed=row['completed'],complete_sha256=sha(out/'complete.json'),trajectory_sha256=sha(out/'trajectory.npz'),surface_prediction_sha256=sha(out/'surface-prediction.npz'),fitted_parameter_commit_sha256=sha(commitpath),forecast_response_read=False))

def forecast(bindings,name,role):
 require_runtime();p,recipe=guard(bindings)
 if name not in p['generator_settings']['world_ids']or role not in ('forecast-40mm','forecast-80mm'):raise ValueError('Undeclared forecast')
 out=ROOT/recipe['forecast_root']/name/role
 if out.exists():raise ValueError('Preserve prior forecast attempt')
 try:_forecast(bindings,name,role)
 except Exception as ex:
  out.mkdir(parents=True,exist_ok=True)
  row=dict(world_id=name,role=role,completed=False,terminal=True,error=f'{type(ex).__name__}: {ex}',protocol_sha256=bindings['protocol_sha'],forecast_response_read=False)
  write(out/'complete.json',row);write(out/'prediction-commit.json',dict(world_id=name,role=role,completed=False,terminal=True,method_freeze_sha256=bindings.get('freeze_sha'),protocol_sha256=bindings['protocol_sha'],complete_sha256=sha(out/'complete.json'),forecast_response_read=False))

if __name__=='__main__':
 parser=argparse.ArgumentParser()
 for key in ['protocol-path','freeze-path','approval-path','protocol-sha','freeze-sha']:parser.add_argument('--'+key,required=True)
 parser.add_argument('--stage',choices=['fit','forecast'],required=True);parser.add_argument('--world');parser.add_argument('--role');args=vars(parser.parse_args());stage=args.pop('stage');name=args.pop('world');role=args.pop('role');fit(args)if stage=='fit'else forecast(args,name,role)
