"""Read-only shared calibration/forecast audit. Never opens reference trajectories."""
import argparse,json
from pathlib import Path
import numpy as np
from scripts.compositional_dynamics_v17.prototype import ROOT,REPORT,sha,write,face_grid
BASE=ROOT/'runs/compositional-dynamics-v17/fresh-cohort'

def main(final=False):
 checks=[];evidence={};pending=[];optimizers=[]
 def check(value,message):
  checks.append(dict(passed=bool(value),check=message))
  if not value:raise ValueError(message)
 def read(path):
  evidence[str(path.relative_to(ROOT))]=sha(path);return json.loads(path.read_text())
 protocol_path=REPORT/'fresh-cohort-protocol.json';freeze_path=REPORT/'fresh-cohort-method-freeze.json';p=read(protocol_path);freeze=read(freeze_path);ph=sha(protocol_path);fh=sha(freeze_path)
 check(freeze['protocol_sha256']==ph,'Exact protocol/freeze binding')
 for group in ('source_sha256','input_sha256'):
  for name,digest in freeze[group].items():check(sha(ROOT/name)==digest,'Frozen '+group+' '+name)
 recipe=p['method_recipe']['shared'];fit=ROOT/recipe['fit_root'];manifest_path=fit/'runtime-input-freeze.json';manifest=read(manifest_path)
 check(manifest['protocol_sha256']==ph and manifest['method_freeze_sha256']==fh,'Runtime fit binding')
 for name,digest in manifest['public_input_sha256'].items():check(sha(ROOT/name)==digest,'Calibration snapshot '+name)
 original=read(ROOT/recipe['catalogue_results']);catalogue={r['task']['id']:r for r in original['rows']if r['task']['id'].startswith('grid-')};bounds=np.array(read(ROOT/recipe['catalogue_plan'])['bounds']);summary=[];totals=dict(catalogue=0,coarse=0,fine=0,completed_coarse=0,completed_fine=0)
 for world in p['generator_settings']['world_ids']:
  public=ROOT/recipe['public_root']/world;status=read(public/'complete.json');check(status['world_id']==world and status['protocol_sha256']==ph and status['method_freeze_sha256']==fh,'Public identity '+world)
  if not status['completed']:summary.append(dict(world_id=world,available=False));continue
  metadata=read(public/'metadata.json');obsfile=public/'observations.npz';check(sha(obsfile)==status['public_npz_sha256']==metadata['observations_sha256']and sha(public/'metadata.json')==status['public_metadata_sha256'],'Public data hashes '+world);evidence[str(obsfile.relative_to(ROOT))]=sha(obsfile)
  with np.load(obsfile)as f:obs=f['object_points'].astype(float);vis=f['object_visibilities']
  check(obs.shape==(21,98,3)and vis.shape==(21,98)and vis.dtype==bool,'Calibration-only21x98 contract '+world)
  def verify_prediction(row,path):
   check(sha(path)==row['prediction_sha256'],'Candidate array digest '+str(path.relative_to(ROOT)));evidence[str(path.relative_to(ROOT))]=sha(path)
   with np.load(path)as f:x=f['positions_m'];t=f['times_s']
   check(x.shape[1:]==(98,3)and len(x)==len(t)and np.isfinite(x).all(),'Finite candidate prefix '+str(path.relative_to(ROOT)))
   if row['completed']:
    check(x.shape==obs.shape and np.allclose(t,np.arange(21)*.01,rtol=0,atol=1e-12),'Complete candidate clock')
    delta=(x[1:]-obs[1:])[vis[1:]];loss=float(np.mean(delta**2));vrms=float(np.sqrt(np.mean(np.sum(delta**2,axis=-1))))
    return loss,vrms,int(vis[1:].sum())
   return None
  allfine=[];families=[]
  for flag in (-1.,1.):
   folder=fit/world/f'family-{int(flag):+d}';rows=[];phase_rows={}
   cached=[]
   for path in sorted((folder/'catalogue').glob('*.json')):
    row=read(path);task=row['task'];old=catalogue[task['id']];source=ROOT/row['source_path'];check(row['protocol_sha256']==ph and task==old['task']and task['flag']==flag and row['n']==2,'Reused candidate identity')
    check(sha(source/'complete.json')==row['source_status_sha256']and read(source/'complete.json')==old,'Original cached status')
    check(row['completed']==old['completed']and row['source_prediction_sha256']==old.get('prediction_sha256'),'Cache completion unchanged')
    metrics=verify_prediction(old,source/'prediction.npz')if 'prediction_sha256'in old else None
    if row['completed']:
     check(np.isclose(row['loss_m2'],metrics[0],rtol=1e-12,atol=1e-20)and np.isclose(row['visible_vector_rms_m'],metrics[1],rtol=1e-12,atol=1e-15)and row['observed_point_count']==metrics[2],'Recomputed fresh visibility loss for reused forward')
    else:check(row['loss_m2']==1e6,'Failed cache retains penalty')
    cached.append(row);rows.append(row)
   check(len(cached)==27,'Full27-candidate family catalogue '+world+str(flag));totals['catalogue']+=len(cached)
   for phase,n,budget in [('coarse',2,24),('fine',3,32)]:
    found=[]
    for path in sorted((folder/phase).glob('*/complete.json')):
     row=read(path);task=row['task'];params=np.array(task['parameters']);check(row['protocol_sha256']==ph and task['flag']==flag and row['n']==n and task['id']==path.parent.name,'New candidate identity')
     check(params.shape==(3,)and np.isfinite(params).all()and np.all(params>=bounds[:,0]-1e-12)and np.all(params<=bounds[:,1]+1e-12),'Candidate parameter bounds')
     metrics=verify_prediction(row,path.parent/'prediction.npz')if 'prediction_sha256'in row else None
     if row['completed']:
      check(metrics is not None and np.isclose(row['loss_m2'],metrics[0],rtol=1e-12,atol=1e-20)and np.isclose(row['visible_vector_rms_m'],metrics[1],rtol=1e-12,atol=1e-15)and row['observed_point_count']==metrics[2],'Independent candidate calibration objective')
     else:check(row['loss_m2']==1e6,'Failed candidate explicit penalty')
     found.append(row);rows.append(row)
    check(len(found)<=budget and {r['task']['id']for r in found}=={f'{phase}-{i:03d}'for i in range(len(found))},'Contiguous bounded '+phase+' evaluations')
    totals[phase]+=len(found);totals['completed_'+phase]+=sum(r['completed']for r in found);phase_rows[phase]=found
    optimizer=folder/f'{phase}-optimizer.json'
    if optimizer.exists():
     opt=read(optimizer);optimizers.append(dict(world_id=world,flag=flag,phase=phase,optimizer_success=opt.get('optimizer_success',False),evaluations=opt.get('evaluations'),message=opt.get('message'),error=opt.get('error')));eligible=[r for r in (cached if phase=='coarse'else cached+phase_rows['coarse'])if r['completed']]
     check(opt['initial_parameters']==min(eligible,key=lambda r:r['loss_m2'])['task']['parameters'],'Training-only '+phase+' initialization')
     if 'evaluations'in opt:check(opt['evaluations']==len(found),'Optimizer evaluation accounting')
    else:pending.append(str(optimizer.relative_to(ROOT)))
   terminal=folder/'family-complete.json'
   if terminal.exists():
    report=read(terminal);check(report['protocol_sha256']==ph and report['world_id']==world and report['flag']==flag and report['terminal'],'Family completion identity')
    check({r['task']['id']:r for r in report['rows']}=={r['task']['id']:r for r in rows},'Full candidate ledger retained')
    families.append(report)
   else:pending.append(str(terminal.relative_to(ROOT)))
   allfine += [r for r in phase_rows['fine']if r['completed']]
  commit=fit/world/'fitted-parameter-commit.json'
  if commit.exists():
   c=read(commit);result_path=fit/world/'fit-results.json';result=read(result_path);check(c['protocol_sha256']==ph and c['results_sha256']==sha(result_path)and c['runtime_input_freeze_sha256']==sha(manifest_path),'Selected fit provenance')
   check(len(families)==2 and all(f['search_complete']for f in families)and c['search_complete'],'Both family searches finished')
   best=min(allfine,key=lambda r:r['loss_m2']);check(c['task']==best['task']and c['loss_m2']==best['loss_m2']and result['best_training_candidate']==best,'Global fine calibration minimum selected without targets')
   summary.append(dict(world_id=world,available=True,fit_committed=True,selected_task=c['task'],calibration_loss_m2=c['loss_m2']))
  else:
   terminal_fit=fit/world/'fit-results.json'
   if terminal_fit.exists():
    failed=read(terminal_fit);check(failed['terminal']and failed['protocol_sha256']==ph,'Terminal unsuccessful fit accounting')
    check(not(failed['search_complete']and failed['best_training_candidate']),'A valid complete fit must be committed')
    summary.append(dict(world_id=world,available=True,fit_committed=False,terminal_failure=True))
   else:pending.append(str(commit.relative_to(ROOT)));summary.append(dict(world_id=world,available=True,fit_committed=False))
 forecast_rows=[]
 for world in p['generator_settings']['world_ids']:
  for role in ('forecast-40mm','forecast-80mm'):
   folder=ROOT/recipe['forecast_root']/world/role;commit_path=folder/'prediction-commit.json'
   if not commit_path.exists():pending.append(str(commit_path.relative_to(ROOT)));forecast_rows.append(dict(world_id=world,role=role,terminal=False));continue
   c=read(commit_path);check(c['world_id']==world and c['role']==role and c['protocol_sha256']==ph and c['method_freeze_sha256']==fh and type(c['completed'])is bool,'Forecast seal identity')
   row=dict(world_id=world,role=role,terminal=True,completed=c['completed'],error=c.get('error'))
   for filename,key in [('complete.json','complete_sha256'),('trajectory.npz','trajectory_sha256'),('surface-prediction.npz','surface_prediction_sha256')]:
    if key in c:
     artifact=folder/filename;check(sha(artifact)==c[key],'Forecast success/failure artifact integrity');evidence[str(artifact.relative_to(ROOT))]=sha(artifact)
   if c['completed']:
    status_path=folder/'complete.json';array=folder/'surface-prediction.npz';full=folder/'trajectory.npz';check(sha(status_path)==c['complete_sha256']and sha(array)==c['surface_prediction_sha256']and sha(full)==c['trajectory_sha256'],'Forecast sealed arrays')
    status=read(status_path);fitcommit=fit/world/'fitted-parameter-commit.json';check(status['fitted_parameter_commit_sha256']==c['fitted_parameter_commit_sha256']==sha(fitcommit),'Forecast committed parameter provenance')
    with np.load(array)as f:t=f['times_s'];x=f['positions_m'];q=f['material_queries'];tracked=f['tracked_positions_m']
    check(x.shape==(31,98,3)and np.array_equal(x,tracked)and np.isfinite(x).all()and np.allclose(t,np.arange(31)*.01,rtol=0,atol=1e-12),'Full finite31-frame forecast clock')
    with np.load(full)as f:
     times=f['times_s'];frames=f['surface_frames'];dense=face_grid(17);indices=[np.flatnonzero(np.all(dense==point,axis=1))[0]for point in q]
     check(len(times)==301 and np.allclose(times,np.arange(301)*.001,rtol=0,atol=1e-12),'FullQ2 prediction clock')
     check(np.array_equal(x,frames[::10].reshape(31,-1,3)[:,indices]),'Public forecast is exact fullQ2 material query')
    for f in (array,full):evidence[str(f.relative_to(ROOT))]=sha(f)
   forecast_rows.append(row)
 terminal_path=fit/'complete.json'
 if terminal_path.exists():check(read(terminal_path)['terminal'],'Fit supervisor terminal')
 else:pending.append(str(terminal_path.relative_to(ROOT)))
 if final:check(not pending,'Final audit has every fit/forecast terminal artifact')
 out=REPORT/('fresh-cohort-shared-final-independent-audit.json'if final else'fresh-cohort-shared-progress-independent-audit.json')
 if final and out.exists():raise ValueError('Preserve final audit')
 write(out,dict(passed=True,final=final,checks=len(checks),protocol_sha256=ph,method_freeze_sha256=fh,budgets=dict(cached_forward_rows=216,maximum_coarse_new_evaluations=192,maximum_fine_new_evaluations=256),counts=totals,optimizers=optimizers,optimizer_converged_count=sum(o['optimizer_success']for o in optimizers),optimizer_terminal_count=len(optimizers),worlds=summary,forecasts=forecast_rows,pending=pending,scope='Calibration-only independent objective/selection and prediction commitment audit. No reference/target arrays read and no forecast errors scored. Optimization maximum budgets differ from convergence; all failures preserved.',evidence_sha256=evidence,source_sha256={str(Path(__file__).relative_to(ROOT)):sha(__file__)}))
 print(json.dumps(dict(output=str(out),checks=len(checks),counts=totals,pending=len(pending))))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');main(p.parse_args().final)
