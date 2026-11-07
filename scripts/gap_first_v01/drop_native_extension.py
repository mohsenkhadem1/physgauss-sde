"""Posthoc named-native-model risk: actual passive movie -> frozen NN -> new force.
No training, parameter fitting or selection using response outcomes.
"""
from pathlib import Path
import sys,json,hashlib,datetime,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.gap_first_v01.drop_protocol import verify
from scripts.gap_first_v01.drop_run import geometry,config,ForceAdapter,render_utils,BASE,REPORT,sha
import torch,imageio.v2 as imageio
from internvit_predictor import VideoPhysicsPredictor_internvit300m_temporalattn
from release_utils import load_video_as_tensor
EXT=REPORT/'native-reference-extension';RUN=BASE/'native-reference-extension'
def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,allow_nan=False))
def movie(asset,world):
 source=BASE/'cases'/asset/('passive-'+world)/'trajectory.npz';rgb=np.load(source)['rgb'][:,0];path=RUN/asset/f'passive-{world}.mp4';path.parent.mkdir(parents=True,exist_ok=True)
 with imageio.get_writer(path,fps=10,codec='libx264rgb',pixelformat='rgb24',ffmpeg_params=['-crf','0']) as writer:
  for frame in rgb:writer.append_data(frame)
 decoded=np.array(imageio.mimread(path));assert np.array_equal(rgb,decoded),'Lossless RGB movie must preserve every pixel'
 return path
@torch.no_grad()
def simulate(asset,params,condition,deadline):
 g,x=geometry(asset);sim=ForceAdapter(config(),x,init_velocity=[0,0,0]).cuda();sim.set_all_particle(g._xyz);sim.set_physics_params(params.copy());v=sim.init_v.clone();xa=g._xyz.clone();rgb=[];centroids=[];controls=[]
 for frame in range(30):
  if time.time()>deadline:raise TimeoutError('Campaign deadline')
  if frame:
   sim.known_force=torch.tensor([0.,0.,20.],device='cuda').expand_as(x) if condition=='changed' and 12<=frame<=17 else None
   xa,x,v,bad=sim(xa,x,v,frame)
   if bool(bad) or not bool(torch.isfinite(xa).all()):raise RuntimeError('Nonfinite native replay')
  g._xyz=xa;im=render_utils.render_snapshot_with_grad(g,offset=(0,0),r=2,fov=40,resolution=512)['color'];rgb.append((im.permute(0,2,3,1).clamp(0,1).cpu().numpy()*255).astype(np.uint8));centroids.append(x.mean(0).cpu().numpy());controls.append(x.cpu().numpy())
 path=RUN/asset/(condition+'.npz');np.savez_compressed(path,rgb=np.array(rgb),centroids=np.array(centroids),controls=np.array(controls));return dict(rgb=np.array(rgb),centroids=np.array(centroids),path=path)
def main():
 protocol=verify();manifest=json.loads((EXT/'protocol.json').read_text());deadline=datetime.datetime.fromisoformat(manifest['deadline_utc']).timestamp();assert sha(__file__)==manifest['source_sha256'];
 for p,h in manifest['input_sha256'].items():assert sha(ROOT/p)==h,p
 if (EXT/'status.json').exists():raise RuntimeError('Extension status exists; no silent rerun')
 write(EXT/'status.json',dict(status='predicting',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
 torch.set_num_threads(4);torch.manual_seed(123);model=VideoPhysicsPredictor_internvit300m_temporalattn(internvit_model_name=str(BASE/'internvit')).cuda();model.load_state_dict(torch.load(BASE/'ckpt.pt',map_location='cpu',weights_only=False)['model'],strict=True);model.eval();predictions={}
 for asset in protocol['assets']:
  predictions[asset]={}
  for world in 'AB':
   path=movie(asset,world);video=load_video_as_tensor(str(path),num_frames=30,resize=(512,512))[None,:,:20].cuda()
   with torch.no_grad():params=model(video)
   predictions[asset][world]={k:float(v.cpu()) for k,v in params.items()}
  write(EXT/f'{asset}-frozen-prediction.json',dict(asset=asset,input='passive-A native camera0 frames0..19',native_params=predictions[asset]['A'],passive_B_sensitivity_params=predictions[asset]['B'],max_AB_parameter_difference=max(abs(predictions[asset]['A'][k]-predictions[asset]['B'][k]) for k in predictions[asset]['A']),input_movie_sha256={w:sha(RUN/asset/f'passive-{w}.mp4') for w in 'AB'},source_sha256=sha(__file__),manifest_sha256=sha(EXT/'protocol.json'),checkpoint_sha256=sha(BASE/'ckpt.pt'),prediction_fixed_before_force_simulation=True))
 del model;torch.cuda.empty_cache();results=[]
 for asset in protocol['assets']:
  params=predictions[asset]['A'];passive=simulate(asset,params,'passive',deadline);changed=simulate(asset,params,'changed',deadline);qhat=changed['centroids'][18:25].astype(float).reshape(-1)
  bank=json.loads((REPORT/f'{asset}-preobservation-bank.json').read_text());q=np.array(bank['target_bank']);errors=np.sqrt(np.mean((q-qhat)**2,axis=1));events=errors>.01;prior_risk=float(np.mean(events))
  # Freeze named-model risk from candidate predictions BEFORE reading truth-indexed noise/evaluation rows.
  write(EXT/f'{asset}-preoutcome-risk.json',dict(asset=asset,named_predictor='Unchanged ReconPhys checkpoint on passive-A movie, with frozen output parameters',qhat=qhat.tolist(),qhat_sha256=hashlib.sha256(qhat.tobytes()).hexdigest(),candidate_target_bank_sha256=bank['measurement_target_bank_sha256'],candidate_errors=errors.tolist(),prior_risk=prior_risk,threshold=.01,prediction_fixed_before_truth_loop=True))
  primary=json.loads((REPORT/f'{asset}-analysis.json').read_text());rows=[]
  for r in primary['noise_rows']:
   w=np.array([r['posterior_A'],1-r['posterior_A']]);risk=float(w@events);actual=float(errors[r['world']]);rows.append(dict(world=r['world'],replicate=r['replicate'],stage=r['stage'],native_predicted_failure_probability=risk,native_realized_coordinate_rms=actual,native_realized_failure=bool(events[r['world']]),native_brier=(risk-float(events[r['world']]))**2,corrected_MAP_error=r['point_refit_rms_error']))
  gt=np.load(BASE/'cases'/asset/'passive-A/trajectory.npz')['rgb'];fit=[]
  for label,indices in [('conditioning',slice(0,20)),('withheld_recorded',slice(20,30))]:
   diff=gt[indices,0].astype(float)-passive['rgb'][indices,0].astype(float);mse=float(np.mean(diff**2));fit.append(dict(split=label,view=0,mse=mse,psnr_db=None if mse==0 else float(10*np.log10(255**2/mse))))
  summary=[]
  for stage in ['before','after']:
   rr=[r for r in rows if r['stage']==stage];summary.append(dict(stage=stage,n=len(rr),mean_native_rms=float(np.mean([r['native_realized_coordinate_rms'] for r in rr])),mean_native_risk=float(np.mean([r['native_predicted_failure_probability'] for r in rr])),native_failure_rate=float(np.mean([r['native_realized_failure'] for r in rr])),native_brier=float(np.mean([r['native_brier'] for r in rr])),corrected_MAP_mean_error=float(np.mean([r['corrected_MAP_error'] for r in rr]))))
  d=dict(asset=asset,complete=True,params=params,fit=fit,candidate_errors=errors.tolist(),risk_summary=summary,rows=rows,passive_archive_sha256=sha(passive['path']),changed_archive_sha256=sha(changed['path']),limits=['Posthoc extension on predeclared primaryassets; no predictor training or hyperparameter tuning.','Actual native passive-A video supplied to unchanged predictor; passive-B sensitivity separately reported, never selected by forceoutcomes.','Native prediction remains fixed after correction; MAP reference is separately refit from identical calibrated evidence, not a retrained NN.','Known uniform bodyforce peranchor; normalizedsceneunits; native defaultcamera mayclipfloor.'])
  write(EXT/f'{asset}-result.json',d);results.append(d);print(asset,fit,summary,flush=True)
 write(EXT/'summary.json',dict(complete=True,assets=len(results),results=[{k:v for k,v in d.items() if k!='rows'} for d in results]));write(EXT/'status.json',dict(status='complete',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),gpu_released=True))
if __name__=='__main__':main()
