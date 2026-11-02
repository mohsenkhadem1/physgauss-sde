"""Two-world DEVELOPMENT 15-continuous-parameter conditional inference pilot."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,time,signal,subprocess,argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.compositional_neural_mechanics.gravity_inference_pilot import closure
from scripts.compositional_neural_mechanics.conditional_inference_pilot import POINTS,BULK,INTERFACE,sha,write,predict
from scripts.compositional_neural_mechanics.joint_inference import JointMAP,neutral_parameters,NAMES
from scripts.compositional_neural_mechanics.scenes import sample_scene,split_cells,poses
from scripts.compositional_neural_mechanics.mechanics import GeometrySpec
from scripts.compositional_neural_mechanics.scale_separated_material import ScaleSeparatedMaterialLaw
from scripts.compositional_neural_mechanics.interfaces import SharedInterfaceLaw
R=ROOT/'reports/compositional-neural-mechanics-v16';OUT=ROOT/'runs/compositional-neural-mechanics-v16/full-continuous-inference-pilot';PLAN=R/'full-continuous-inference-pilot-plan.json'

def pack(scene):
 return np.array([scene['bulk_latent'][0],scene['bulk_latent'][1],scene['gradient'],scene['interface_latents'][0][0],scene['interface_latents'][0][1],scene['interface_latents'][1][0],scene['interface_latents'][1][1],*np.array(scene['centers']).reshape(-1),*np.log(np.array(scene['widths'])).reshape(-1)])
def build(geometry,categories,observations,learned):
 bulk=interface=None
 if learned:
  bulk=ScaleSeparatedMaterialLaw().double();bulk.load_state_dict(torch.load(BULK,weights_only=False)['state_dict'])
  interface=SharedInterfaceLaw();interface.load_state_dict(torch.load(INTERFACE,weights_only=False)['state_dict'])
 return JointMAP(geometry,bulk_family=categories['bulk_family'],interface_family=categories['interface_family'],heterogeneity=categories['heterogeneity'],observations=observations,position_sigma=50e-6,n=2,bulk_law=bulk,interface_law=interface,device='cpu',gravity=(0.,0.,-9.81),density=125.)
def verify(plan):
 if any(sha(ROOT/p)!=h for p,h in plan['source_sha256'].items()):raise ValueError('Frozen pilot dependency mismatch')
def worker(index):
 torch.set_num_threads(1);plan=json.loads(PLAN.read_text());verify(plan)
 if time.time()>=plan['deadline_unix']:raise TimeoutError('Pilot deadline')
 rng=np.random.default_rng(plan['scene_seed']);scenes=[sample_scene(split_cells()['train'][j],rng,f'full-continuous-dev-{i}',gravity=(0.,0.,-9.81))for i,j in enumerate(plan['train_cell_indices'])]
 scene=scenes[index];geometry=GeometrySpec(**scene['geometry']);truth=pack(scene)
 categories=dict(bulk_family=scene['bulk_family'],interface_family=int(scene['interface_latents'][0][2]),heterogeneity=scene['factors']['heterogeneity'])
 commands=poses(geometry);observed=[commands[k]for k in plan['observed_actions']]
 held=[];p=np.zeros((2,6));p[1,4]=.30;held.append(p)
 p=np.zeros((2,6));p[0,0]=-.07*geometry.dimensions[0];p[1,0]=.07*geometry.dimensions[0];p[0,3]=-.3;p[1,3]=.3;held.append(p)
 dummy=[dict(pose=p,points=POINTS,displacement=np.zeros_like(POINTS))for p in observed]
 teacher=build(geometry,categories,dummy,False);start=time.monotonic();path=R/f'full-continuous-inference-pilot-world{index}.json';local=OUT/f'world{index}';local.mkdir(parents=True)
 def timeout(*args):raise TimeoutError('External full-continuous pilot/fit timeout')
 signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,max(.01,plan['deadline_unix']-time.time()))
 report=dict(plan_sha256=sha(PLAN),world=index,scene=scene,categories_conditioned=categories,force_observation_count=0,fits=[])
 try:
  clean=[predict(teacher,truth,p)for p in observed];future=[predict(teacher,truth,p)for p in held]
 except Exception as e:
  report.update(reference_failure=f'{type(e).__name__}: {e}',incomplete=True);write(path,report);return
 # Same declared RNG stream split into nonoverlapping scene blocks; no outcome-dependent noise.
 noise_rng=np.random.default_rng(plan['noise_seed']);noise=noise_rng.normal(size=(2,4,len(POINTS),3))[index]*50e-6
 obs=[dict(pose=p,points=POINTS,displacement=c['displacement']+noise[j])for j,(p,c)in enumerate(zip(observed,clean))]
 assert all(set(o)=={'pose','points','displacement'} for o in obs)
 np.savez_compressed(local/'observations-and-development-truth.npz',truth=truth,points=POINTS,observed_poses=observed,held_poses=held,**{f'obs_{j}':o['displacement']for j,o in enumerate(obs)},**{f'future_{j}_{k}':v for j,c in enumerate(future)for k,v in c.items()})
 report['reference_seconds']=time.monotonic()-start;write(path,report)
 for learned in (False,True):
  estimator=build(geometry,categories,obs,learned);row=dict(model='learned'if learned else 'analytic',active=list(range(15)),initial=neutral_parameters().tolist(),force_observation_count=0)
  remaining=plan['deadline_unix']-time.time()
  if remaining<=1:row.update(incomplete=True,error='Whole pilot deadline');report['fits'].append(row);continue
  signal.setitimer(signal.ITIMER_REAL,min(plan['fit_seconds'],remaining))
  try:
   result=estimator.fit(active=list(range(15)),max_iterations=80,max_evaluations=160,seconds=min(899,remaining));row.update({k:v.tolist()if isinstance(v,np.ndarray)else v for k,v in result.items()})
  except Exception as e:row.update(incomplete=True,optimizer_success=False,error=f'{type(e).__name__}: {e}')
  signal.setitimer(signal.ITIMER_REAL,max(.01,plan['deadline_unix']-time.time()))
  if 'parameters'in row:
   row['packed_latent_error']=(np.asarray(row['parameters'])-truth).tolist();row['heldout_errors']=[]
   for j,p in enumerate(held):
    try:
     pred=predict(estimator,row['parameters'],p);row['heldout_errors'].append({k:float(np.sqrt(np.mean((pred[k]-future[j][k])**2)))for k in future[j]});np.savez_compressed(local/f"forecast-{row['model']}-{j}.npz",**pred)
    except Exception as e:row['heldout_errors'].append({'failure':f'{type(e).__name__}: {e}'})
  report['fits'].append(row);write(path,report);print(index,row['model'],row.get('optimizer_success'),row.get('loss'),row.get('heldout_errors'),flush=True)
 report.update(elapsed_seconds=time.monotonic()-start,source_hashes_unchanged=all(sha(ROOT/p)==h for p,h in plan['source_sha256'].items()));write(path,report)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--worker',type=int,choices=[0,1]);args=parser.parse_args()
 if args.worker is not None:return worker(args.worker)
 if OUT.exists() or PLAN.exists():raise ValueError('Preserve previous pilot')
 files=closure([Path(__file__)])+[BULK,INTERFACE]
 plan=dict(scope='DEVELOPMENT all15continuous unknown; known geometry,categorical bulk/interface families,heterogeneity pattern,density/gravity; no reserved cases',scene_seed=926811,noise_seed=926813,train_cell_indices=[2,13],active=list(NAMES),observed_actions=['initial','stretch','right_bend','opposed_twist'],held_actions=['right_bend_y_.30','stretch_.07L_and_opposed_twist_.30'],point_count=26,noise_position_m=50e-6,force_observation_count=0,gravity=[0.,0.,-9.81],density=125.,n=2,prior='Neutral parameters, independent unit Gaussian on15packed descriptors; defaultbounds; no truth starts',max_iterations=80,max_evaluations=160,fit_seconds=900,whole_seconds=3600,workers=2,threads=1,device='cpu',created_unix=time.time(),deadline_unix=time.time()+3600,source_sha256={str(p.relative_to(ROOT)):sha(p)for p in files})
 write(PLAN,plan);OUT.mkdir(parents=True);processes=[]
 for i in range(2):
  log=(OUT/f'world{i}.log').open('w');processes.append((subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker',str(i)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True),log))
 try:
  for process,_ in processes:process.wait(timeout=max(.01,plan['deadline_unix']-time.time()))
 except subprocess.TimeoutExpired:
  for process,_ in processes:
   if process.poll() is None:os.killpg(process.pid,signal.SIGTERM)
 finally:
  for process,log in processes:
   if process.poll() is None:
    try:process.wait(timeout=5)
    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
   log.close()
 write(R/'full-continuous-inference-pilot-status.json',dict(plan_sha256=sha(PLAN),exit_codes=[p.returncode for p,_ in processes],finished_unix=time.time(),world_reports=[str(R/f'full-continuous-inference-pilot-world{i}.json')for i in range(2)]))
if __name__=='__main__':main()
