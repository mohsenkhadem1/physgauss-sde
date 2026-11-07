"""Native ReconPhys replay with an explicitly isolated additive-force interface.
Original solver, renderer, interpolation, parameter transforms and floor are imported.
"""
from pathlib import Path
import argparse,datetime,hashlib,json,random,sys,time,pickle
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'runs/gap-first-v01/drop';VENDOR=BASE/'vendor/ReconPhys';REPORT=ROOT/'reports/gap-first-v01/drop';OUT=ROOT/'output/gap-first-v01/drop'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(VENDOR))
from scripts.gap_first_v01.drop_protocol import verify
import numpy as np
import torch
from yacs.config import CfgNode as CN
from trellis.representations.gaussian.gaussian_model import Gaussian
from trellis.utils import render_utils
from sms_lib.utils.transform import uniform_sampling
from sms_lib.models.spring_mass.Spring_Mass_simplify import Spring_Mass

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def config():
 c=CN(new_allowed=True);c.merge_from_file(str(VENDOR/'config/mpm_synthetic/default.yaml'));c.merge_from_file(str(VENDOR/'config/mpm_synthetic/multiscene.yaml'));d=c.DYNAMIC.clone();d.set_new_allowed(True);d.DATA=c.DATA;return d
class ForceAdapter(Spring_Mass):
    """Only intervention changes: add known per-anchor force to native mechanics."""
    def compute_force(self,xyz,v,K,damp):
        native=super().compute_force(xyz,v,K,damp)
        extra=getattr(self,'known_force',None)
        return native if extra is None else native+extra

def geometry(asset):
 random.seed(123);np.random.seed(123);torch.manual_seed(123);torch.cuda.manual_seed_all(123)
 g=Gaussian(sh_degree=0,aabb=[-.5,-.5,-.5,1,1,1],mininum_kernel_size=.0009,scaling_bias=.004,opacity_bias=.1,scaling_activation='softplus');g.load_ply(str(BASE/'assets'/asset/'gaussian.ply'))
 cache=BASE/'initial'/f'{asset}.pt'
 if cache.exists():
  saved=torch.load(cache,map_location='cuda',weights_only=False);assert saved['gaussian_sha256']==sha(BASE/'assets'/asset/'gaussian.ply');x=saved['controls']
 else:
  candidates=uniform_sampling(g._xyz,voxel_size=.01);x=candidates[random.sample(range(len(candidates)),2048)].clone();cache.parent.mkdir(parents=True,exist_ok=True);torch.save(dict(controls=x.cpu(),seed=123,gaussian_sha256=sha(BASE/'assets'/asset/'gaussian.ply')),cache)
 return g,x

def specification(case,asset):
 a=dict(m=1.,k=300.,damp=1.,fric_k=0.);b=dict(m=2.,k=600.,damp=2.,fric_k=0.)
 if case=='native-groundtruth':
  params=pickle.load(open(BASE/'assets'/asset/'sample_00000/physics_params.pkl','rb'));return {k:params[k] for k in ['m','k','damp','fric_k']},None
 if case=='native-predicted':return json.loads((REPORT/f'{asset}-native-predictor.json').read_text())['params'],None
 if case=='known-mass':return dict(m=1.,k=600.,damp=1.,fric_k=0.),None
 p=a if case.endswith('A') else b
 f=[4.,0.,0.] if case.startswith('calibration') else ([0.,0.,20.] if case.startswith('changed') else None)
 return p,f

@torch.no_grad()
def run(asset,case,pilot=False):
 folder=BASE/('pilot' if pilot else 'cases')/asset/case
 if (folder/'result.json').exists():
  prior=json.loads((folder/'result.json').read_text())
  if prior.get('complete'):return prior
  raise RuntimeError('Incomplete case retained; explicit new run namespace required')
 if folder.exists() and any(folder.iterdir()):raise RuntimeError('Partial case output retained; no silent overwrite')
 start=time.time();torch.set_num_threads(4);g,x=geometry(asset);init=x.clone();params,force=specification(case,asset);s=ForceAdapter(config(),x,init_velocity=[0,0,0]).cuda();s.set_all_particle(g._xyz);s.set_physics_params(params.copy());v=s.init_v.clone();xa=g._xyz.clone();frames=4 if pilot else 30
 folder=BASE/('pilot' if pilot else 'cases')/asset/case;folder.mkdir(parents=True,exist_ok=True);imgs=[];xs=[];vs=[];centroids=[];contacts=[];error=None
 for frame in range(frames):
  if frame:
   s.known_force=torch.tensor(force,device='cuda').expand_as(x) if force is not None and 12<=frame<=17 else None
   xa,x,v,bad=s(xa,x,v,frame)
   if bool(bad) or not bool(torch.isfinite(xa).all()):error=f'nonfinite frame{frame}';break
  g._xyz=xa
  rgb=render_utils.render_snapshot_with_grad(g,offset=(0,0),r=2,fov=40,resolution=512)['color'].permute(0,2,3,1).clamp(0,1)
  imgs.append((rgb.cpu().numpy()*255).astype(np.uint8));xs.append(x.cpu().numpy());vs.append(v.cpu().numpy());centroids.append(x.mean(0).cpu().numpy());contacts.append(int((x[:,2]<=s.ground+1e-6).sum()))
  np.save(folder/f'gaussians-{frame:03d}.npy',xa.cpu().numpy())
  print(asset,case,frame,round(time.time()-start,2),flush=True)
 np.savez_compressed(folder/'trajectory.npz',controls=np.array(xs),velocities=np.array(vs),centroids=np.array(centroids),contacts=np.array(contacts),initial_controls=init.cpu().numpy(),rgb=np.array(imgs))
 doc=dict(asset=asset,case=case,pilot=pilot,complete=error is None and len(imgs)==frames,error=error,frames=len(imgs),expected_frames=frames,params=params,force_per_anchor=force,force_frames=[12,17] if force else None,controls=2048,gaussians=len(xa),seconds=time.time()-start,dt=s.dt,substeps=s.n_step,ground=float(s.ground),contact_counts=contacts,trajectory=str((folder/'trajectory.npz').relative_to(ROOT)),initial_sha256=sha(BASE/'initial'/f'{asset}.pt'),source_sha256=sha(__file__),native_source_sha256=sha(VENDOR/'sms_lib/models/spring_mass/Spring_Mass_simplify.py'),gaussian_sha256=sha(BASE/'assets'/asset/'gaussian.ply'),units='Native normalized scene coordinates; mass and applied force per control anchor. No metric real-object calibration.',native_model='Spring_Mass_simplify; only force branch adds explicit known per-anchor force; native contact/interpolation/rendering unchanged.',recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 (folder/'result.json').write_text(json.dumps(doc,indent=2));print(json.dumps(doc),flush=True)
 return doc
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--asset',required=True);p.add_argument('--case',default='passive-A');p.add_argument('--pilot',action='store_true');a=p.parse_args()
 if not a.pilot:
  protocol=verify();assert a.asset in protocol['assets'] and a.case in protocol['cases'];assert protocol['source_sha256'][str(Path(__file__).relative_to(ROOT))]==sha(__file__)
 run(a.asset,a.case,a.pilot)
