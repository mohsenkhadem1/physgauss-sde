"""Native M-PhyGs low-level forward diagnostic, not fitted/rendered pipeline."""
import sys,json,time,hashlib
from pathlib import Path
import numpy as np,torch,warp as wp
V=Path('runs/gap-crossmethod-v02/vendor/m-phygs/mphygs/warp_mpm');sys.path.insert(0,str(V.resolve()))
from mpm_solver_diff import MPMWARPDiff
from mpm_data_structure import MPMStateStruct,MPMModelStruct
R=Path('reports/gap-crossmethod-v02');U=Path('runs/gap-crossmethod-v02/mphygs-native-contact');U.mkdir(exist_ok=True)
wp.init();torch.set_num_threads(1)
def run(scale,forced=False):
 d='cuda:0';N=8;axis=np.linspace(.42,.58,N,dtype=np.float32);x=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),-1).reshape(-1,3);n=len(x)
 state=MPMStateStruct();state.init(n,device=d,requires_grad=False);state.from_torch(torch.tensor(x,device=d),torch.full((n,),(.16/N)**3,device=d),n_grid=32,requires_grad=False)
 model=MPMModelStruct();model.init(n,device=d);model.init_other_params(n_grid=32);solver=MPMWARPDiff(n,n_grid=32)
 solver.set_parameters_dict(model,state,dict(material='jelly',density=1000.*scale,g=[0,0,-9.81],grid_v_damping_scale=1.,rpic_damping=0.))
 solver.set_E_nu(model,2000.*scale,.3);solver.prepare_mu_lam(model,state)
 solver.add_surface_collider(point=[0,0,.4],normal=[0,0,1],surface='sticky')
 dt=.0002
 if forced:solver.add_impulse_on_particles(state,[0.01,0,0],dt,point=[.5,.5,.5],size=[.2,.2,.2],num_dt=100,start_time=.08)
 frames=[];start=time.monotonic()
 for step in range(800):
  solver.p2g2p(model,state,step,dt)
  if step%10==0:frames.append(state.particle_x.numpy())
 wp.synchronize();a=np.array(frames);assert np.isfinite(a).all();return a,time.monotonic()-start
out={};arrays={}
for scale in [1.,2.]:
 for forced in [False,True]:
  key=f'scale{scale:g}-force{forced}';a,t=run(scale,forced);arrays[key]=a;out[key]={'finite':True,'seconds_local_wall':t,'frames':len(a),'particles':a.shape[1]};print(key,out[key],flush=True)
a,b=arrays['scale1-forceFalse'],arrays['scale2-forceFalse'];c,d=arrays['scale1-forceTrue'],arrays['scale2-forceTrue'];out['passive_max_abs']=float(np.max(abs(a-b)));out['forced_coordinate_rms_gap']=float(np.sqrt(np.mean((c-d)**2)));out['passive_centered_shape_change_rms']=float(np.sqrt(np.mean(((a[-1]-a[-1].mean(0))-(a[0]-a[0].mean(0)))**2)));out['scope']='Unmodified native low-level Warp MPM with supplied synthetic particles/materials; no material checkpoint, appearance reconstruction or rendering. E and density common scaling; known per-particle force breaks symmetry.';out['versions']={'torch':torch.__version__,'warp':wp.__version__};out['source_hashes']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [V/'mpm_solver_diff.py',V/'mpm_utils.py',V/'mpm_data_structure.py',Path(__file__)]};np.savez_compressed(U/'trajectories.npz',**arrays);(R/'mphygs-native-contact.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
