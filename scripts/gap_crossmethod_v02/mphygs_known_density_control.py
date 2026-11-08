"""Native M-PhyGs low-level forward diagnostic, not fitted/rendered pipeline."""
import sys,json,time,hashlib
from pathlib import Path
import numpy as np,torch,warp as wp
V=Path('runs/gap-crossmethod-v02/vendor/m-phygs/mphygs/warp_mpm');sys.path.insert(0,str(V.resolve()))
from mpm_solver_diff import MPMWARPDiff
from mpm_data_structure import MPMStateStruct,MPMModelStruct
R=Path('reports/gap-crossmethod-v02');U=Path('runs/gap-crossmethod-v02/mphygs-native-contact');U.mkdir(exist_ok=True)
wp.init();torch.set_num_threads(1)
def run(scale,forced=False,forcevector=(.01,0,0),density_override=None):
 d='cuda:0';N=8;axis=np.linspace(.42,.58,N,dtype=np.float32);x=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),-1).reshape(-1,3);n=len(x)
 state=MPMStateStruct();state.init(n,device=d,requires_grad=False);state.from_torch(torch.tensor(x,device=d),torch.full((n,),(.16/N)**3,device=d),n_grid=32,requires_grad=False)
 model=MPMModelStruct();model.init(n,device=d);model.init_other_params(n_grid=32);solver=MPMWARPDiff(n,n_grid=32)
 solver.set_parameters_dict(model,state,dict(material='jelly',density=1000.*scale if density_override is None else density_override,g=[0,0,-9.81],grid_v_damping_scale=1.,rpic_damping=0.))
 solver.set_E_nu(model,2000.*scale,.3);solver.prepare_mu_lam(model,state)
 solver.add_surface_collider(point=[0,0,.4],normal=[0,0,1],surface='sticky')
 dt=.0002
 if forced:solver.add_impulse_on_particles(state,list(forcevector),dt,point=[.5,.5,.5],size=[.2,.2,.2],num_dt=100,start_time=.08)
 frames=[];start=time.monotonic()
 for step in range(800):
  solver.p2g2p(model,state,step,dt)
  if step%10==0:frames.append(state.particle_x.numpy())
 wp.synchronize();a=np.array(frames);assert np.isfinite(a).all();return a,time.monotonic()-start


base=np.load(U/'trajectories.npz')['scale1-forceFalse'];a,t=run(2.,False,density_override=1000.);np.save(U/'known-density-control.npy',a)
(R/'mphygs-known-density-control.json').write_text(json.dumps(dict(scope='Native supplied-parameter positive control: E doubles while density is known/fixed1000; compare with original E2000rho1000gravitycontact.',coordinate_rms_difference=float(np.sqrt(np.mean((a-base)**2))),max_abs_difference=float(np.max(abs(a-base))),seconds_local_wall=t,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
