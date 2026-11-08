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

# Frozen conditional two-world mechanics diagnostic; not full M-PhyGs inference.
import datetime
protocol=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Unknown density and modulus two-world native solver reference; native full pipeline normally supplies density. Not a rendering or fitted identification benchmark.',worlds=[dict(E=2000,rho=1000),dict(E=4000,rho=2000)],calibration='Previously computed reset+x0.01per-particle force at.08-.10s; final saved x-centroid projected metric coordinate',held='Separate reset+y0.015per-particle force at.08-.10s; full 3DpointEuclidean RMS across savedframes40..79',noise_sigma=.00002,tolerance=.0001,prior=[.5,.5],noise_repeats_per_world=100,seed=771209,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
p=R/'mphygs-second-interaction-protocol.json'
if p.exists():raise RuntimeError('No overwrite')
p.write_text(json.dumps(protocol,indent=2))
cal=np.load('runs/gap-crossmethod-v02/mphygs-native-contact/trajectories.npz');mu=np.array([cal['scale1-forceTrue'][-1,:,0].mean(),cal['scale2-forceTrue'][-1,:,0].mean()]);targets=[]
for scale in (1.,2.):
 a,t=run(scale,True,(0,.015,0));targets.append(a[40:].astype(float));print('held',scale,t,flush=True)
q=np.stack(targets);np.savez_compressed(U/'second-interaction-target-bank.npz',targets=q,calibration_means=mu)
err=np.sqrt(np.mean(np.sum((q-q[0])**2,axis=-1),axis=(1,2)));priorrisk=float(np.mean(err>.0001))
(R/'mphygs-second-interaction-prediction-freeze.json').write_text(json.dumps(dict(means=mu.tolist(),candidate_pair_errors=err.tolist(),prior_map_index=0,prior_risk=priorrisk,target_bank_sha256=hashlib.sha256((U/'second-interaction-target-bank.npz').read_bytes()).hexdigest(),created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2))
from scipy.special import softmax
rng=np.random.default_rng(771209);rows=[]
for world in range(2):
 for rep in range(100):
  y=float(mu[world]+rng.normal(scale=.00002));post=softmax(-.5*((y-mu)/.00002)**2)
  for stage,w in [('before',np.array([.5,.5])),('after',post)]:
   j=int(w.argmax());errors=np.sqrt(np.mean(np.sum((q-q[j])**2,axis=-1),axis=(1,2)));risk=float(w@(errors>.0001));actual=float(errors[world]);rows.append(dict(world=world,replicate=rep,stage=stage,observation=y,weights=w.tolist(),map_index=j,risk=risk,error=actual,failed=actual>.0001,same_evidence_MAP_error=actual,brier=(risk-float(actual>.0001))**2))
summary={}
for stage in ['before','after']:
 rr=[r for r in rows if r['stage']==stage];summary[stage]={k:float(np.mean([r[k] for r in rr])) for k in ['risk','error','failed','brier']}
(R/'mphygs-second-interaction.json').write_text(json.dumps(dict(scope=protocol['scope'],summary=summary,rows=rows,calibration_means=mu.tolist(),held_pair_point_rms=float(err[1]),limitations=['Finite two-world reference includes truth by construction.','Supplied synthetic particle geometry and calibrated metric projection, not RGB inference.','Same-evidence MAP has identical point performance; no unique accuracy win.','Unknown density is an extension beyond known-density native fit assumptions.','No native M-PhyGs trained checkpoint or material-field inference tested.']),indent=2));print(summary,flush=True)
