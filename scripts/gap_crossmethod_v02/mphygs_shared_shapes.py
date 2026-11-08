"""Shared nonbox geometry, unmodified native M-PhyGs mechanics; supplied parameters."""
import sys,json,time,hashlib,datetime
from pathlib import Path
import numpy as np,torch,warp as wp
V=Path('runs/gap-crossmethod-v02/vendor/m-phygs/mphygs/warp_mpm');sys.path.insert(0,str(V.resolve()))
from mpm_solver_diff import MPMWARPDiff
from mpm_data_structure import MPMStateStruct,MPMModelStruct
R=Path('reports/gap-crossmethod-v02');U=Path('runs/gap-crossmethod-v02/mphygs-shared-shapes');U.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(shape,scale,action):
 name=f'{shape}-scale{scale}-{action}';path=U/(name+'.npz')
 if path.exists():raise RuntimeError('Refuse overwrite '+name)
 source=Path(f'runs/gap-crossmethod-v02/masiv/contact-v02/{shape}-scale1-gaussian0x.npz');x=np.load(source)['x'][0].astype(np.float32);n=len(x);d='cuda:0';vol=.018**3
 state=MPMStateStruct();state.init(n,device=d);state.from_torch(torch.tensor(x,device=d),torch.full((n,),vol,device=d),n_grid=32,requires_grad=False)
 model=MPMModelStruct();model.init(n,device=d);model.init_other_params(n_grid=32);solver=MPMWARPDiff(n,n_grid=32)
 solver.set_parameters_dict(model,state,dict(material='jelly',density=1000.*scale,g=[0,-9.8,0],grid_v_damping_scale=1.,rpic_damping=0.))
 solver.set_E_nu(model,50000.*scale,.3);solver.prepare_mu_lam(model,state)
 solver.add_surface_collider(point=[0,1/32,0],normal=[0,1,0],surface='sticky')
 center=np.array([.60,.16,.50],dtype=np.float32);half=np.array([.06,.20,.20],dtype=np.float32);selected=np.all(np.abs(x-center)<half,axis=1);count=int(selected.sum());assert count>0
 force={'passive':(0.,0.,0.),'calibration10x':(10.,0.,0.),'held20z':(0.,0.,20.)}[action]
 if action!='passive':
  solver.add_impulse_on_particles(state,(np.array(force)/count).tolist(),.0001,point=center.tolist(),size=half.tolist(),num_dt=600,start_time=.16)
  assert np.array_equal(solver.impulse_params[-1].mask.numpy().astype(bool),selected)
 frames=[x.copy()];times=[0.];jac=[1.];start=time.monotonic()
 for step in range(3000):
  solver.p2g2p(model,state,step,.0001)
  if (step+1)%100==0:
   a=state.particle_x.numpy();j=float(np.linalg.det(state.particle_F.numpy()).min());assert np.isfinite(a).all() and j>0,(step,j);frames.append(a);times.append((step+1)*.0001);jac.append(j)
 wp.synchronize();elapsed=time.monotonic()-start
 np.savez_compressed(path,x=np.array(frames),time=np.array(times),min_J=np.array(jac))
 meta=dict(shape=shape,scale=scale,action=action,particles=n,selected_particles=count,total_force_N=force,per_selected_particle_force_N=(np.array(force)/count).tolist(),force_window_s=[.16,.22],dt=.0001,steps=3000,E_Pa=50000.*scale,density_kg_m3=1000.*scale,particle_volume_m3=vol,mass_kg=n*vol*1000.*scale,min_saved_J=min(jac),seconds_local_wall=elapsed,geometry_sha256=sha(source),classification='Native M-PhyGs low-level jelly MPM supplied-parameter forward diagnostic; shared geometry with MASIV but different constitutive law and force spatial basis, not a method accuracy leaderboard')
 path.with_suffix('.json').write_text(json.dumps(meta,indent=2));print(name,elapsed,min(jac),flush=True)
def main():
 torch.set_num_threads(1);wp.init();manifest=R/'mphygs-shared-shapes-protocol.json'
 if manifest.exists():raise RuntimeError('Frozen protocol exists; new namespace required')
 manifest.write_text(json.dumps(dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),shapes=['torus','ellipsoid'],scales=[1,2],actions=['passive','calibration10x','held20z'],E=50000,density=1000,nu=.3,grid=32,dt=.0001,duration=.3,force_window=[.16,.22],force_basis='Native preselected material-particle box center(.60,.16,.50), halfextent(.06,.20,.20), totalforce divided by native selected count',scope='Native mechanics only; known parameter scale pairs, no fitted scene or image reconstruction; no matched constitutive response or Gaussian force profile across MASIV/MPhyGs',source_sha256={str(p):sha(p) for p in [Path(__file__),V/'mpm_solver_diff.py',V/'mpm_utils.py',V/'mpm_data_structure.py']}),indent=2))
 results=[]
 for shape in ['torus','ellipsoid']:
  for scale in [1,2]:
   for action in ['passive','calibration10x','held20z']:
    try:run(shape,scale,action);results.append(dict(shape=shape,scale=scale,action=action,status='complete'))
    except Exception as e:results.append(dict(shape=shape,scale=scale,action=action,status='failed',error=repr(e)));print(results[-1],flush=True)
    (R/'mphygs-shared-shapes-status.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__':main()
