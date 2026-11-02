import sys,os,json,hashlib,time
from pathlib import Path
export=Path(__file__).resolve().parent
original=Path('/home/bayespc2/vscode/guassianSDE').resolve();environment=original/'runs/phystwin-env'
sys.path=[str(export)]+[p for p in sys.path if p and (not Path(p).resolve().is_relative_to(original)or Path(p).resolve().is_relative_to(environment))]
reads=[];blocked=[]
def audit(event,args):
 if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
 path=Path(os.fsdecode(args[0])).resolve()
 if path.is_relative_to(original)and not path.is_relative_to(environment):
  blocked.append(str(path));raise PermissionError('Original workspace access forbidden in isolated source smoke')
 if path.is_relative_to(export)and path.suffix.lower()in ('.pt','.pth','.npz','.h5','.hdf5','.usd'):
  blocked.append(str(path));raise PermissionError('Dataset/checkpoint access forbidden')
 reads.append(str(path))
sys.addaudithook(audit)
import numpy as np
import torch
from scripts.compositional_neural_mechanics.scenes import sample_scene,make_model
from scripts.compositional_neural_mechanics.mechanics import GeometrySpec
torch.set_num_threads(1);started=time.monotonic()
scene=sample_scene((1,1,1,1),np.random.default_rng(926999),'source-only-analytic-smoke',gravity=(0,0,-9.81))
for z in scene['interface_latents']:z[2]=1.
model=make_model(scene,n=2,order=3,device='cpu');pose=np.zeros((2,6));pose[0,0]=-.04*model.geometry.dimensions[0];pose[1,0]=.04*model.geometry.dimensions[0];pose[1,5]=.25
u=None
for f in (.25,.5,.75,1):
 model.set_controls(pose*f);state=model.solve(initial=u,force_tolerance_N=1e-9,maximum_iterations=60);u=state['displacement_local_m']
w=model.reaction_wrenches(state);residual=float(np.linalg.norm(state['residual']));minimum_J=float(np.min(state['J']));balance=float(w['force_balance_N']);x=model.rest;y=x+u;a=x-x.mean(0);b=y-y.mean(0);U,S,V=np.linalg.svd(a.T@b);D=np.eye(3);D[-1,-1]=np.linalg.det(U@V);rot=U@D@V;nonrigid=float(np.sqrt(np.mean((a@rot-b)**2)))
assert np.isfinite(u).all()and np.isfinite(w['forces_N']).all()and np.isfinite(w['torques_Nm']).all()
assert residual<1e-8 and minimum_J>.05 and balance<1e-8 and nonrigid>1e-5
imports={}
for name,module in list(sys.modules.items()):
 if name.startswith(('scripts.','experiments.','src.'))and getattr(module,'__file__',None):
  p=Path(module.__file__).resolve();assert p.is_relative_to(export),(name,str(p));imports[name]=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
assert imports and not blocked
print(json.dumps(dict(passed=True,scope='Exported committed source only, existing Python environment; no learned checkpoint/data or full training replication.',cwd=str(Path.cwd()),export=str(export),sys_path=sys.path,imported_project_modules=imports,blocked_accesses=blocked,original_workspace_reads_outside_runtime=[p for p in reads if Path(p).is_relative_to(original)and not Path(p).is_relative_to(environment)],n=2,order=3,bulk_family=scene['bulk_family'],interface_family=1,gravity=[0,0,-9.81],pose=pose.tolist(),nodes=len(model.rest),elements=len(model.elements),residual_N=residual,minimum_J=minimum_J,force_balance_N=balance,nonrigid_coordinate_rms_m=nonrigid,seconds=time.monotonic()-started,runtime=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__))),flush=True)
