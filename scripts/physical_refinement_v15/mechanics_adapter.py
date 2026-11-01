"""Bounded position-controlled Q2 adapter. Existing mechanics are unchanged.
Private state/parameters and reaction-wrench output are separate from geometry-
only sensor inputs. No inverse labels or forces enter the geometry payload.
"""
from contextlib import contextmanager
from pathlib import Path
import argparse,hashlib,json,signal,sys,time,threading,gc
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.position_ood.smooth_fixture import SmoothMaterialFixture
from scripts.position_ood.reference_solver import solve
from scripts.physical_refinement_v15.state_scene import boundary_grid,scene_from_state
from scripts.physical_refinement_v15.mechanics_qualification import wrench
from scripts.physical_refinement_v15.mechanics_pilot import SOURCES,write,sha

@contextmanager
def bounded(seconds):
 if not np.isfinite(seconds) or seconds<=0:raise ValueError('Explicit positive per-job budget required')
 if threading.current_thread() is not threading.main_thread():raise ValueError('Run mechanics jobs in isolated worker processes')
 old=signal.getsignal(signal.SIGALRM);timer=signal.getitimer(signal.ITIMER_REAL);started=time.monotonic();duration=min(seconds,timer[0]) if timer[0]>0 else seconds
 def timeout(signum,frame):raise TimeoutError('Explicit mechanics per-job runtime budget exhausted')
 signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,duration)
 try:yield
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
  if timer[0]>0 and timer[0]>time.monotonic()-started:signal.setitimer(signal.ITIMER_REAL,timer[0]-(time.monotonic()-started),timer[1])

def specification(case,poses,n,order):
 allowed={'E','Kleft','Kright','layout','material_gradient','density','gravity','field_amplitude_J','field_center','field_width'}
 if set(case)-allowed:raise ValueError('Unknown mechanics parameters')
 normalized=dict(material_gradient=0.,density=625.,gravity=[0.,0.,0.],field_amplitude_J=0.,field_center=[.12,0.,-.09],field_width=.08);normalized.update(case)
 for key in ('E','Kleft','Kright'):
  if key not in normalized or not np.isfinite(normalized[key]) or normalized[key]<=0:raise ValueError('Positive finite physical stiffness required')
 for key in ('material_gradient','density','field_amplitude_J','field_width'):
  if not np.isfinite(normalized[key]):raise ValueError('Finite constitutive/body parameters required')
 if normalized['density']<=0 or normalized['field_width']<=0:raise ValueError('Positive density and field width required')
 for key in ('gravity','field_center'):
  value=np.asarray(normalized[key],float)
  if value.shape!=(3,) or not np.isfinite(value).all():raise ValueError('Finite world vectors required')
  normalized[key]=value.tolist()
 if normalized['layout'] not in ('parallel_lower','crossed'):raise ValueError('Only the two geometry-matched v15 support layouts are supported')
 if n%2 or n<2 or order not in (3,4):raise ValueError('Even Q2 mesh and supported quadrature required')
 p=np.asarray(poses,float)
 if p.ndim!=3 or p.shape[1:]!=(2,6) or not len(p) or not np.isfinite(p).all():raise ValueError('Finite prescribed poses[T,2,6] required')
 sources=list(SOURCES)+['scripts/phystwin/gpu_multigrid.py','scripts/physical_refinement_v15/mechanics_adapter.py','scripts/physical_refinement_v15/mechanics_qualification.py','scripts/physical_refinement_v15/state_scene.py','scripts/physical_refinement_v15/fixture_geometry.py']
 return dict(case=normalized,poses_m_rad=p.tolist(),n=int(n),order=int(order),surface_cells=[24,8,8],reset_policy='Pose0 equilibrium computed once; every later action initialized independently from an exact copy of pose0 equilibrium, including resumed jobs',source_sha256={s:sha(ROOT/s) for s in sorted(set(sources))},semantics='Quasi-static Q2 Neo-Hookean with rest-frame smooth material ratio and physical Robin fixture compliance; no impact/contact or independent physical truth')

def run(case,poses,out,*,n=6,order=3,device='cuda:0',budget_seconds):
 if device not in ('cpu','cuda:0','cuda:1','cuda:2'):raise ValueError('Only authorized CPU/GPU0–2')
 spec=specification(case,poses,n,order);signature=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest();out=Path(out).resolve()
 allowed=(ROOT/'runs/physical-refinement-v15').resolve()
 if not out.is_relative_to(allowed) or out==allowed:raise ValueError('New isolated v15 run subdirectory required')
 out.mkdir(parents=True,exist_ok=True);manifest=out/'manifest.json'
 if manifest.exists():
  if json.loads(manifest.read_text())['signature']!=signature:raise ValueError('Refuse changed mechanics cache')
 else:write(manifest,dict(signature=signature,spec=spec,device=device,budget_seconds=budget_seconds,python=sys.version))
 if (out/'complete.json').exists():
  complete=json.loads((out/'complete.json').read_text())
  if complete['signature']!=signature or complete['poses']!=len(poses) or complete['surface_rest_sha256']!=sha(out/'surface-rest.npz'):raise ValueError('Changed completed mechanics job')
  for i in range(len(poses)):
   folder=out/f'pose-{i:03d}';row=json.loads((folder/'complete.json').read_text())
   if complete['pose_completion_sha256'][str(i)]!=sha(folder/'complete.json') or row['signature']!=signature or any(sha(folder/key)!=digest for key,digest in row['artifact_sha256'].items()):raise ValueError('Changed completed pose artifacts')
  return complete
 started=time.monotonic();model=None;previous=None;initial_equilibrium=None;rows=[]
 try:
  with bounded(budget_seconds):
   model=SmoothMaterialFixture(n,**spec['case'],order=order,device=device);surface,triangles=boundary_grid()
   if (out/'surface-rest.npz').exists():
    with np.load(out/'surface-rest.npz') as saved:
     np.testing.assert_array_equal(saved['vertices'],surface);np.testing.assert_array_equal(saved['triangles'],triangles)
   else:np.savez_compressed(out/'surface-rest.npz',vertices=surface,triangles=triangles)
   for i,pose in enumerate(np.asarray(poses,float)):
    folder=out/f'pose-{i:03d}';record=folder/'complete.json'
    if record.exists():
     row=json.loads(record.read_text())
     if row['signature']!=signature or any(sha(folder/key)!=digest for key,digest in row['artifact_sha256'].items()):raise ValueError('Changed pose cache')
     with np.load(folder/'private-state.npz') as saved:previous=saved['nodal_displacement'].copy()
     if i==0:initial_equilibrium=previous.copy()
     rows.append(row);continue
    model.set_controls(pose);state=solve(model,None if i==0 else initial_equilibrium.copy(),force_tolerance_N=1e-9);previous=state['displacement_local_m']
    if i==0:initial_equilibrium=previous.copy()
    sample=model.sampled(state);forces,torques=wrench(model,previous,pose)
    if not all(np.isfinite(np.asarray(value)).all() for value in [previous,state['J'],state['residual'],forces,torques,*sample.values()]):raise RuntimeError('Nonfinite mechanics state, sample, or wrench')
    if sample['residual_N']>1e-9 or sample['reaction_balance_N']>1e-7 or sample['minimum_J']<=.1:raise RuntimeError('Mechanics residual/J/balance qualification failed')
    folder.mkdir(parents=True,exist_ok=True);np.savez_compressed(folder/'private-state.npz',rest=model.rest,nodal_displacement=previous,**sample)
    np.savez_compressed(folder/'reaction-wrench.npz',forces_N=forces,torques_Nm=torques,origins_world_m=np.array([[0.,0,0],[.24,0,0]])+pose[:,:3])
    surface_positions=surface+model.query_field(previous,surface)
    if not np.isfinite(surface_positions).all():raise RuntimeError('Nonfinite fixed render surface')
    np.savez_compressed(folder/'surface.npz',vertices=surface_positions,triangles=triangles)
    # Renderer receives this file only: pure geometry/appearance, no case labels.
    meshes=scene_from_state(model,previous,spec['case']['layout'],pose);geometry={}
    for j,mesh in enumerate(meshes):
     for key in ('vertices','triangles','color'):geometry[f'mesh{j}_{key}']=np.asarray(mesh[key])
    if not all(np.isfinite(value).all() for value in geometry.values()):raise RuntimeError('Nonfinite geometry-only sensor payload')
    np.savez_compressed(folder/'sensor-geometry.npz',**geometry)
    row=dict(signature=signature,pose_index=i,pose_m_rad=pose.tolist(),residual_N=sample['residual_N'],minimum_J=sample['minimum_J'],reaction_balance_N=sample['reaction_balance_N'],artifact_sha256={key:sha(folder/key) for key in ('private-state.npz','reaction-wrench.npz','surface.npz','sensor-geometry.npz')});write(record,row);rows.append(row)
   if specification(case,poses,n,order)!=spec:raise ValueError('Mechanics/geometry sources changed during job')
   result=dict(state='complete',signature=signature,poses=len(rows),elapsed_seconds=time.monotonic()-started,pose_completion_sha256={str(i):sha(out/f'pose-{i:03d}/complete.json') for i in range(len(rows))},surface_rest_sha256=sha(out/'surface-rest.npz'));write(out/'complete.json',result);return result
 except BaseException as error:
  write(out/'status.json',dict(state='failed',signature=signature,completed_poses=len(rows),elapsed_seconds=time.monotonic()-started,error=repr(error)));raise
 finally:
  if model is not None:model.clear_linear_cache();model.engine=None;del model;gc.collect()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--n',type=int,default=6);p.add_argument('--order',type=int,default=3);p.add_argument('--device',default='cuda:0');p.add_argument('--budget-seconds',type=float,required=True);a=p.parse_args();s=json.loads(a.spec.read_text());run(s['case'],s['poses'],a.out,n=a.n,order=a.order,device=a.device,budget_seconds=a.budget_seconds)
