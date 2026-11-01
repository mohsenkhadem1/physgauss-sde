"""Development-only mechanics feasibility; unchanged existing Q2 solver.
Position commands are inputs. Integrated fixture reactions are outputs, not
inputs to the motion-only identification track. No reserved cases/actions.
"""
from pathlib import Path
import argparse,hashlib,json,sys,time,gc
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.position_ood.smooth_fixture import SmoothMaterialFixture
from scripts.position_ood.fixture import controls
from scripts.position_ood.reference_solver import solve
from scripts.phystwin.sequential_fixture import points
BASE=ROOT/'runs/physical-refinement-v15';REPORT=ROOT/'reports/physical-refinement-v15'
CASES={'homogeneous':dict(E=12000.,Kleft=6000.,Kright=6000.,layout='parallel_lower',material_gradient=0.),'gradient_plus':dict(E=12000.,Kleft=6000.,Kright=6000.,layout='parallel_lower',material_gradient=.8),'gradient_minus':dict(E=12000.,Kleft=6000.,Kright=6000.,layout='parallel_lower',material_gradient=-.8),'crossed_support':dict(E=12000.,Kleft=6000.,Kright=6000.,layout='crossed',material_gradient=0.),'scaled_identifiability_control':dict(E=24000.,Kleft=12000.,Kright=12000.,layout='parallel_lower',material_gradient=0.)}
ACTIONS=[('stretch',.001),('translate_y',.006),('twist_x',.006)]
SOURCES=['scripts/position_ood/smooth_fixture.py','scripts/position_ood/material_field.py','scripts/position_ood/fixture.py','scripts/position_ood/reference_solver.py','scripts/phystwin/tensor_fixture.py','scripts/phystwin/consistent_fixture.py','scripts/phystwin/sequential_fixture.py','scripts/phystwin/v08_reference.py','scripts/physical_refinement_v15/mechanics_pilot.py']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
def shape(x,u):
 y=x+u;xc=x-x.mean(0);yc=y-y.mean(0);U,S,V=np.linalg.svd(xc.T@yc);d=np.eye(3);d[-1,-1]=np.linalg.det(U@V);R=U@d@V;res=yc-xc@R
 total=float(np.sqrt(np.mean(u*u)));nonrigid=float(np.sqrt(np.mean(res*res)))
 return dict(coordinate_rms_mm=total*1000,nonrigid_coordinate_rms_mm=nonrigid*1000,rms_ratio=nonrigid/max(total,1e-30))
def run(name,n,order=3,device='cuda:0'):
 spec=dict(case=CASES[name],n=n,order=order,gravity=[0,0,0],actions=ACTIONS,source_sha256={p:sha(ROOT/p) for p in SOURCES});signature=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest();out=BASE/'mechanics-pilot'/name/f'n{n}-q{order}';out.mkdir(parents=True,exist_ok=True)
 if (out/'complete.json').exists():
  old=json.loads((out/'complete.json').read_text());assert old['signature']==signature;return old
 write(out/'plan.json',dict(spec=spec,signature=signature,scope='Development only; no test generation or uncertainty claims'))
 model=SmoothMaterialFixture(n,**CASES[name],gravity=(0,0,0),field_amplitude_J=0.,order=order,device=device);start=time.monotonic();previous=None;rows=[]
 try:
  for j,(action,amp) in enumerate(ACTIONS):
   model.set_controls(controls(action,amp));state=solve(model,previous,force_tolerance_N=1e-9);previous=state['displacement_local_m'];sample=model.sampled(state);path=out/f'action{j}.npz';np.savez_compressed(path,rest=model.rest,nodal_displacement=previous,query_points=points(),**sample)
   row=dict(action=action,amplitude=amp,residual_N=sample['residual_N'],reaction_balance_N=sample['reaction_balance_N'],minimum_J=sample['minimum_J'],reaction_forces_N=sample['reaction_forces_N'].tolist(),elastic_J=sample['elastic_J'],**shape(points(),sample['field']));rows.append(row);print(name,n,order,json.dumps(row),flush=True)
  result=dict(signature=signature,spec=spec,rows=rows,elapsed_seconds=time.monotonic()-start,node_count=len(model.rest),element_count=len(model.elements),artifact_sha256={f'action{j}.npz':sha(out/f'action{j}.npz') for j in range(len(ACTIONS))});write(out/'complete.json',result);return result
 finally:
  model.clear_linear_cache();model.engine=None;del model;gc.collect()
  import torch
  if device.startswith('cuda'):torch.cuda.empty_cache()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--name',choices=list(CASES),required=True);p.add_argument('--n',type=int,required=True);p.add_argument('--order',type=int,default=3);p.add_argument('--device',default='cuda:0');a=p.parse_args();run(a.name,a.n,a.order,a.device)
