"""Privileged development audit of mesh-error extrapolation."""
from pathlib import Path
import json,sys,time,gc
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.position_ood.study import BASE,OUT,protocol,within_deadline
from scripts.position_ood.pilot import CASES
from scripts.position_ood.differentiable import DensePositionFEM
from scripts.phystwin.sequential_fixture import points,LAYOUTS
from scripts.phystwin.sequential_reference import write
cases=[('development-1',protocol()['development'][1],BASE/'fields/development-1.npz')]
for i in [0,2]:
 q=json.loads((OUT/f'pilot-{i}-qualification.json').read_text());cases.append((f'pilot-{i}',CASES[i],BASE/f'pilot-{i}/n{q["mesh"]}-q3.npz'))
rows=[]
for name,case,path in cases:
 data=np.load(path);pred={};actions=[0,2,4,6,9]
 for n in [1,2,3]:
  within_deadline();m=DensePositionFEM(n);theta=m.tensor([*np.log([case['E'],case['Kleft'],case['Kright']]),*case['gravity'],case['field_amplitude_J'],case.get('material_gradient',0.),*case.get('field_center',[.12,0,-.09]),np.log(case.get('field_width',.08))]);ids=np.argmin(np.linalg.norm(m.geo.rest[:,None]-points()[None],axis=-1),axis=0);values=[]
  with torch.no_grad():
   for action in actions:values.append(m.predict(theta,m.tensor(data['poses'][action]),LAYOUTS.index(case['layout']))[ids].cpu().numpy())
  pred[n]=np.array(values);del m;gc.collect();torch.cuda.empty_cache()
 target=data['field'][actions];variants={f'n{n}':p for n,p in pred.items()};variants['extrap12']=(4*pred[2]-pred[1])/3;variants['extrap23']=(9*pred[3]-4*pred[2])/5
 errors={k:(1000*np.sqrt(np.mean((p-target)**2,axis=(1,2)))).tolist() for k,p in variants.items()};rows.append(dict(case=name,actions=actions,errors_mm=errors));write(OUT/'extrapolation-development-audit.json',dict(scope='Development-only privileged forward diagnostic; exponent 2 checked against independent refined references, not assumed certified.',rows=rows));np.savez_compressed(BASE/f'{name}-extrapolation-audit.npz',**variants);print(name,errors,flush=True)
