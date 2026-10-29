"""Privileged development forward check; never a primary observation interface."""
from pathlib import Path
import argparse,json,sys,time,gc
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.position_ood.study import BASE,OUT,protocol,within_deadline
from scripts.position_ood.runtime import dense_model
from scripts.phystwin.sequential_fixture import points,LAYOUTS
from scripts.phystwin.sequential_reference import write
p=argparse.ArgumentParser();p.add_argument('--object',type=int,default=1);p.add_argument('--meshes',default='2,3,4');a=p.parse_args();case=protocol()['development'][a.object];reference=np.load(BASE/f'fields/development-{a.object}.npz');rows=[]
for n in map(int,a.meshes.split(',')):
 within_deadline();m=dense_model(n);theta=m.tensor([*np.log([case['E'],case['Kleft'],case['Kright']]),*case['gravity'],case['field_amplitude_J'],case['material_gradient'],*case['field_center'],np.log(case['field_width'])]);ids=np.argmin(np.linalg.norm(m.geo.rest[:,None]-points()[None],axis=-1),axis=0);pred=[];start=time.time()
 with torch.no_grad():
  for action in [0,2,4,6,9]:
   within_deadline();u=m.predict(theta,m.tensor(reference['poses'][action]),LAYOUTS.index(case['layout']));pred.append(u[ids].cpu().numpy());print('ORACLE',n,action,flush=True)
 error=1000*np.sqrt(np.mean((np.array(pred)-reference['field'][[0,2,4,6,9]])**2,axis=(1,2)));rows.append(dict(mesh=n,actions=[0,2,4,6,9],coordinate_rmse_mm=error.tolist(),seconds=time.time()-start));write(OUT/f'development-{a.object}-forward-oracle.json',dict(scope='True physical parameters supplied for forward-model diagnosis only',rows=rows));print('ORACLE_RESULT',rows[-1],flush=True);del m;gc.collect();torch.cuda.empty_cache()
