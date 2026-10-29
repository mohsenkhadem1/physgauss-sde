"""Development oracle selection of forward fidelity for smooth attachment physics."""
from pathlib import Path
import json,sys,time
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.position_ood.study import ROOT,within_deadline
from scripts.position_ood.smooth_fixture import SmoothDensePositionFEM
from scripts.phystwin.sequential_fixture import points
from scripts.phystwin.sequential_reference import write
b=ROOT/'runs/position-ood-v09';case=json.loads((b/'cohorts.json').read_text())['development'][3];ref=np.load(b/'smooth-reference/fields/development-3.npz');actions=[0,4,7];predictions={};rows=[]
for n in [2,3,4]:
 within_deadline();m=SmoothDensePositionFEM(n);theta=m.tensor([*np.log([case['E'],case['Kleft'],case['Kright']]),*case['gravity'],case['field_amplitude_J'],case['material_gradient'],*case['field_center'],np.log(case['field_width'])]);ids=np.argmin(np.linalg.norm(m.geo.rest[:,None]-points()[None],axis=-1),axis=0);start=time.time()
 with torch.no_grad():u=np.array([m.predict(theta,m.tensor(ref['poses'][a]),0)[ids].cpu().numpy() for a in actions])
 predictions[str(n)]=u;rows.append(dict(method=f'dense_n{n}',coordinate_rmse_mm=(1000*np.sqrt(np.mean((u-ref['field'][actions])**2,axis=(1,2)))).tolist(),seconds=time.time()-start));del m;torch.cuda.empty_cache()
for power in [2,4]:
 u=(3**power*predictions['3']-2**power*predictions['2'])/(3**power-2**power);predictions[f'extrap{power}']=u;rows.append(dict(method=f'extrapolated_power{power}',coordinate_rmse_mm=(1000*np.sqrt(np.mean((u-ref['field'][actions])**2,axis=(1,2)))).tolist()))
write(ROOT/'reports/position-ood-v10/forward-development-audit.json',dict(scope='True-parameter development oracle, smooth attachment, qualified n8 reference; no held-out inverse results.',object=3,actions=actions,rows=rows));np.savez_compressed(ROOT/'runs/position-ood-v10/forward-development-audit.npz',**predictions);print(rows,flush=True)
