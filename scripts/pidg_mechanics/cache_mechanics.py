import os
"""Mechanics-only replay from fitted descriptors and training-derived controls."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from scripts.real_transfer_v19.neural_fem import NeuralFEM
from scripts.pidg_mechanics.fit_mechanics import payload
out=ROOT/os.environ.get('PIDG_MECHANICS_ROOT','runs/pidg-mechanics-v01');torch.set_num_threads(1);g=np.load(out/'training-geometry.npz');fit=json.loads((out/'mechanics-fit.json').read_text());assert json.loads((out/'mechanics-seal.json').read_text())['completed'];times=np.arange(150)/149;positions=[];diagnostics=[]
for b,row in enumerate(fit):
 data=payload(g,b,times);m=NeuralFEM(data,device='cuda:1',materials=1);th=m.t(row['parameters']);states=[];diags=[]
 for f in range(150):
  guess=m.t((data['observed'][f]-m.origin)/m.scale);x,H,diag=m.solve(th,f,guess);states.append(x.cpu().numpy());diags.append(diag)
 positions.append(states);diagnostics.append(diags);np.savez_compressed(out/f'body{b}-nodes.npz',positions=states,times=times,rest=m.rest.cpu().numpy(),elements=m.elements.cpu().numpy());print('Completed body',b,flush=True)
np.savez_compressed(out/'mechanical-states.npz',positions=np.stack(positions,axis=1),times=times)
(out/'replay.json').write_text(json.dumps({'completed':True,'method':'Existing NeuralFEM.solve; frozen material decoder + fitted attachment descriptors','uses_pidg_deformation_network_at_query':False,'controls':'PCHIP of fitting-frame inferred position targets; endpoints clamped outside first/last fitting time','ground_truth_images_read':False,'diagnostics':diagnostics},indent=2))
