import sys,json
from pathlib import Path
import numpy as np
root=Path('/home/bayespc2/vscode/guassianSDE');base=root/'runs/coupled-contact-v13/clipped-mesh-qualification/clipped-mesh-7822c76f99d9'
out=[]
for E,mu in [(6000,.35),(3000,.1)]:
 for n in [6,8]:
  p=base/f'edge_drop-E{E}-mu{mu}-n{n}-dt2.5e-05-spacing0.0015/trajectory.npz'
  with np.load(p) as d:
   x=d['sampled'];t=d['times'];ref=x[0]-x[0].mean(0);angles=[]
   for f in x:
    U,_,V=np.linalg.svd(ref.T@(f-f.mean(0)));R=U@V
    angles.append(float(np.rad2deg(np.arccos(np.clip((np.trace(R)-1)/2,-1,1)))))
   edge=np.linalg.norm(d['reactions'][:,1],axis=1)>1e-4;floor=np.linalg.norm(d['reactions'][:,0],axis=1)>1e-4
   out.append(dict(E=E,mu=mu,cells=n,first_saved_edge_contact=float(t[edge][0]),last_saved_edge_contact=float(t[edge][-1]),first_saved_floor_contact=float(t[floor][0]) if floor.any() else None,final_rotation_deg=angles[-1],rotation_45deg_first_saved=float(t[np.array(angles)>45][0]) if (np.array(angles)>45).any() else None,final_sample_mean=x[-1].mean(0).tolist(),times=t.tolist(),rotation_deg=angles,sampled_mean=x.mean(1).tolist(),edge_contact_saved=edge.tolist(),floor_contact_saved=floor.tolist()))
(root/'reports/coupled-contact-v13/clipped-edge-phase-diagnostic.json').write_text(json.dumps(dict(scope='Development diagnostic from saved 100 Hz states; contact timing quantized to frames, not integration event logs. Sample means are not mass-weighted COM.',rows=out),indent=2)+'\n')
for r in out:print({k:v for k,v in r.items() if k not in ['times','rotation_deg','sampled_mean','edge_contact_saved','floor_contact_saved']})
