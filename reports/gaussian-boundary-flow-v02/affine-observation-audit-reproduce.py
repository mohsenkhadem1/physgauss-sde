from pathlib import Path
import json
import numpy as np
from scripts.gaussian_boundary_flow_v02.affine_observation import affine_observation_matrix
from scripts.gaussian_boundary_flow_v02.export_inputs import load_package,sha

rows=[]
for directory in sorted(Path('runs/gaussian-boundary-flow-v02-inputs').iterdir()):
    if not(directory/'manifest.json').exists():continue
    p=load_package(directory);nodes=p['nodes'];points=p['observed_initial'];old=p['observation_matrix']
    A,d=affine_observation_matrix(nodes,points)
    # Pure 90 degree material rotation about z; no future targets needed.
    R=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    rigid=nodes@R
    old_pred=points+old@(rigid-nodes);new_pred=points+A@(rigid-nodes);truth=points@R
    row={'name':p['name'],'manifest_sha256':sha(directory/'manifest.json'),'nodes':len(nodes),'points':len(points),'old_initial_geometry_rms_mm':float(np.sqrt(np.mean((old@nodes-points)**2))*1000),'new_initial_geometry_rms_mm':float(np.sqrt(np.mean((A@nodes-points)**2))*1000),'old_rotation90_rms_mm':float(np.sqrt(np.mean((old_pred-truth)**2))*1000),'new_rotation90_rms_mm':float(np.sqrt(np.mean((new_pred-truth)**2))*1000),'maximum_l1':d['maximum_row_l1'],'maximum_l2':d['maximum_row_l2'],'old_maximum_l1':float(abs(old).sum(1).max()),'old_maximum_l2':float(np.linalg.norm(old,axis=1).max()),'negative_row_fraction':float(np.mean((A < -1e-14).any(1))),'maximum_negative_weight':float(A.min()),'neighbors_used_quantiles':np.quantile([r['neighbors']for r in d['rows']],[0,.5,1]).tolist(),'max_nonzero_condition':max(r['condition_nonzero']for r in d['rows'])}
    rows.append(row);print(row['name'],row['old_rotation90_rms_mm'],row['new_rotation90_rms_mm'],row['maximum_l1'],flush=True)
Path('reports/gaussian-boundary-flow-v02/affine-observation-audit.json').write_text(json.dumps({'scope':'Eight existing immutable initial-geometry packages only; no future targets or existing package mutations','method':'16-neighbor affine-complete corrected inverse-distance map; expand stencil if needed; L1<=8','rows':rows},indent=2)+'\n')
