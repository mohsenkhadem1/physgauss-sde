"""Read-only hash-bound comparison; never alters the prospective gates."""
import json, hashlib, os
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def record(p):return dict(path=str(p.relative_to(ROOT)),sha256=sha(p))
op=ROOT/'runs/operator-dev/p2-contact-dop853-v03';out=ROOT/'runs/gaussian-campaign-v01/p2-contact-dop853-v03'
s=json.loads((out/'summary.json').read_text());assert s.get('completed') is True;captures=[];arrays=[]
for row in s['cases']:
 p=ROOT/row['capture']['path'];assert sha(p)==row['capture']['sha256'];captures.append(record(p));arrays.append(np.load(p,allow_pickle=False))
a,b=arrays;np.testing.assert_array_equal(a['times_s'],b['times_s']);dv=a['velocities_m_s']-b['velocities_m_s'];dq=a['positions_m']-b['positions_m'];i,n=map(int,np.unravel_index(np.linalg.norm(dv,axis=2).argmax(),dv.shape[:2]));component=int(abs(dv[i,n]).argmax());corners=int(b['tets'].max())+1
from experiments.run_p2_contact_validation import build
from experiments.campaign_p2_consistent_transient import scalar_mass
m=build([4,2,2],300000.,9.81);M=scalar_mass(m);mass=float(M.sum());rms=[float(np.sqrt(np.sum(e*(M@e))/(3*mass))) for e in dv];com=[float(np.linalg.norm(np.asarray(M.sum(axis=1)).ravel()@e/mass)) for e in dv]
r=dict(schema='DOP853-tighter-tightest-accuracy-pair-diagnosis-v0.3',captures=captures,maximum_velocity_difference_m_s=float(np.linalg.norm(dv,axis=2).max()),argmax=dict(output_index=i,time_s=float(b['times_s'][i]),node=n,node_kind='P1_vertex' if n<corners else 'P2_edge_midpoint',rest_m=b['rest_m'][n].tolist(),loose_velocity_m_s=a['velocities_m_s'][i,n].tolist(),tight_velocity_m_s=b['velocities_m_s'][i,n].tolist(),difference_components_m_s=dv[i,n].tolist(),largest_component='xyz'[component]),maximum_allnode_position_difference_m=float(np.linalg.norm(dq,axis=2).max()),maximum_consistent_mass_coordinate_RMS_velocity_difference_m_s=max(rms),maximum_COM_velocity_difference_m_s=max(com),interpretation='All-node velocity and fixed-query position gates remain primary; mass-weighted RMS/COM are supplementary descriptions only. Two tolerances and conservation checks are numerical evidence, not rigorous exact-error bounds.',source=record(Path(__file__)))
(op/'pair-diagnosis.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
