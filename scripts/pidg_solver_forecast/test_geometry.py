"""Exact identity/affine binding and disconnected-volume tests."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.pidg_solver_forecast.geometry import volume
rng=np.random.default_rng(8);points=[]
for center in [[-.8,0,0],[.8,0,0]]:
 p=rng.normal(size=(1200,3));p=p/np.linalg.norm(p,axis=1)[:,None]*.3+center;points.append(p)
p=np.concatenate(points);g=volume(p,np.ones(len(p)));x=g['nodes'];e=g['elements'];ids=e[g['bind_tet']];N=g['bind_N'];error=float(abs(np.einsum('pa,pai->pi',N,x[ids])-p).max());assert error<1e-10;assert g['components']==2
labels=g['component_labels'];sets=[set(e[labels==l].ravel()) for l in np.unique(labels)];assert not sets[0]&sets[1]
A=np.array([[1.1,.1,0],[0,.95,.03],[0,0,1.]]);shift=np.array([.2,-.1,.05]);pred=p+np.einsum('pa,pai->pi',N,((x@A.T+shift)-x)[ids]);assert np.max(abs(pred-(p@A.T+shift)))<1e-10
report=dict(passed=True,identity_max_error=error,components=g['components'],node_sets_disjoint=True,affine_exact=True,scope='Proxy volume geometry and affine binding invariants; not a test of reconstructed anatomical/material correctness.')
(ROOT/'reports/pidg-solver-forecast-v01/geometry-tests.json').write_text(json.dumps(report,indent=2));print(report)
