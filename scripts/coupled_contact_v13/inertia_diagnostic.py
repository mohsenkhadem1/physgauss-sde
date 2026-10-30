"""Compare row-sum lumped tetrahedral inertia with an analytic uniform block."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.coupled_contact_v13.physics import FEMSystem,make_block,Material
from scripts.coupled_contact_v13.campaign import REPORT,write
size=np.array([.08,.06,.05]);density=1000.;mass=np.prod(size)*density
analytic=np.diag(mass*(np.sum(size**2)-size**2)/12)
rows=[]
for n in [2,3,4,6,8,10]:
 rest,tets=make_block(size,(n,n,n),(0,0,0));s=FEMSystem(rest,tets,Material(density=density),[])
 center=np.sum(s.mass[:,None]*rest,axis=0)/np.sum(s.mass);r=rest-center
 I=np.eye(3)*np.sum(s.mass*np.sum(r*r,axis=1))-np.einsum('n,ni,nj->ij',s.mass,r,r)
 rows.append(dict(cells=n,mass_kg=float(s.mass.sum()),center_m=center.tolist(),lumped_inertia_kg_m2=I.tolist(),diagonal_relative_bias=(np.diag(I)/np.diag(analytic)-1).tolist(),relative_frobenius_error=float(np.linalg.norm(I-analytic)/np.linalg.norm(analytic))))
write(REPORT/'lumped-inertia-diagnostic.json',dict(scope='Initial geometric inertial quadrature diagnostic; no dynamics or algorithm changes',analytic_inertia_kg_m2=analytic.tolist(),rows=rows))
for r in rows:print(r['cells'],r['diagonal_relative_bias'],r['relative_frobenius_error'])
