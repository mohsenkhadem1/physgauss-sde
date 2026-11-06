"""Quantify the MLS ridge's residual rotation bias instead of assuming exact objectivity."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.pidg_comparison.run import configure,EXTERNAL
configure(0);sys.path.insert(0,str(EXTERNAL/'upstream'))
import torch
from scripts.pidg_dense_mechanics.field import DenseMechanics
class G:
 def __init__(self):self.get_xyz=torch.randn(80,3,device='cuda')*.3;self.get_opacity=torch.ones(80,1,device='cuda')
class D:
 def step(self,q,t):
  d=q.new_zeros(len(q),7);d[:,0]=1;return {'d_xyz':d}
torch.manual_seed(40);m=DenseMechanics(particles=64,kernels=8,knots=8);m.initialize(D(),G());rows=[]
for angle in [0.,.5,1.,2.]:
 t=torch.tensor(angle,device='cuda');R=torch.stack([torch.stack([t.cos(),-t.sin(),t*0]),torch.stack([t.sin(),t.cos(),t*0]),torch.tensor([0.,0.,1.],device='cuda')]);x=(m.X@R.T).requires_grad_();force,F,J,E=m.internal(x);rows.append({'rotation_rad':angle,'energy':float(E.detach()),'force_rms':float(force.detach().square().mean().sqrt()),'net_force_norm':float(force.detach().sum(0).norm())})
_,_,_,E=m.internal((m.X*1.05).requires_grad_());r={'rows':rows,'five_percent_stretch_energy':float(E.detach()),'maximum_rotation_to_stretch_energy_ratio':max(x['energy'] for x in rows)/(float(E.detach())+1e-12),'limitation':'The frozen continuum energy is objective; regularized least-squares spatial reconstruction introduces residual rotation bias. This is quantified, not claimed exact.'};(ROOT/'reports/pidg-dense-mechanics-v01/objectivity-audit.json').write_text(json.dumps(r,indent=2));print(r)
