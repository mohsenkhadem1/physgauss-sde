import json,math,itertools,time
from pathlib import Path
import torch
from scripts.gaussian_boundary_flow.refined import RefinedBlock
from scripts.gaussian_boundary_flow.qualification import refined_block
from scripts.gaussian_boundary_flow.position_action import lift
from physgauss.gaussian_boundary import gaussian_gate
from physgauss.physics import deformation

torch.set_num_threads(2);results={};qr=[]
def interval(c,s):return s*math.sqrt(math.pi/2)*(math.erf((.05-c)/(math.sqrt(2)*s))-math.erf((-.05-c)/(math.sqrt(2)*s)))
for n in (1,2,3,4):
 b=refined_block(n)
 for cx in (-.035,0.,.035):
  g=gaussian_gate(b.p,torch.tensor([[cx,.2,0.]],dtype=torch.float64),torch.diag(torch.tensor([.027**2,.02**2,.045**2],dtype=torch.float64))[None])*b.bottom
  numerical=float((g*b.a).sum());exact=interval(cx,.027)*interval(0,.045)
  qr.append({'n':n,'cx':cx,'integral':numerical,'exact':exact,'relative_error':(numerical-exact)/exact})
results['support_footprint_quadrature']=qr
b=RefinedBlock(dtype=torch.float64);z=torch.tensor(list(itertools.product((-8.,8.),repeat=6)),dtype=torch.float64)
maxs=[];minjs=[];positions=[]
with torch.no_grad():
 for j in range(0,len(z),8):
  u=b.rollout(z[j:j+8],corners_only=False);F=deformation(u+b.mesh.rest,b.mesh);strain=.5*(F+F.transpose(-1,-2))-torch.eye(3)
  maxs.extend(strain.abs().flatten(1).max(-1).values.tolist());minjs.extend(torch.linalg.det(F).flatten(1).min(-1).values.tolist())
results['parameter_box_corner_strain']={'cases':len(z),'max_abs_strain':max(maxs),'fraction_above_10pct':sum(s>.1 for s in maxs)/len(maxs),'min_J':min(minjs),'per_case_max_abs_strain':maxs,'worst_z':z[max(range(len(z)),key=lambda i:maxs[i])].tolist()}
torch.manual_seed(744);z=torch.logit(torch.rand(16,6,dtype=torch.float64).clamp(.001,.999))
with torch.no_grad():
 u=b.rollout(z,kind='release');ul=lift(b,z);delta=ul-u
 mean=delta.mean(-2,keepdim=True);d=delta[:,17:];res=d-d.mean(-2,keepdim=True)
 results['position_lift_added_motion']={'cases':16,'rms_added_motion_mm':float(d.square().mean().sqrt()*1000),'rms_after_removing_translation_mm':float(res.square().mean().sqrt()*1000),'nontranslation_fraction_RMS':float(res.square().mean().sqrt()/d.square().mean().sqrt())}
 print(results['parameter_box_corner_strain'],results['position_lift_added_motion'])
Path('reports/gaussian-boundary-flow-v02/numerical-audit-evidence.json').write_text(json.dumps(results,indent=2)+'\n')
