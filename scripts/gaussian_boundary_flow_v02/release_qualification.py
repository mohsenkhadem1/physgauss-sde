"""Smooth finite-range Gaussian grasp: gravity-triggered release, no impact."""
from pathlib import Path
import json
import numpy as np
from scipy.special import logit
from scripts.gaussian_boundary_flow_v02.object_dynamics import DynamicObjectBoundaryBackend,falling_cube_payload
from scripts.gaussian_boundary_flow_v02.finite_range_attachment import FiniteRangeAttachmentFEM


def main():
    d=falling_cube_payload(duration=.15,step=.005);n=len(d['nodes']);T=len(d['times']);top=d['nodes'][:,2]>.29
    d['controller_targets']=np.broadcast_to(d['nodes'],(T,1,n,3)).copy();d['attachment_weights']=top[None].astype(float);d['attachment_active']=np.ones((T,1));d['controls']=d['nodes'][top].mean(0)[None,None].repeat(T,0)
    rows=[];saved={}
    for name,logk in [('weak',.1),('strong',1.5)]:
        previous=None
        for sub in (2,4,8):
            b=DynamicObjectBoundaryBackend(d,'cuda:0',floor_mode='off',substeps=sub)
            b.model=FiniteRangeAttachmentFEM(d,'cuda:0',field=False,contact=False,nonconvex=False,materials=1,interaction_range=.015)
            b.base_weights=b.model.weights.clone()
            z=np.array([0.,logit(logk/7.),0.]);x,diag=b.rollout(z,np.arange(T));center=x.mean(1)
            # The hand command remains fixed and active; no manual release switch.
            row={'grasp':name,'substeps':sub,'dt_s':.005/sub,'terminal_drop_mm':float((d['nodes'].mean(0)[2]-center[-1,2])*1000),'minJ':min(v['minJ']for v in diag),'maximum_residual':b.last_path_diagnostics['max_residual'],'difference_previous_mm':None if previous is None else float(np.sqrt(np.mean((x-previous)**2))*1000)}
            rows.append(row);previous=x;saved[f'{name}_{sub}']=x;print(row,flush=True)
    Path('reports/gaussian-boundary-flow-v02/release-qualification.json').write_text(json.dumps({'scope':'Untrained effective cohesive-grasp operator; fixed active hand, gravity-driven detachment; no Coulomb friction or impact','known_interaction_range_dimensionless':.015,'rows':rows},indent=2)+'\n')
    np.savez_compressed('runs/gaussian-boundary-flow-v02-release-trajectories.npz',times=d['times'],**saved)

if __name__=='__main__':main()
