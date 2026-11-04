"""Smooth finite-range Gaussian grasp: gravity-triggered release, no impact."""
from pathlib import Path
import json
import numpy as np
import torch
from scipy.special import logit
from scripts.gaussian_boundary_flow_v02.object_dynamics import DynamicObjectBoundaryBackend,falling_cube_payload
from scripts.gaussian_boundary_flow_v02.robust_dynamic_step import FiniteRangeAttachmentFEMSafeguardedV3 as FiniteRangeAttachmentFEM


def main():
    torch.set_num_threads(1)
    d=falling_cube_payload(duration=.15,step=.005);n=len(d['nodes']);T=len(d['times']);top=d['nodes'][:,2]>.29
    d['controller_targets']=np.broadcast_to(d['nodes'],(T,1,n,3)).copy();d['attachment_weights']=top[None].astype(float);d['attachment_active']=np.ones((T,1));d['controls']=d['nodes'][top].mean(0)[None,None].repeat(T,0)
    rows=[];saved={}
    for name,logk in [('weak',.1),('strong',1.5)]:
        previous=None
        for sub in (2,4,8,16,32):
            try:
                b=DynamicObjectBoundaryBackend(d,'cpu',floor_mode='off',substeps=sub)
                b.model=FiniteRangeAttachmentFEM(d,'cpu',field=False,contact=False,nonconvex=False,materials=1,interaction_range=.015)
                b.base_weights=b.model.weights.clone()
                z=np.array([0.,logit(logk/7.),0.]);energy=[];solve=b.model.solve
                def tracked_solve(theta,frame,initial=None,**kwargs):
                    state,H,q=solve(theta,frame,initial,**kwargs)
                    m=b.model;old_target=m.inertial_target
                    velocity=(state-initial)/m.inertial_dt
                    m.inertial_target=state
                    potential,_,_=m.evaluate(state,theta,frame)
                    m.inertial_target=old_target
                    kinetic=.5*theta[0].exp()*m.scale/9.81*(m.mass[:,None]*velocity**2).sum()
                    energy.append(float(potential+kinetic))
                    return state,H,q
                b.model.solve=tracked_solve
                x,diag=b.rollout(z,np.arange(T));center=x.mean(1)
                # The hand command remains fixed and active; no manual release switch.
                row={'grasp':name,'substeps':sub,'dt_s':.005/sub,'terminal_drop_mm':float((d['nodes'].mean(0)[2]-center[-1,2])*1000),'minJ':min(v['minJ']for v in diag),'maximum_residual':b.last_path_diagnostics['max_residual'],'difference_previous_mm':None if previous is None else float(np.sqrt(np.mean((x-previous)**2))*1000)}
                energy_scale=float(b.bind(z)[0].exp())*.5*9.81*.15**2/b.model.scale
                row['maximum_energy_error_fraction_freefall_potential']=float(np.max(np.abs(energy))/energy_scale)
                row['passes_refinement_gates']=row['difference_previous_mm'] is not None and row['difference_previous_mm']<=.1 and row['maximum_energy_error_fraction_freefall_potential']<=.01
                rows.append(row);previous=x;saved[f'{name}_{sub}']=x;print(row,flush=True)
            except Exception as error:
                row={'grasp':name,'substeps':sub,'success':False,'error':str(error)}
                rows.append(row);print(row,flush=True)
    Path('reports/gaussian-boundary-flow-v02/release-safeguarded-v3-qualification.json').write_text(json.dumps({'scope':'Untrained effective cohesive-grasp operator; fixed active hand, gravity-driven detachment; no Coulomb friction or impact','protocol':'solver-safeguarded-v3','inner_solver_target':1e-8,'unchanged_admission_residual_gate':1e-7,'trajectory_gate_mm':.1,'known_interaction_range_dimensionless':.015,'rows':rows},indent=2)+'\n')
    np.savez_compressed('runs/gaussian-boundary-flow-v02-release-safeguarded-v3-trajectories.npz',times=d['times'],**saved)

if __name__=='__main__':main()
