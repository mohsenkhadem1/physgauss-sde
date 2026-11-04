"""Bounded untrained drop refinement; no admission inferred from successful solves."""
import json,time
from pathlib import Path
import numpy as np
import torch
from scripts.gaussian_boundary_flow_v02.object_dynamics import falling_cube_payload
from scripts.gaussian_boundary_flow_v02.object_midpoint import MidpointObjectBoundaryBackend


def main():
    torch.set_num_threads(1)
    data=falling_cube_payload(duration=.5,step=.01);rows=[];saved={};previous=None
    # Frozen gates: successive trajectory RMS <= 0.1mm AND rebound change <=1%.
    for sub in (64,128,256):
        b=MidpointObjectBoundaryBackend(data,'cpu',floor_mode='inferred',substeps=sub)
        start=time.time()
        try:
            x,diag=b.rollout(np.zeros(b.parameter_dim),np.arange(len(data['times'])))
            center=x.mean(1);bottom=x[:,:,2].min(1);low=np.argmin(center[:,2])
            rebound=float((center[low:,2].max()-center[low,2])*1000)
            difference=None if previous is None else float(np.sqrt(np.mean((x-previous)**2))*1000)
            change=None if not rows or not rows[-1]['success'] else abs(rebound-rows[-1]['center_rebound_mm'])/max(rebound,1e-12)
            row={'substeps':sub,'dt_s':.01/sub,'success':True,'seconds':time.time()-start,'center_rebound_mm':rebound,'sampled_penetration_mm':float(max(0,-bottom.min())*1000),'minJ':b.last_path_diagnostics['minimum_J'],'max_residual':b.last_path_diagnostics['max_residual'],'difference_previous_mm':difference,'rebound_relative_change':change,'passes_refinement_gates':difference is not None and difference<=.1 and change<=.01}
            energy_scale=float(b.bind(np.zeros(b.parameter_dim))[0].exp())*(data['nodes'].mean(0)[2]-data['floor'])/b.model.scale
            energy_error=float(np.max(abs(b.last_energies-b.last_energies[0]))/energy_scale)
            row.update(maximum_penetration_mm=b.last_maximum_penetration*1000,maximum_energy_error_fraction_drop_potential=energy_error,final_energy_error_fraction_drop_potential=float((b.last_energies[-1]-b.last_energies[0])/energy_scale))
            row['passes_refinement_gates']=row['passes_refinement_gates'] and energy_error<=.01
            previous=x;saved[f'substeps_{sub}']=x
        except Exception as e:row={'substeps':sub,'success':False,'error':str(e),'seconds':time.time()-start}
        rows.append(row);print(row,flush=True)
        Path('reports/gaussian-boundary-flow-v02/midpoint-qualification.json').write_text(json.dumps({'scope':'8node conservative compliant-floor drop; CPU; exposed synthetic neutral world only','criteria':{'successive_trajectory_rms_mm':.1,'rebound_relative_change':.01,'two_successive_refinements_required':True,'maximum_energy_error_fraction_drop_potential':.01},'rows':rows,'admitted':len(rows)>2 and all(r.get('passes_refinement_gates',False) for r in rows[-2:]),'exact_nonpenetration':False},indent=2)+'\n')
    np.savez_compressed('runs/gaussian-boundary-flow-v02-midpoint-trajectories.npz',times=data['times'],**saved)

if __name__=='__main__':main()
