"""Prospective coupled drop/contact qualification; no trained-model claims."""
from pathlib import Path
import json,time
import numpy as np
import torch
from scripts.gaussian_boundary_flow_v02.object_dynamics import DynamicObjectBoundaryBackend,falling_cube_payload


def main():
    torch.set_num_threads(2);d=falling_cube_payload(duration=.5,step=.01);rows=[];saved={};previous=None
    for sub in (2,4,8):
        b=DynamicObjectBoundaryBackend(d,'cuda:1',substeps=sub,floor_mode='inferred');z=np.zeros(b.parameter_dim);start=time.time()
        try:
            x,diag=b.rollout(z,np.arange(len(d['times'])));center=x.mean(1);bottom=x[:,:,2].min(1);contact=np.flatnonzero(bottom<0.)
            first=int(contact[0])if len(contact)else None;post=center[first:,2]if first is not None else np.array([])
            low=int(np.argmin(post))if len(post)else 0
            row={'substeps':sub,'dt_s':.01/sub,'success':True,'seconds':time.time()-start,'first_sampled_contact_s':None if first is None else float(d['times'][first]),'maximum_penetration_mm':float(max(0,-bottom.min())*1000),'center_rebound_mm':None if not len(post)else float((post[low:].max()-post[low])*1000),'minJ':min(q['minJ']for q in diag),'max_residual':max(q['residual']or 0 for q in diag),'difference_previous_mm':None if previous is None else float(np.sqrt(np.mean((x-previous)**2))*1000)}
            previous=x;saved[f'substeps_{sub}']=x
        except Exception as e:row={'substeps':sub,'success':False,'error':str(e),'seconds':time.time()-start}
        rows.append(row);print(row,flush=True)
    path=Path('reports/gaussian-boundary-flow-v02/drop-qualification.json');path.write_text(json.dumps({'scope':'Untrained 8-node finite-strain drop with compliant normal plane; no Coulomb friction, no rendering or uncertainty claim','rows':rows,'exact_nonpenetration':False},indent=2)+'\n')
    np.savez_compressed('runs/gaussian-boundary-flow-v02-drop-trajectories.npz',times=d['times'],**saved)

if __name__=='__main__':main()
