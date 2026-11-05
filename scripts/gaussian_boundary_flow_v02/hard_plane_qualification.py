"""Small CPU-only static contact qualification; no benchmark/training admission."""
from pathlib import Path
import json,time
import numpy as np
import torch
from scripts.gaussian_boundary_flow_v02.object_dynamics import falling_cube_payload
from scripts.gaussian_boundary_flow_v02.hard_plane_static import StaticHardPlaneFEM


def main():
    torch.set_num_threads(1);rows=[]
    for kind,gravity in [(k,g)for k in ('gravity','push','lift')for g in(.003,.03,.3)]+[('asymmetric_push',.03)]:
        data=falling_cube_payload(duration=.01);data['nodes']=data['nodes']-[0,0,.2]
        if kind!='gravity':
            weights=(data['nodes'][:,2]>.05).astype(float)
            if kind=='asymmetric_push':weights[:]=0;weights[-1]=1
            data['attachment_weights']=weights[None];data['attachment_active']=np.ones((2,1));data['controller_targets']=np.broadcast_to(data['nodes'],(2,1,8,3)).copy()
            data['controller_targets'][:,:,:,2]+=.04 if kind=='lift'else-.005
        model=StaticHardPlaneFEM(data,device='cpu',field=False);theta=torch.zeros(model.i_field+1,dtype=model.dtype);theta[0]=np.log(gravity)
        if kind!='gravity':theta[1]=np.log(2.)
        start=time.monotonic();row={'kind':kind,'gravity_material_ratio':gravity,'soft_floor_parameter':'ignored compatibility slot; not an inference parameter'}
        try:
            q,_,d=model.solve(theta,0,tolerance=1e-8,maxiter=100)
            _,r,_=model.evaluate(q,theta,0);reaction=model.t(d['normal_reactions']);kkt=model.kkt(q,r)
            sumforce=r.sum(0)-torch.stack([reaction.sum()*0,reaction.sum()*0,reaction.sum()]);contactforces=torch.zeros_like(q);contactforces[:,2]=reaction
            moment=torch.cross(q,r-contactforces,dim=1).sum(0)
            row.update(completed=True,**d,total_reaction=float(reaction.sum()),force_balance_norm=float(torch.linalg.vector_norm(sumforce)),moment_balance_norm=float(torch.linalg.vector_norm(moment)),minimum_gap_m=d['minimum_gap']*model.scale,maximum_penetration_m=d['maximum_penetration']*model.scale)
        except(ValueError,RuntimeError)as error:row.update(completed=False,error=str(error))
        row['seconds']=time.monotonic()-start;rows.append(row);print(kind,gravity,row.get('completed'),row.get('residual'),row.get('minJ'),flush=True)
    report={'scope':'Ten controlled8nodeCPU static frictionless hard-plane worlds; no training, benchmark or impact claim','gates':{'stationarity_norm':1e-8,'penetration_dimensionless':1e-12,'complementarity':1e-10,'minimum_J':.03,'reactions':'nonnegative; zero on separated nodes'},'rows':rows,'all_controlled_cases_pass':all(r['completed']for r in rows),'future_backend_contract':'Remove ignored soft-floor stiffness coordinate from latent vector; fixed known floor geometry only','original_ExternalFEM_modified':False}
    Path('reports/gaussian-boundary-flow-v02/hard-plane-static-qualification.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
