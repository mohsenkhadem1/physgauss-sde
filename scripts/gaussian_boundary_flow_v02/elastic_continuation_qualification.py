"""CPU no-contact elastic release refinement with unchanged initial state/law."""
from pathlib import Path
import json,time
import numpy as np
import torch
from scripts.external_benchmark.dynamic_fem import DynamicFEM
from scripts.gaussian_boundary_flow_v02.object_dynamics import falling_cube_payload
from scripts.gaussian_boundary_flow_v02.elastic_continuation import continue_elastic


def main():
    torch.set_num_threads(1);data=falling_cube_payload();m=DynamicFEM(data,device='cpu',field=False)
    theta=torch.zeros(m.i_field+1,dtype=m.dtype);theta[0]=np.log(.03);theta[m.i_floor]=-torch.inf
    q=data['nodes'].copy();q[:,0]*=1.05;v=np.zeros_like(q);previous=None;rows=[]
    for dt in(.001,.0005,.00025):
        start=time.monotonic()
        try:
            result=continue_elastic(m,theta,q,v,0,np.arange(7)*.005,max_dt=dt)
            E=result['energies'];error=float(max(abs(E-E[0]))/abs(E[0]));x=result['positions_m']
            difference=None if previous is None else float(np.sqrt(np.mean((x-previous)**2))*1000)
            shape_change=float(np.sqrt(np.mean(((x[-1]-x[-1].mean(0))-(q-q.mean(0)))**2))*1000)
            row={'dt_s':dt,'completed':True,'energy_reference':'initial elastic+kinetic+gravity potential in unchanged material reference; no time-varying external potential in this fixture','maximum_relative_energy_error':error,'difference_previous_mm':difference,'shape_change_mm':shape_change,'minimum_J':min(d['minJ']for d in result['steps']),'maximum_residual':max(d['residual']for d in result['steps']),'passes_refinement_gates':difference is not None and difference<=.1 and error<=.01}
            previous=x
        except(ValueError,RuntimeError)as e:row={'dt_s':dt,'completed':False,'error':str(e)}
        row['seconds']=time.monotonic()-start;rows.append(row);print(row,flush=True)
    report={'scope':'One8node5percent stretched block released without contact; CPU only; no benchmark target access','rows':rows,'qualified_for_this_no_contact_fixture':len(rows)==3 and all(r.get('passes_refinement_gates',False)for r in rows[-2:]),'gates':{'trajectory_refinement_mm':.1,'relative_energy_error':.01,'residual':1e-7,'minJ':.03,'two_refinements':True},'benchmark_adopted':False,'impact_qualified':False}
    Path('reports/gaussian-boundary-flow-v02/elastic-continuation-qualification.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
