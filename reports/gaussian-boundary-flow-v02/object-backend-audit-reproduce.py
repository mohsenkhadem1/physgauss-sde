"""Independent prefix-only floor/schedule diagnostic. Does not mutate parent code."""
import json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from scripts.gaussian_boundary_flow_v02.benchmark_inputs import load_phystwin_predictor,predictor_digest
from scripts.gaussian_boundary_flow_v02.object_backend import ObjectBoundaryBackend

class NoFloorDiagnostic(ObjectBoundaryBackend):
    def bind(self,z):
        theta=super().bind(z);theta[self.model.i_floor]=-torch.inf;return theta

def main():
    torch.set_num_threads(2);data=load_phystwin_predictor('single_lift_sloth',max_nodes=64)
    sparse=np.unique(np.linspace(0,data['train_end']-1,12,dtype=int));dense=np.arange(data['train_end'])
    on=ObjectBoundaryBackend(data,'cuda:2');off=NoFloorDiagnostic(data,'cuda:2')
    rng=np.random.default_rng(913);zs=[np.zeros(on.parameter_dim),*rng.normal(size=(3,on.parameter_dim))];rows=[]
    for i,z in enumerate(zs):
        row={'world':i,'z':z.tolist(),'runs':{}};states={}
        for label,model,frames in [('on_sparse',on,sparse),('on_dense',on,dense),('off_sparse',off,sparse),('off_dense',off,dense)]:
            start=time.time()
            try:
                x,d=model.rollout(z,frames);states[label]=x if 'sparse' in label else x[sparse]
                row['runs'][label]={'ok':True,'seconds':time.time()-start,'minimum_J':min(a['minJ']for a in d),'maximum_residual':max(a['residual']for a in d),'iterations':sum(a['iterations']for a in d)}
            except Exception as e:row['runs'][label]={'ok':False,'seconds':time.time()-start,'error':repr(e)}
        metrics={}
        for label,a,b in [('schedule_floor_on','on_sparse','on_dense'),('schedule_floor_off','off_sparse','off_dense'),('floor_effect_sparse','on_sparse','off_sparse'),('floor_effect_dense','on_dense','off_dense')]:
            if a in states and b in states:
                delta=states[a]-states[b];relative=(states[a]-states[a][:1])-(states[b]-states[b][:1])
                metrics[label]={'absolute_RMS_mm':float(np.sqrt(np.mean(delta**2))*1000),'relative_to_first_RMS_mm':float(np.sqrt(np.mean(relative**2))*1000),'maximum_nodal_distance_mm':float(np.linalg.norm(delta,axis=-1).max()*1000)}
        row['comparisons']=metrics;rows.append(row);print(row,flush=True)
    result={'scope':'Four exposed sloth prior worlds, prefix only; no target observations used for selection/scoring','input_digest':predictor_digest(data),'frames_sparse':sparse.tolist(),'frames_dense':dense.tolist(),'seed':913,'results':rows,'source_sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest()for p in ('scripts/gaussian_boundary_flow_v02/object_backend.py','scripts/gaussian_boundary_flow_v02/object_campaign.py')}}
    Path('reports/gaussian-boundary-flow-v02/object-backend-audit-evidence.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
