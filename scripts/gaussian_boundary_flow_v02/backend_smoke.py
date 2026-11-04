"""Small exposed-case solver qualification before any bank or benchmark rollout."""
from pathlib import Path
import time,json
import numpy as np
from scripts.real_transfer_v19.phystwin_run import load
from scripts.gaussian_boundary_flow_v02.object_backend import ObjectBoundaryBackend


def main():
    data,meta=load('single_lift_sloth',64,True)
    b=ObjectBoundaryBackend(data,'cuda:1');frames=np.unique(np.linspace(0,data['train_end']-1,8,dtype=int))
    rng=np.random.default_rng(913);rows=[]
    # Neutral + three prospective prior draws; never choose based on held targets.
    for i,z in enumerate([np.zeros(b.parameter_dim),*rng.normal(size=(3,b.parameter_dim))]):
        start=time.time();row={'id':i,'parameters':z.tolist()}
        try:
            x,d=b.rollout(z,frames);row.update(ok=True,seconds=time.time()-start,minJ=min(a['minJ']for a in d),max_residual=max(a['residual']for a in d),iterations=sum(a['iterations']for a in d),maximum_displacement_m=float(np.linalg.norm(x-data['nodes'],axis=-1).max()))
        except Exception as e:row.update(ok=False,seconds=time.time()-start,error=str(e))
        rows.append(row);print(row,flush=True)
    out=Path('reports/gaussian-boundary-flow-v02/object-backend-smoke.json');out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({'scope':'Historically exposed sloth development prefix only; no forecast labels read','parameters':b.names,'frames':frames.tolist(),'data_manifest':meta,'results':rows},indent=2)+'\n')

if __name__=='__main__':main()
