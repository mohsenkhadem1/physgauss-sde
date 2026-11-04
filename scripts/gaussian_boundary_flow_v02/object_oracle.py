"""Development-only forward adequacy test; this iterative fit is NOT the predictor.

Fits only the permitted prefix and tests whether support topology/rest assumptions
can explain it. Raw mechanical states and rebased tracking maps remain distinct.
"""
from pathlib import Path
import argparse,json,time
import numpy as np
from scipy.optimize import minimize
from scripts.gaussian_boundary_flow_v02.benchmark_inputs import load_phystwin_predictor
from scripts.gaussian_boundary_flow_v02.object_backend_revision import ObjectBoundaryBackendRevision
from scripts.external_benchmark.data import fps


def main():
    p=argparse.ArgumentParser();p.add_argument('--case',default='single_lift_sloth');args=p.parse_args()
    data=load_phystwin_predictor(args.case,max_nodes=64);frames=np.unique(np.linspace(0,data['train_end']-1,12,dtype=int));ids=fps(data['observed_initial'],48)
    target=data['observed'][frames][:,ids];mask=data['observed_mask'][frames][:,ids]
    out=Path('reports/gaussian-boundary-flow-v02')/f'oracle-{args.case}.json';rows=[]
    for floor in ('inferred','off'):
        b=ObjectBoundaryBackendRevision(data,'cuda:1',floor_mode=floor)
        for mapping in ('raw_mechanics','incremental_tracking'):
            history=[];start=time.time();best={}
            def objective(z):
                try:
                    x,diag=b.rollout(z,frames)
                    pred=b.observation_features(x,frames,ids)
                    if mapping=='incremental_tracking':pred=pred-pred[:1]+data['observed_initial'][ids][None]
                    error=(pred-target)[mask];loss=float(np.mean(error**2))*1e6
                    if not best or loss<best['loss_mm2']:best.update(loss_mm2=loss,z=z.tolist(),minJ=min(d['minJ']for d in diag),max_residual=max(d['residual']for d in diag),initial_error_mm=float(np.sqrt(np.mean((pred[0]-target[0])**2)))*1000)
                except (RuntimeError,ValueError):loss=1e6
                history.append(loss)
                return loss
            fit=minimize(objective,np.zeros(b.parameter_dim),method='L-BFGS-B',bounds=[(-5,5)]*b.parameter_dim,options={'maxiter':25,'maxfun':180,'ftol':1e-8,'finite_diff_rel_step':1e-4,'eps':1e-3})
            row={'floor':floor,'mapping':mapping,'best':best,'prefix_rmse_mm':float(np.sqrt(best['loss_mm2']))if best else None,'optimizer_success':bool(fit.success),'message':str(fit.message),'evaluations':len(history),'seconds':time.time()-start};rows.append(row);print(row,flush=True)
            out.write_text(json.dumps({'scope':'Exposed observed-prefix adequacy diagnostic only; iterative oracle optimization, not the deployed amortized predictor; no future object targets used','case':args.case,'frames':frames.tolist(),'rows':rows},indent=2)+'\n')

if __name__=='__main__':main()
