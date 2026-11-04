"""Prefix-only adequacy of the added Gaussian table operator; not deployment."""
from pathlib import Path
import json,time
import numpy as np
from scipy.optimize import minimize
from scripts.gaussian_boundary_flow_v02.benchmark_inputs import load_phystwin_predictor
from scripts.gaussian_boundary_flow_v02.object_backend_table import GaussianTableBoundaryBackend
from scripts.external_benchmark.data import fps


def main():
    d=load_phystwin_predictor('single_lift_sloth',64);b=GaussianTableBoundaryBackend(d,'cuda:0')
    frames=np.unique(np.linspace(0,d['train_end']-1,12,dtype=int));ids=fps(d['observed_initial'],48);mask=d['observed_mask'][frames][:,ids];target=d['observed'][frames][:,ids]
    start=time.time();best={};calls=0
    def objective(z):
        nonlocal calls
        calls+=1
        try:
            x,diag=b.rollout(z,frames);pred=b.observation_features(x,frames,ids)
            loss=float(np.mean(((pred-target)[mask])**2))*1e6
            if not best or loss<best['loss_mm2']:best.update(loss_mm2=loss,z=z.tolist(),minJ=min(v['minJ']for v in diag),max_residual=max(v['residual']for v in diag))
        except (ValueError,RuntimeError):loss=1e6
        return loss
    z=np.array([1.9,-5.,5.,-3.,0.])
    result=minimize(objective,z,method='L-BFGS-B',bounds=[(-5,5)]*5,options={'maxiter':20,'maxfun':150,'eps':1e-3,'ftol':1e-8})
    out={'scope':'Exposed prefix-only finite-strain forward-adequacy diagnostic; raw mechanical states, not rebased; iterative oracle not deployed predictor','operator':'Conservative Gaussian tangential table compliance, not Coulomb friction','best':best,'prefix_rmse_mm':np.sqrt(best['loss_mm2']),'calls':calls,'seconds':time.time()-start,'optimizer_success':bool(result.success),'message':str(result.message)}
    Path('reports/gaussian-boundary-flow-v02/table-oracle.json').write_text(json.dumps(out,indent=2)+'\n');print(out)

if __name__=='__main__':main()
