"""Prefix-only bounded-prior adequacy diagnostic, no new architecture tuning."""
from pathlib import Path
import json,time
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit,logit
from scripts.gaussian_boundary_flow_v02.benchmark_inputs import load_phystwin_predictor
from scripts.gaussian_boundary_flow_v02.object_backend_wide import WideGaussianTableBoundaryBackend
from scripts.external_benchmark.data import fps


def main():
    d=load_phystwin_predictor('single_lift_sloth',64);b=WideGaussianTableBoundaryBackend(d,'cuda:0')
    frames=np.unique(np.linspace(0,d['train_end']-1,12,dtype=int));ids=fps(d['observed_initial'],48);mask=d['observed_mask'][frames][:,ids];target=d['observed'][frames][:,ids]
    old=json.loads(Path('reports/gaussian-boundary-flow-v02/table-oracle.json').read_text())['best']['z']
    lower=np.array([-8.,0.,3.,-.8,-8.]);upper=np.array([-1.,7.,8.,.8,4.]);physical=lower+(upper-lower)*expit(old)
    z=logit(np.clip((physical-b.lower)/(b.upper-b.lower),1e-5,1-1e-5));start=time.time();best={};calls=0
    def objective(z):
        nonlocal calls
        calls+=1
        try:
            x,diag=b.rollout(z,frames);pred=b.observation_features(x,frames,ids)
            loss=float(np.mean(((pred-target)[mask])**2))*1e6
            if not best or loss<best['loss_mm2']:best.update(loss_mm2=loss,z=z.tolist(),physical_ratios=(b.lower+(b.upper-b.lower)*expit(z)).tolist(),minJ=min(v['minJ']for v in diag),max_residual=max(v['residual']for v in diag))
        except (ValueError,RuntimeError):loss=1e6
        if calls%12==0:print('calls',calls,'best_mm',np.sqrt(best.get('loss_mm2',1e6)),flush=True)
        return loss
    result=minimize(objective,z,method='L-BFGS-B',bounds=[(-6,6)]*5,options={'maxiter':30,'maxfun':220,'eps':1e-3,'ftol':1e-8})
    out={'scope':'Exposed prefix-only forward-adequacy oracle, raw states, not deployed predictor','prior_bounds':[b.lower.tolist(),b.upper.tolist()],'prior_basis':'Historical external-FEM gravity/attachment/floor bounds plus broader Gaussian footprint and tangential compliance','best':best,'prefix_rmse_mm':np.sqrt(best['loss_mm2']),'calls':calls,'seconds':time.time()-start,'optimizer_success':bool(result.success),'message':str(result.message)}
    Path('reports/gaussian-boundary-flow-v02/wide-oracle.json').write_text(json.dumps(out,indent=2)+'\n');print(out)

if __name__=='__main__':main()
