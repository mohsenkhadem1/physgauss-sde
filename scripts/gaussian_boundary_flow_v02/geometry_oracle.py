"""Initial-geometry correction, prefix-only oracle, sealed held-motion predictions."""
from pathlib import Path
import json,time,hashlib
import numpy as np
from scipy.optimize import minimize
from scripts.gaussian_boundary_flow_v02.benchmark_inputs import load_phystwin_predictor,predictor_digest
from scripts.gaussian_boundary_flow_v02.export_inputs import write_package
from scripts.gaussian_boundary_flow_v02.object_backend_geometry import GeometryAwareGaussianBoundaryBackend
from scripts.external_benchmark.data import fps


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')


def main():
    case='single_lift_sloth';data=load_phystwin_predictor(case,64);b=GeometryAwareGaussianBoundaryBackend(data,'cuda:0')
    out=Path('runs/gaussian-boundary-flow-v02-geometry-oracle')/case;out.mkdir(parents=True,exist_ok=True);package=out/'input-package';write_package(data,package)
    frames=np.unique(np.linspace(0,data['train_end']-1,12,dtype=int));ids=fps(data['observed_initial'],48);mask=data['observed_mask'][frames][:,ids];target=data['observed'][frames][:,ids]
    plan={'case':case,'args':{'nodes':64},'input_package':str(package),'input_digest':predictor_digest(data),'input_package_manifest_sha256':sha(package/'manifest.json'),'prefix_frames':frames.tolist(),'point_ids':ids.tolist(),'prediction_methods':['oracle_raw','oracle_tracking'],'scope':'Classical prefix-only forward-adequacy oracle, not amortized inference; raw force-balanced mechanical states and separate incremental tracking map','geometry':b.geometry_diagnostic,'bounds':[b.lower.tolist(),b.upper.tolist()],'source_sha256':{str(p):sha(p)for p in [Path(__file__),Path('scripts/gaussian_boundary_flow_v02/object_backend_geometry.py'),Path('scripts/gaussian_boundary_flow_v02/object_backend_wide.py')]}}
    write(out/'plan.json',plan);best={};calls=0;start=time.time()
    def objective(z):
        nonlocal calls
        calls+=1
        try:
            x,diag=b.rollout(z,frames);pred=b.observation_features(x,frames,ids);loss=float(np.mean(((pred-target)[mask])**2))*1e6
            if not best or loss<best['loss_mm2']:best.update(loss_mm2=loss,z=z.tolist(),path=b.last_path_diagnostics.copy())
        except (ValueError,RuntimeError):loss=1e6
        if calls%12==0:
            write(out/'progress.json',{'calls':calls,'best':best,'seconds':time.time()-start});print('calls',calls,'best_mm',np.sqrt(best.get('loss_mm2',1e6)),flush=True)
        return loss
    old=np.array(json.loads(Path('reports/gaussian-boundary-flow-v02/wide-oracle.json').read_text())['best']['z']);candidates=[old,np.zeros(b.parameter_dim)];losses=[objective(z)for z in candidates];z=candidates[int(np.argmin(losses))]
    fit=minimize(objective,z,method='L-BFGS-B',bounds=[(-6,6)]*b.parameter_dim,options={'maxiter':30,'maxfun':220,'eps':1e-3,'ftol':1e-8})
    if not best:raise RuntimeError('No feasible prefix state')
    report={'scope':plan['scope'],'best':best,'prefix_rmse_mm':np.sqrt(best['loss_mm2']),'calls':calls,'seconds':time.time()-start,'optimizer_success':bool(fit.success),'message':str(fit.message)}
    write(out/'fit.json',report);write(Path('reports/gaussian-boundary-flow-v02/geometry-oracle.json'),report)
    write(out/'inference-sealed.json',{'forecast_target_access':False,'plan_sha256':sha(out/'plan.json'),'fit_sha256':sha(out/'fit.json'),'train_end':data['train_end']})
    raw,diag=b.rollout(np.array(best['z']),np.arange(len(data['times'])));tracking=data['nodes'][None]+raw-raw[:1]
    np.savez_compressed(out/'prediction.npz',oracle_raw=raw,oracle_tracking=tracking,nodes=data['nodes'],observation_matrix=data['observation_matrix'],observed_initial=data['observed_initial'])
    write(out/'complete.json',{'completed':True,'held_target_access':False,'fit_end':data['train_end'],'frames':len(raw),'prediction_sha256':sha(out/'prediction.npz'),'raw_physics_path':b.last_path_diagnostics,'tracking_map_is_force_balanced':False,'forecast_samples':{'oracle_raw':{'requested':1,'valid':1,'failures':[]},'oracle_tracking':{'requested':1,'valid':1,'failures':[]}}})
    print(report,flush=True)

if __name__=='__main__':main()
