"""Bounded matched prefix-only mechanical oracle; never imports an evaluator."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
from scripts.external_benchmark.data import fps,rigid_map
from scripts.gaussian_boundary_flow_v02.export_inputs import load_package
from scripts.gaussian_boundary_flow_v02.push_backend import PrefixPlanePushBackend
from scripts.gaussian_boundary_flow_v02.object_backend_geometry import GeometryAwareGaussianBoundaryBackend

OUT=Path('reports/gaussian-boundary-flow-v02/matched-push-oracle')
PACKAGE=Path('runs/gaussian-boundary-flow-v02-expansion-affine-inputs/single_push_sloth')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'plan.json').exists():raise RuntimeError('Frozen oracle plan already exists')
    data=load_package(PACKAGE)
    frames=np.unique(np.linspace(0,data['train_end']-1,12,dtype=int));ids=fps(data['observed_initial'],48)
    plan=dict(scope='Matched exposed-push PREFIX ONLY diagnostic, not global optimum or future prediction',
              package=str(PACKAGE),manifest_sha256=sha(PACKAGE/'manifest.json'),
              prefix_frames=frames.tolist(),point_ids=ids.tolist(),train_end=data['train_end'],
              objective='Mean squared relative displacement coordinate error over same v03 prefix mask, mm^2',
              bounds=[[-6.,6.]]*5,total_forward_evaluations_per_law=200,
              budgets=[67,67,66],optimizer='scipy.optimize.minimize Powell; maxfev hard counts, xtol/ftol1e-4',
              bilateral_starts=[[0.,0.,0.,0.,0.],[-1.,1.,0.,0.,0.],[1.,-1.,0.,0.,0.]],
              unilateral_starts=[[0.,0.,0.,0.,0.]]*3,unilateral_standoffs_m=[-.01,0.,.01],
              initial_diagnostic='First evaluation of each hypothesis retained, including absolute frame-zero mismatch',
              solver='Existing canonical full-command prefix continuation; unchanged residual<=1e-5 admission and element J>.03',
              failure_penalty_mm2=1e12,failure_policy='Every failed call counts; preserve failures and budget stops; no offset/normal/bound widening',
              source_sha256={str(p):sha(p) for p in [Path(__file__),Path('scripts/gaussian_boundary_flow_v02/push_backend.py'),Path('scripts/gaussian_boundary_flow_v02/unilateral_attachment.py'),Path('scripts/gaussian_boundary_flow_v02/object_backend_geometry.py'),Path('scripts/gaussian_boundary_flow_v02/object_backend_revision.py'),Path('scripts/gaussian_boundary_flow_v02/object_backend_wide.py'),Path('scripts/external_benchmark/fem.py')]})
    write(OUT/'plan.json',plan)


def worker(law,device):
    torch.set_num_threads(2);plan=json.loads((OUT/'plan.json').read_text())
    for name,digest in plan['source_sha256'].items():
        if sha(name)!=digest:raise ValueError('Changed frozen source: '+name)
    if sha(PACKAGE/'manifest.json')!=plan['manifest_sha256']:raise ValueError('Changed frozen package')
    result_path=OUT/(law+'-results.json')
    if result_path.exists():raise RuntimeError('Completed worker immutable')
    data=load_package(PACKAGE);frames=np.array(plan['prefix_frames']);ids=np.array(plan['point_ids'])
    target=data['observed'][frames][:,ids];mask=data['observed_mask'][frames][:,ids]
    target_relative=target-target[:1];initial=data['observed_initial'][ids]
    reports=[];begin=time.time()
    for h,budget in enumerate(plan['budgets']):
        output=OUT/f'{law}-hypothesis-{h}.json'
        if output.exists():raise RuntimeError('Partial/completed hypotheses preserved; no automatic restart')
        offset=plan['unilateral_standoffs_m'][h] if law=='unilateral' else None
        backend=(PrefixPlanePushBackend(data,device,prefix_frames=frames,standoff_m=offset) if law=='unilateral' else GeometryAwareGaussianBoundaryBackend(data,device))
        if backend.parameter_dim!=5:raise ValueError('Unexpected response parameter dimension')
        start=np.array(plan[law+'_starts'][h]);calls=[];best=None;t0=time.time()
        def objective(z):
            nonlocal best
            if len(calls)>=budget:raise RuntimeError('Hard call budget exceeded')
            record=dict(call=len(calls)+1,logits=np.asarray(z).tolist(),completed=False)
            try:
                positions,_=backend.rollout(z,frames)
                observed=backend.observation_features(positions,frames,ids)
                delta=((observed-observed[:1])-target_relative)[mask]
                loss=float(np.mean(delta**2)*1e6)
                shape=[]
                for i in range(len(frames)):
                    m=mask[i]
                    if m.sum()>=3:
                        R,b=rigid_map(observed[i,m],target[i,m]);shape.append(float(np.mean((observed[i,m]@R+b-target[i,m])**2)))
                record.update(completed=True,objective_mm2=loss,relative_coordinate_rmse_mm=float(np.sqrt(loss)),
                              absolute_coordinate_rmse_mm=float(np.sqrt(np.mean((observed-target)[mask]**2))*1000),
                              frame_zero_coordinate_rmse_mm=float(np.sqrt(np.mean((observed[0]-target[0])[mask[0]]**2))*1000),
                              aligned_shape_rmse_mm=float(np.sqrt(np.mean(shape))*1000),path=backend.last_path_diagnostics)
                if best is None or loss<best['objective_mm2']:best=record.copy()
            except (RuntimeError,ValueError) as e:
                record.update(objective_mm2=plan['failure_penalty_mm2'],failure=str(e))
            calls.append(record)
            write(OUT/f'{law}-progress.json',dict(hypothesis=h,calls_this_hypothesis=len(calls),budget=budget,best=best,elapsed_s=time.time()-begin))
            return record['objective_mm2']
        objective(start) # Explicit initial raw mismatch, counted in matched budget.
        fit=minimize(objective,start,method='Powell',bounds=plan['bounds'],
                     options=dict(maxfev=budget-1,xtol=1e-4,ftol=1e-4))
        report=dict(law=law,hypothesis=h,standoff_m=offset,start=start.tolist(),budget=budget,
                    actual_calls=len(calls),optimizer_success=bool(fit.success),optimizer_message=str(fit.message),
                    budget_exhausted=len(calls)>=budget,best=best,initial=calls[0],calls=calls,
                    seconds=time.time()-t0,plane_hypothesis=getattr(backend,'plane_hypothesis',None))
        write(output,report);reports.append(report)
        print(law,h,len(calls),None if best is None else best['relative_coordinate_rmse_mm'],flush=True)
    bests=[r['best'] for r in reports if r['best'] is not None]
    write(result_path,dict(law=law,plan_sha256=sha(OUT/'plan.json'),hypotheses=reports,
                          total_forward_calls=sum(r['actual_calls'] for r in reports),
                          best=min(bests,key=lambda r:r['objective_mm2']) if bests else None,
                          failed_calls=sum(not c['completed'] for r in reports for c in r['calls']),
                          seconds=time.time()-begin,held_target_access=False,
                          limitations='Local bounded prefix-only fits; no global optimum, true-contact identification or forecast claim'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--law',choices=['bilateral','unilateral']);p.add_argument('--device',default='cpu');a=p.parse_args()
    prepare() if a.prepare else worker(a.law,a.device)
