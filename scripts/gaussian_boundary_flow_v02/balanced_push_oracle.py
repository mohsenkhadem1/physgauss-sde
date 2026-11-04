"""Balanced finite-difference prefix-only mechanical diagnostic; never imports an evaluator."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
from scripts.external_benchmark.data import fps,rigid_map
from scripts.gaussian_boundary_flow_v02.export_inputs import load_package
from scripts.gaussian_boundary_flow_v02.push_backend import PrefixPlanePushBackend
from scripts.gaussian_boundary_flow_v02.object_backend_geometry import GeometryAwareGaussianBoundaryBackend

OUT=Path('reports/gaussian-boundary-flow-v02/balanced-push-oracle')
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
              budgets=[67,67,66],optimizer='L-BFGS-B with explicit forward differences, epsilon1e-3 logits; complete five-coordinate gradients only',
              finite_difference_step=.001,gradient_block_calls=6,
              leftover_budget='Deterministic local full-coordinate audit probes; no incomplete gradient blocks',
              qualitative_prefix_evidence='Root inspected public camera0 frames0,25,46: hand appears to pinch nose/face while moving toy; qualitative observation, not verified contact truth',
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
        gradients=[];cache={}
        class BudgetStop(Exception):pass
        def objective(z):
            nonlocal best
            if len(calls)>=budget:raise BudgetStop('Hard forward-call cap reached')
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
        def block(z):
            z=np.asarray(z,dtype=float);key=z.tobytes()
            if key in cache:return cache[key]
            # No incomplete gradients: reserve base plus all five perturbations.
            if budget-len(calls)<6:raise BudgetStop('Insufficient budget for a complete five-coordinate gradient')
            first=len(calls);value=objective(z);gradient=np.zeros(5);steps=[]
            for j in range(5):
                step=.001 if z[j]+.001<=6 else -.001
                perturbed=z.copy();perturbed[j]+=step
                gradient[j]=(objective(perturbed)-value)/step;steps.append(step)
            gradients.append(dict(base_call=first+1,perturbation_calls=list(range(first+2,first+7)),
                                  coordinate_ids=[0,1,2,3,4],steps=steps,
                                  failed_calls=sum(not c['completed'] for c in calls[first:first+6])))
            cache[key]=(value,gradient);return cache[key]
        stopped=False
        try:
            fit=minimize(lambda z:block(z)[0],start,jac=lambda z:block(z)[1],method='L-BFGS-B',
                         bounds=plan['bounds'],options=dict(maxiter=1000,ftol=1e-12,gtol=1e-6,maxls=20))
            optimizer_success=bool(fit.success);optimizer_message=str(fit.message)
        except BudgetStop as e:
            stopped=True;optimizer_success=False;optimizer_message=str(e)
        # Spend residual calls only on PREDECLARED local probes touching all five
        # coordinates, never a partial derivative. These are reported separately.
        audit_calls=[];anchor=np.array(best['logits']) if best is not None else start.copy()
        for j in range(budget-len(calls)):
            signs=np.array([1.,-1.,1.,-1.,1.])*(-1 if j%2 else 1)
            probe=np.clip(anchor+.001*(j+1)*signs,-6,6)
            objective(probe);audit_calls.append(len(calls))
        report=dict(law=law,hypothesis=h,standoff_m=offset,start=start.tolist(),budget=budget,
                    actual_calls=len(calls),optimizer_success=optimizer_success,optimizer_message=optimizer_message,
                    complete_gradient_blocks=gradients,local_audit_probe_calls=audit_calls,
                    optimizer_budget_stopped=stopped,
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
                          limitations='Balanced complete-coordinate finite-difference diagnostic; no global optimum, true-contact identification or forecast claim'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--law',choices=['bilateral','unilateral']);p.add_argument('--device',default='cpu');a=p.parse_args()
    prepare() if a.prepare else worker(a.law,a.device)
