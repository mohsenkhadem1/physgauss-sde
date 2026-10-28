"""Summarize sealed v07 evidence without hiding incomplete work."""
from pathlib import Path
import json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.phystwin.v07_study import BASE,OUT,SOURCE,POLICIES,protocol,write
from scripts.phystwin.sequential_policy import verify

def ci(delta):
 # Paired object bootstrap, stratified by the three support layouts.
 rng=np.random.default_rng(7107);ids=np.concatenate([rng.choice(np.arange(l,len(delta),3),(10000,len(delta)//3),replace=True) for l in range(3)],axis=1)
 return np.quantile(delta[ids].mean(1),[.025,.975]).tolist()

def main():
 s=protocol();seal=json.loads((BASE/'seal.json').read_text());verify(seal);summary=dict(protocol=s,calibration=seal['method'],policies={},comparisons={},numerics={},phystwin=[])
 errors={};costs={};scales=seal['scales']
 for policy in POLICIES:
  z=np.load(BASE/f'test-{policy}-65.npz');err=z['mean']-z['truth'][:,None];plans=json.loads((BASE/'plans'/f'test-{policy}.json').read_text());cost=np.array([sum(s['costs'][a] for a in r['history']) for r in plans]).reshape(24,3).mean(1)
  mse=(err**2).mean(2).reshape(24,3,3).mean(1);errors[policy]=mse;costs[policy]=cost
  inside=(abs(err)<=np.array(scales[policy])[None,:,None]*1.644853626951*z['sd']).reshape(24,3,3,243).all((1,3))
  summary['policies'][policy]=dict(rmse_mm=(1000*np.sqrt(mse.mean(0))).tolist(),object_coverage=inside.mean(0).tolist(),covered_objects=inside.sum(0).tolist(),band_width_mm=(2000*np.array(scales[policy])*1.644853626951*z['sd'].mean((0,2))).tolist(),mean_cost=float(cost.mean()),object_mse_m2=mse.tolist(),stop_counts={str(n):sum(len(r['history'])==n for r in plans) for n in range(3)},view_count=sum(6 in r['history'] for r in plans),scales=scales[policy])
  coarse=np.load(BASE/f'test-{policy}-33.npz');summary['numerics'][policy]=dict(max_mean_change_mm=float(1000*abs(z['mean']-coarse['mean']).max()),rms_mean_change_mm=float(1000*np.sqrt(np.mean((z['mean']-coarse['mean'])**2))),max_sd_change_mm=float(1000*abs(z['sd']-coarse['sd']).max()),scope='Saved histories only; policy selected on 33^3 per layout, predictions and calibration on 65^3. Does not establish action convergence.')
 for p in ['fixed','random','adaptive_pokes']:
  delta=1000*(np.sqrt(errors[p][:,-1])-np.sqrt(errors['cost_aware'][:,-1]));objective=(errors[p][:,-1]+s['cost_penalty_m2']*costs[p])-(errors['cost_aware'][:,-1]+s['cost_penalty_m2']*costs['cost_aware'])
  summary['comparisons'][p]=dict(mean_object_rmse_improvement_mm=float(delta.mean()),paired_95_interval_mm=ci(delta),cost_adjusted_mse_improvement_mm2=float(1e6*objective.mean()),cost_adjusted_interval_mm2=ci(1e6*objective))
 if (BASE/'test-legacy-initial.npz').exists():
  z=np.load(BASE/'test-legacy-initial.npz');e=z['mean']-z['truth'];inside=(abs(e)<=float(z['scale'])*1.644853626951*z['sd']).reshape(24,3,243).all((1,2));summary['legacy_initial']=dict(rmse_mm=float(1000*np.sqrt(np.mean(e**2))),object_coverage=float(inside.mean()),covered_objects=int(inside.sum()),scope='Frozen v06 model and band multiplier on fresh objects; different development/calibration data than v07, so an end-to-end comparison, not an isolated covariance ablation.')
 for obj in range(6):
  for policy in ['repeat_initial','fixed','random','adaptive_pokes']:
   p=BASE/'phystwin'/f'object-{obj}-{policy}'
   if not (p/'summary-dt2.json').exists():continue
   a=json.loads((p/'summary-dt1.json').read_text());b=json.loads((p/'summary-dt2.json').read_text());summary['phystwin'].append(dict(object=obj,policy=policy,rmse_mm=a['results']['held_oblique']['hold_rmse_mm'],refined_rmse_mm=b['results']['held_oblique']['hold_rmse_mm'],training_frames=a['training_frames'],epochs=a['iterations'],visible_markers=a['visible_markers'],actions=a['fit_actions']))
 summary['phystwin_complete']=len(summary['phystwin'])==24
 write(OUT/'comparison.json',summary);write(OUT/'protocol.json',s)
 lines=['# Cost-aware views and pokes: v07','',s['scope'],'','## Held-out prediction','', '| Policy | Initial / one / two decision RMSE (mm) | Object coverage | Mean cost |','|---|---|---|---|']
 for p,r in summary['policies'].items():lines.append(f"| {p} | {' / '.join(f'{v:.3f}' for v in r['rmse_mm'])} | {' / '.join(f'{v:.0%}' for v in r['object_coverage'])} | {r['mean_cost']:.3f} |")
 lines+=['','Coverage requires all 243 coordinates and all three noise repeats for an object. Bands are calibrated separately by policy and decision budget using 24 independent calibration objects. A stopped policy retains its prediction at subsequent budgets. Test has 24 objects, so coverage estimates are noisy.','', '## Cost-aware choices','',json.dumps({k:summary['policies']['cost_aware'][k] for k in ['stop_counts','view_count','mean_cost']},indent=2),'','Paired stratified object-bootstrap comparisons (positive improvement favors cost-aware):','', '```json',json.dumps(summary['comparisons'],indent=2),'```','', '## Numerical limits','','Planning uses 33³ quadrature points per layout and 128 simulated outcomes. Final predictions use 65³. Refinement checks below concern the saved predictions, not the selected actions. They are reported rather than treated as evidence of convergence when unresolved.','','```json',json.dumps(summary['numerics'],indent=2),'```']
 if 'legacy_initial' in summary:lines+=['','## Frozen earlier uncertainty on the fresh cohort','','```json',json.dumps(summary['legacy_initial'],indent=2),'```']
 lines+=['','## Matched PhysTwin comparison','',f"Completed {len(summary['phystwin'])}/24 fits with timestep checks. Six fresh objects, four policies, identical 91-frame trajectories, 27 visible markers, 30 epochs and optimizer settings. The no-new-force control repeats the initial load three times. Adaptive actions use sparse Bayesian observations; all PhysTwin policies fit the same richer marker set. This compares action sequences within PhysTwin, not PhysTwin versus the Bayesian model. One optimizer seed; ramp targets interpolate static endpoints and are not validated dynamic truth.",'','| Object | Policy | Held-load RMSE mm | Half-step RMSE mm |','|---|---|---|---|']
 for r in summary['phystwin']:lines.append(f"| {r['object']} | {r['policy']} | {r['rmse_mm']:.3f} | {r['refined_rmse_mm']:.3f} |")
 lines+=['','## Real-object pilot','','Prepared in [REAL_PILOT.md](REAL_PILOT.md). Hardware acquisition has not been performed. Raw datasets and fitted checkpoints remain local.','']
 (OUT/'REPORT.md').write_text('\n'.join(lines));print(json.dumps({k:v for k,v in summary.items() if k in ['comparisons','legacy_initial','phystwin_complete']},indent=2))
if __name__=='__main__':main()
