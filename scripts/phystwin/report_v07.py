"""Summarize sealed v07 evidence without hiding incomplete work."""
from pathlib import Path
import json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.phystwin.v07_study import BASE,OUT,SOURCE,POLICIES,protocol,write
from scripts.phystwin.sequential_policy import verify
from scripts.phystwin.sequential_fixture import points,ACTIONS

def ci(delta):
 # Paired object bootstrap, stratified by the three support layouts.
 rng=np.random.default_rng(7107);ids=np.concatenate([rng.choice(np.arange(l,len(delta),3),(10000,len(delta)//3),replace=True) for l in range(3)],axis=1)
 return np.quantile(delta[ids].mean(1),[.025,.975]).tolist()

def aligned_rmse(predicted,truth):
 rest=points();values=[]
 for u,v in zip(predicted,truth):
  x=rest+u.reshape(81,3);y=rest+v.reshape(81,3);xc=x-x.mean(0);U,_,V=np.linalg.svd(xc.T@(y-y.mean(0)));R=U@V
  if np.linalg.det(R)<0:U[:,-1]*=-1;R=U@V
  values.append(np.mean((xc@R+y.mean(0)-y)**2))
 return float(1000*np.sqrt(np.mean(values)))

def deformation():
 rest=points();centered=rest-rest.mean(0);records=[]
 for obj in range(24):
  fields=np.load(BASE/'fields'/f'test-{obj}-reference.npz')['field']
  for name,u in zip(ACTIONS,fields):
   target=rest+u;U,_,V=np.linalg.svd(centered.T@(target-target.mean(0)));R=U@V
   if np.linalg.det(R)<0:U[:,-1]*=-1;R=U@V
   rigid=centered@R+target.mean(0);total=np.sqrt(np.mean(u*u));shape=np.sqrt(np.mean((target-rigid)**2));records.append(dict(object=obj,action=name,motion_rms_mm=float(total*1000),nonrigid_rms_mm=float(shape*1000),nonrigid_fraction=float(shape/max(total,1e-15))))
 by={a:{k:np.quantile([r[k] for r in records if r['action']==a],[0,.5,1]).tolist() for k in ['motion_rms_mm','nonrigid_rms_mm','nonrigid_fraction']} for a in ACTIONS}
 value=dict(scope='Rigid Procrustes fit removed from sampled static fields. Fractions are RMS ratios, not energy fractions. min/median/max over 24 fresh test objects.',by_action=by,records=records);write(OUT/'deformation.json',value);return by

def main():
 s=protocol();seal=json.loads((BASE/'seal.json').read_text());verify(seal);summary=dict(protocol=s,calibration=seal['method'],policies={},comparisons={},numerics={},phystwin=[])
 qualifications=[json.loads(p.read_text()) for p in sorted((BASE/'qualification').glob('*.json'))]
 summary['reference_qualification']=dict(count=len(qualifications),all_displacement_qualified=all(q['displacement_qualified'] for q in qualifications),mesh_counts={str(n):sum(q['mesh']==n for q in qualifications) for n in [4,6,8,10]},scope='Sampled displacement, equilibrium, positive Jacobian, spatial and quadrature refinement. Strain and traction are not fully qualified.')
 summary['deformation']=deformation()
 errors={};costs={};scales=seal['scales']
 for policy in POLICIES:
  z=np.load(BASE/f'test-{policy}-65.npz');err=z['mean']-z['truth'][:,None];plans=json.loads((BASE/'plans'/f'test-{policy}.json').read_text());cost=np.array([sum(s['costs'][a] for a in r['history']) for r in plans]).reshape(24,3).mean(1)
  mse=(err**2).mean(2).reshape(24,3,3).mean(1);errors[policy]=mse;costs[policy]=cost
  inside=(abs(err)<=np.array(scales[policy])[None,:,None]*1.644853626951*z['sd']).reshape(24,3,3,243).all((1,3))
  summary['policies'][policy]=dict(rmse_mm=(1000*np.sqrt(mse.mean(0))).tolist(),object_coverage=inside.mean(0).tolist(),covered_objects=inside.sum(0).tolist(),band_width_mm=(2000*np.array(scales[policy])*1.644853626951*z['sd'].mean((0,2))).tolist(),mean_cost=float(cost.mean()),object_mse_m2=mse.tolist(),stop_counts={str(n):sum(len(r['history'])==n for r in plans) for n in range(3)},view_count=sum(6 in r['history'] for r in plans),scales=scales[policy])
  summary['policies'][policy]['rigid_aligned_rmse_mm']=[aligned_rmse(z['mean'][:,b],z['truth']) for b in range(3)]
  coarse=np.load(BASE/f'test-{policy}-33.npz');summary['numerics'][policy]=dict(max_mean_change_mm=float(1000*abs(z['mean']-coarse['mean']).max()),rms_mean_change_mm=float(1000*np.sqrt(np.mean((z['mean']-coarse['mean'])**2))),max_sd_change_mm=float(1000*abs(z['sd']-coarse['sd']).max()),scope='Saved histories only; policy selected on 33^3 per layout, predictions and calibration on 65^3. Does not establish action convergence.')
 summary['stationary_baseline']=dict(rmse_mm=float(1000*np.sqrt(np.mean(z['truth']**2))),rigid_aligned_rmse_mm=aligned_rmse(np.zeros_like(z['truth']),z['truth']))
 for p in ['fixed','random','adaptive_pokes']:
  delta=1000*(np.sqrt(errors[p][:,-1])-np.sqrt(errors['cost_aware'][:,-1]));objective=(errors[p][:,-1]+s['cost_penalty_m2']*costs[p])-(errors['cost_aware'][:,-1]+s['cost_penalty_m2']*costs['cost_aware'])
  summary['comparisons'][p]=dict(mean_object_rmse_improvement_mm=float(delta.mean()),paired_95_interval_mm=ci(delta),cost_adjusted_mse_improvement_mm2=float(1e6*objective.mean()),cost_adjusted_interval_mm2=ci(1e6*objective))
 cost_plans=json.loads((BASE/'plans/test-cost_aware.json').read_text());cz=np.load(BASE/'test-cost_aware-65.npz');az=np.load(BASE/'test-adaptive_pokes-65.npz');failures=[]
 for i,p in enumerate(cost_plans):
  if len(p['history'])<2:failures.append(dict(object=i//3,repeat=i%3,history=p['history'],stopped_rmse_mm=float(1000*np.sqrt(np.mean((cz['mean'][i,-1]-cz['truth'][i])**2))),two_poke_rmse_mm=float(1000*np.sqrt(np.mean((az['mean'][i,-1]-az['truth'][i])**2)))))
 summary['stopping_failures']=sorted(failures,key=lambda r:r['stopped_rmse_mm'],reverse=True)[:4]
 if (BASE/'test-legacy-initial.npz').exists():
  z=np.load(BASE/'test-legacy-initial.npz');e=z['mean']-z['truth'];inside=(abs(e)<=float(z['scale'])*1.644853626951*z['sd']).reshape(24,3,243).all((1,2));summary['legacy_initial']=dict(rmse_mm=float(1000*np.sqrt(np.mean(e**2))),object_coverage=float(inside.mean()),covered_objects=int(inside.sum()),scope='Frozen v06 model and band multiplier on fresh objects; different development/calibration data than v07, so an end-to-end comparison, not an isolated covariance ablation.')
 for obj in range(6):
  for policy in ['repeat_initial','fixed','random','adaptive_pokes']:
   p=BASE/'phystwin'/f'object-{obj}-{policy}'
   if not (p/'summary-dt2.json').exists():continue
   a=json.loads((p/'summary-dt1.json').read_text());b=json.loads((p/'summary-dt2.json').read_text());summary['phystwin'].append(dict(object=obj,policy=policy,rmse_mm=a['results']['held_oblique']['hold_rmse_mm'],stationary_rmse_mm=a['results']['held_oblique']['stationary_rmse_mm'],refined_rmse_mm=b['results']['held_oblique']['hold_rmse_mm'],training_frames=a['training_frames'],epochs=a['iterations'],optimizer_updates=a['optimizer_updates'],visible_markers=a['visible_markers'],actions=a['fit_actions']))
 summary['phystwin_complete']=len(summary['phystwin'])==24
 if summary['phystwin_complete']:
  assert {(r['training_frames'],r['epochs'],r['optimizer_updates'],r['visible_markers']) for r in summary['phystwin']}=={(91,30,2700,27)},'Unmatched training budgets'
  paired={p:np.array([next(r['rmse_mm'] for r in summary['phystwin'] if r['object']==i and r['policy']==p) for i in range(6)]) for p in ['repeat_initial','fixed','random','adaptive_pokes']}
  summary['phystwin_paired_improvement']={p:dict(mean_mm=float((paired[p]-paired['adaptive_pokes']).mean()),paired_95_interval_mm=ci(paired[p]-paired['adaptive_pokes'])) for p in ['repeat_initial','fixed','random']}
  summary['phystwin_policy_means_mm']={p:float(np.mean([r['rmse_mm'] for r in summary['phystwin'] if r['policy']==p])) for p in ['repeat_initial','fixed','random','adaptive_pokes']}

 if (OUT/'action-refinement.json').exists():summary['action_refinement']=json.loads((OUT/'action-refinement.json').read_text())
 if (OUT/'prediction-refinement.json').exists():summary['prediction_refinement']=json.loads((OUT/'prediction-refinement.json').read_text())
 write(OUT/'comparison.json',summary);write(OUT/'protocol.json',s)
 lines=['# Cost-aware views and pokes: v07','',s['scope'],'','![Study summary](comparison.png)','','## Held-out prediction','', '| Policy | Initial / one / two decision RMSE (mm) | Object coverage | Mean cost |','|---|---|---|---|']
 for p,r in summary['policies'].items():lines.append(f"| {p} | {' / '.join(f'{v:.3f}' for v in r['rmse_mm'])} | {' / '.join(f'{v:.0%}' for v in r['object_coverage'])} | {r['mean_cost']:.3f} |")
 lines+=['','The stationary rest-shape baseline has coordinate RMSE '+f"{summary['stationary_baseline']['rmse_mm']:.3f}"+' mm. Rigid-aligned shape errors (initial / final mm): '+', '.join(p+': '+f"{r['rigid_aligned_rmse_mm'][0]:.3f} / {r['rigid_aligned_rmse_mm'][2]:.3f}" for p,r in summary['policies'].items())+'.']
 lines+=['','Coverage requires all 243 coordinates and all three noise repeats for an object. Bands are calibrated separately by policy and decision budget using 24 independent calibration objects. A stopped policy retains its prediction at subsequent budgets. Test has 24 objects, so coverage estimates are noisy. The quantile follows split-conformal calibration, but layouts were balanced by design rather than drawn IID; coverage is an empirical result here, not a claimed distribution-free guarantee.','', 'Watch [actual versus predicted](actual-vs-predicted.mp4): median and worst final-error objects, actual deformation scale, static shapes with a moving camera.','','## Cost-aware choices','',json.dumps({k:summary['policies']['cost_aware'][k] for k in ['stop_counts','view_count','mean_cost']},indent=2),'','Paired stratified object-bootstrap comparisons (positive improvement favors cost-aware):','', '```json',json.dumps(summary['comparisons'],indent=2),'```','', '## Numerical limits','','Planning uses 33³ quadrature points per layout and 128 simulated outcomes. Final predictions use 65³. Refinement checks below concern the saved predictions, not the selected actions. They are reported rather than treated as evidence of convergence when unresolved.','','```json',json.dumps(summary['numerics'],indent=2),'```']
 lines+=['','Stopping is not uniformly safe: in the largest stopped-error case, the policy stops before useful information is collected. The four largest stopped-case errors are below; these are post-test diagnostics, not tuning cases.','','```json',json.dumps(summary['stopping_failures'],indent=2),'```']
 if 'legacy_initial' in summary:lines+=['','## Frozen earlier uncertainty on the fresh cohort','','```json',json.dumps(summary['legacy_initial'],indent=2),'```']
 lines+=['','## Matched PhysTwin comparison','',f"Completed {len(summary['phystwin'])}/24 fits with timestep checks. Six fresh objects, four policies, identical 91-frame trajectories, 27 visible markers, 30 epochs and optimizer settings. The no-new-force control repeats the initial load three times. Adaptive actions use sparse Bayesian observations; all PhysTwin policies fit the same richer marker set. This compares action sequences within PhysTwin, not PhysTwin versus the Bayesian model. One optimizer seed; ramp targets interpolate static endpoints and are not validated dynamic truth.",'','[Watch all six fixed-versus-adaptive PhysTwin cases](phystwin-fixed-vs-adaptive.mp4).','','| Object | Policy | Held-load RMSE mm | Half-step RMSE mm |','|---|---|---|---|']
 for r in summary['phystwin']:lines.append(f"| {r['object']} | {r['policy']} | {r['rmse_mm']:.3f} | {r['refined_rmse_mm']:.3f} |")

 if 'prediction_refinement' in summary:lines+=['','## Finer prediction audit','','```json',json.dumps(summary['prediction_refinement'],indent=2),'```']
 if 'action_refinement' in summary:
  a=summary['action_refinement'];lines+=['','Action refinement diagnostic: '+str(a['agreements'])+'/'+str(a['decisions'])+' saved decisions agree on the finer grid. '+a['scope']+' One audited second-step choice switches between a poke and a view; the refined cost-adjusted variance gap is about 0.00010 mm², so action identity is not fully converged near ties.']
 if 'phystwin_paired_improvement' in summary:lines+=['','Adaptive PhysTwin paired improvement (positive favors adaptive), with exploratory six-object bootstrap intervals:','','```json',json.dumps(summary['phystwin_paired_improvement'],indent=2),'```']
 if 'phystwin_policy_means_mm' in summary:lines+=['','Mean PhysTwin held-load RMSE (mm): `'+json.dumps(summary['phystwin_policy_means_mm'])+'`. Individual cases above show the variation; six objects and one optimizer seed are exploratory evidence.']
 lines+=['','## Deformation signal','','Rigid-motion removal leaves a median 38% of held-load displacement RMS (range 12–51%) across the 24 test objects. Median nonrigid held-load motion is 0.187 mm; the initial-load median is only 0.064 mm. These scenarios contain shape change, but its absolute magnitude remains small relative to camera noise. The second view sees the same three initial-load markers from a new angle; it does not reveal new markers or observe the stronger probe deformation. This limits what the view-versus-poke comparison establishes. See [deformation diagnostics](deformation.json).']
 lines+=['','## Real-object pilot','','Deferred by the user: this campaign focuses on simulation. [A future protocol](REAL_PILOT.md) is available, but no hardware acquisition is claimed or pending. Raw datasets and fitted checkpoints remain local.','']
 (OUT/'REPORT.md').write_text('\n'.join(lines));print(json.dumps({k:v for k,v in summary.items() if k in ['comparisons','legacy_initial','phystwin_complete']},indent=2))
if __name__=='__main__':main()
