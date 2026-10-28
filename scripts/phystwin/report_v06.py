"""Publish graded v06 evidence, uncertainty limitations, and PhysTwin bridge."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.phystwin.sequential_reference import BASE,ROOT,OUT,protocol
from scripts.phystwin.sequential_policy import verify
from scripts.phystwin.sequential_fixture import points

def clean(x):
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
 if isinstance(x,list):return [clean(v) for v in x]
 if isinstance(x,float) and not np.isfinite(x):return None
 return x

def write(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(clean(value),indent=2,allow_nan=False)+'\n')
def aligned_error(pred,true):
 X=pred-pred.mean(0);Y=true-true.mean(0);U,_,V=np.linalg.svd(X.T@Y);D=np.eye(3);D[-1,-1]=np.linalg.det(U@V);return float(np.mean((X@U@D@V-Y)**2))

def main():
 spec=protocol();OUT.mkdir(parents=True,exist_ok=True);write(OUT/'protocol.json',spec)
 rng=np.random.default_rng(916);groups=[np.array([i for i,r in enumerate(spec['test']) if r['layout']==l]) for l in spec['layouts']];ix=np.concatenate([rng.choice(g,(10000,len(g)),replace=True) for g in groups],axis=1)
 results={};rest=points()
 for mode,directory in [('direct',BASE/'analysis'),('rgbd',BASE/'analysis-rgbd')]:
  verify(json.loads((directory/'seal.json').read_text()));s=json.loads((directory/'test-summary.json').read_text());refined=json.loads((directory/'posthoc-refinement-97.json').read_text());comparisons={};physical={};shape={}
  for f in directory.glob('*.json'):write(OUT/mode/f.name,json.loads(f.read_text()))
  for policy in ['fixed','random','heuristic']:
   a=np.array(s['policies']['adaptive']['object_mse']);b=np.array(s['policies'][policy]['object_mse']);benefit=1000*(np.sqrt(b[ix].mean(1))-np.sqrt(a[ix].mean(1)));ra=np.array(refined['policies']['adaptive']['object_mse']);rb=np.array(refined['policies'][policy]['object_mse']);rbene=1000*(np.sqrt(rb[ix].mean(1))-np.sqrt(ra[ix].mean(1)))
   comparisons[policy]=dict(improvement_mm=(1000*(np.sqrt(b.mean(0))-np.sqrt(a.mean(0)))).tolist(),paired_object_bootstrap95_mm=np.quantile(benefit,[.025,.975],axis=0).T.tolist(),refined97_bootstrap95_mm=np.quantile(rbene,[.025,.975],axis=0).T.tolist())
  param=np.repeat(np.log([r['parameters'] for r in spec['test']]),spec['noise_repeats'],axis=0)
  for policy in ['fixed','random','heuristic','adaptive']:
   with np.load(directory/f'test-{policy}.npz') as z:
    q=z['quantiles'];physical[policy]=dict(coverage=((q[:,:,:,0]<=param[:,None])&(param[:,None]<=q[:,:,:,2])).mean(0).tolist(),log_median_absolute_error=abs(q[:,:,:,1]-param[:,None]).mean(0).tolist(),mean_log_interval_width=(q[:,:,:,2]-q[:,:,:,0]).mean(0).tolist())
    errors=np.array([[aligned_error(rest+m.reshape(81,3),rest+t.reshape(81,3)) for m in row] for row,t in zip(z['mean'],z['truth'])]);shape[policy]=(1000*np.sqrt(errors.mean(0))).tolist()
  stable=s['stability']['adaptive'];results[mode]=dict(summary=s,comparisons=comparisons,physical=physical,aligned_rmse_mm=shape,utility_agreement=sum(r['agreement'] for r in stable),utility_trials=len(stable),refinement97=refined)
 write(OUT/'comparison.json',results)
 qualification=[json.loads(p.read_text()) for p in sorted((BASE/'qualification').glob('*.json'))];assert len(qualification)==36 and all(q['displacement_qualified'] for q in qualification);write(OUT/'reference-qualification.json',qualification)
 write(OUT/'vision-summary.json',json.loads((BASE/'vision/summary.json').read_text()))
 motion=[]
 for i in range(len(spec['test'])):
  with np.load(BASE/'fields'/f'test-{i}-reference.npz') as z:fields=z['field']
  nonrigid=[float(np.sqrt(aligned_error(rest+f,rest))) for f in fields[1:7]];motion.append(dict(object=i,probe_nonrigid_rms_mm=(1000*np.array(nonrigid)).tolist()))
 write(OUT/'deformation.json',motion)
 bridge=[]
 for obj in range(3):
  row=dict(object=obj)
  for budget in ['initial','two_probes']:
   folder=BASE/'phystwin'/f'object-{obj}-{budget}';s=json.loads((folder/'summary-dt1.json').read_text());s2=json.loads((folder/'summary-dt2.json').read_text());one=np.load(folder/'held_oblique-dt1.npz');two=np.load(folder/'held_oblique-dt2.npz');hold=one['hold'];delta=float(1000*np.sqrt(np.mean((one['predicted'][hold]-two['predicted'][hold])**2)));row[budget]=dict(summary=s,timestep_hold_rms_change_mm=delta,refined_summary=s2)
  bridge.append(row)
 write(OUT/'phystwin-bridge.json',bridge)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
 for row,mode in enumerate(['direct','rgbd']):
  for policy in ['fixed','random','heuristic','adaptive']:
   s=results[mode]['summary']['policies'][policy];axes[row,0].plot(range(3),s['rmse_mm'],'o-',label=policy);axes[row,1].plot(range(3),100*np.array(s['object_coverage']),'o-',label=policy)
  axes[row,0].set(title=f'{mode.upper()}: held-load prediction',ylabel='Coordinate RMSE (mm)',xlabel='Acquired probes',xticks=[0,1,2]);axes[row,1].set(title=f'{mode.upper()}: whole-object band coverage',ylabel='Coverage (%)',xlabel='Acquired probes',xticks=[0,1,2],ylim=(0,105));axes[row,1].axhline(90,color='grey',ls='--')
  for ax in axes[row]:ax.grid(alpha=.2);ax.legend()
 fig.savefig(OUT/'sequential-comparison.png',dpi=150);plt.close(fig)
 lines=['# Multi-support sequential probing and visual PhysTwin bridge — v06','','The useful result is earlier information from RGB-D: the first adaptive probe improves held-load prediction over fixed, random and response-based choices. Two probes substantially improve prediction, but adaptive selection is not uniformly best after the second probe. This is graded pilot evidence, not a binary pass/fail verdict.','','## What changed','',
  'A 240 × 80 × 80 mm nonlinear block has compliant patches on both ends. Material stiffness and the two support stiffnesses vary independently and continuously; three categories move the support patches between lower, upper and opposing lateral halves. Six competing probes include localized pushes, opposite-direction twisting loads and axial stretch. A separate oblique load is held out for prediction. Known weak gravity contributes 0.02 N total (equivalent to 0.0208 m/s² for the 0.96 kg block). Each probe is a reset experiment, and the second choice conditions on the first measured response.', '',
  'The split is 6 development, 12 calibration and 18 held-out objects, with three observation repetitions per object. Fixed sequences use development data only. Random and adaptive policies receive the same two-probe budget; the heuristic chooses the largest posterior-predicted marker motion. No true support label enters a policy. Bootstrap resampling is stratified by layout and clustered by object.','',
  '## Held-out prediction','', '| Observations | Policy | Before probes (mm) | After one (mm) | After two (mm) |','|---|---|---:|---:|---:|']
 for mode,r in results.items():
  for policy,s in r['summary']['policies'].items():lines.append(f"| {mode} | {policy} | "+' | '.join(f'{v:.3f}' for v in s['rmse_mm'])+' |')
 lines+=['','## What benefit is supported?','','Positive differences favor adaptive probing. These are paired object-bootstrap 95% intervals in millimetres; the two budgets are reported separately.','', '| Observations | Baseline | First-probe improvement interval | Two-probe improvement interval |','|---|---|---|---|']
 for mode,r in results.items():
  for policy,c in r['comparisons'].items():
   ci=c['paired_object_bootstrap95_mm'];lines.append(f"| {mode} | {policy} | [{ci[1][0]:.3f}, {ci[1][1]:.3f}] | [{ci[2][0]:.3f}, {ci[2][1]:.3f}] |")
 lines+=['','The first RGB-D probe has a supported benefit against all three baselines in this cohort. After two probes, adaptive beats random, while fixed and heuristic differences remain inconclusive. Direct-point observations also benefit from probing, but the fixed sequence has the best mean two-probe score. These are exploratory comparisons across two observation models and budgets, not multiplicity-adjusted confirmatory claims.','',
  '## Uncertainty and numerical resolution','','| Observations | Adaptive coverage: 0 / 1 / 2 probes | Sampling-choice agreement | Finest primary grid |','|---|---|---|---:|']
 for mode,r in results.items():
  s=r['summary'];coverage=' / '.join(f'{100*c:.1f}%' for c in s['policies']['adaptive']['object_coverage']);lines.append(f"| {mode} | {coverage} | {r['utility_agreement']}/{r['utility_trials']} | {s['grid_size']}³ per layout |")
 lines+=['','Prediction bands are calibrated on 12 separate objects, pooling policies and budgets conservatively. The RGB-D initial and one-probe bands under-cover this test cohort; two-probe coverage improves. Thus we do not claim uniformly calibrated uncertainty. Raw physical-parameter intervals are reported separately in [comparison.json](comparison.json), and prediction-band calibration does not calibrate them.','',
  'The primary 33→65 grid comparison did not meet every diagnostic tolerance (10 μm mean/SD and 0.05 log-parameter quantiles). A post-evaluation 97³-per-layout audit holds every saved action and calibration multiplier fixed. All saved prediction histories then meet those tolerances: the largest mean change across policies is 7.4 μm, SD change 4.2 μm, and log-quantile change 0.0171. This supports numerical stability of the reported predictions; it does not establish action-choice stability at the finer grid.','', '| Observations | Adaptive maximum mean change, 65→97 (mm) | RMS mean change (mm) | Refined RMSE after two probes (mm) |','|---|---:|---:|---:|']
 for mode,r in results.items():
  a=r['refinement97']['policies']['adaptive'];lines.append(f"| {mode} | {a['maximum_mean_change_mm']:.5f} | {a['rms_mean_change_mm']:.5f} | {a['rmse_mm'][2]:.3f} |")
 lines+=['','All 36 references passed sampled-displacement spatial/quadrature checks, equilibrium residuals and positive-Jacobian checks. Thirty used mesh level 6 and six used level 8. These are displacement-qualified references, not the stricter v05 traction/strain qualification and not experimental validation. The new boundary energy gradient and tangent were checked independently. The fast model’s development target RMS errors range from 0.003 to 0.035 mm; six development objects are a limited discrepancy-training sample.','',
  '## Visual observations and PhysTwin','','The RGB-D bridge rasterizes perspective surface depth and colored fiducials at 640×480. It measures marker centroids and backprojects noisy depth; measured coordinate error is approximately 0.26 mm. Cameras, rest geometry and marker identities are known. This is a controlled vision interface, not raw-video tracking, COLMAP reconstruction, Gaussian appearance fitting or a foundation-model evaluation.','',
  'The official PhysTwin dense fitter is run on three bridge cases within the held cohort, one per support layout. An isolated opt-in adapter supplies known nodal forces; candidate anchors span both end faces, without supplying true patch labels or stiffnesses. It receives 27 visible markers per frame, whereas the sparse Bayesian policy gets three initial markers and one per probe. Consequently this is a richer-observation integration test, not a budget-matched PhysTwin-versus-FEM ranking.','',
  '| Object | Initial-only PhysTwin held error (mm) | After two probe interactions (mm) | Stationary reference error (mm) | Maximum timestep-refinement change (mm) |','|---:|---:|---:|---:|---:|']
 for r in bridge:
  a=r['initial']['summary']['results']['held_oblique'];b=r['two_probes']['summary']['results']['held_oblique'];dt=max(r[k]['timestep_hold_rms_change_mm'] for k in ['initial','two_probes']);lines.append(f"| {r['object']} | {a['hold_rmse_mm']:.3f} | {b['hold_rmse_mm']:.3f} | {b['stationary_rmse_mm']:.3f} | {dt:.5f} |")
 lines+=['','Both fits use 30 epochs, so the three-interaction fit also gets more frame-level optimizer updates. Improvement cannot be attributed solely to action selection. Checkpoints are selected using training loss; the oblique load never enters fitting. Ramp targets interpolate static equilibria; final-hold comparisons and drift are reported, but the video is not validated dynamic ground truth. PhysTwin uncertainty is not learned here; the Bayesian posterior remains a separate physics reference.','',
  '## Diagnosis of the old six weak probes','','Five of six nearly rigid v05 probes followed a mistaken maximum-posterior support-side estimate. The true-side probabilities were roughly 29–58%, so ambiguity, not just numerical instability or high-confidence certainty, matters. One probe with negligible strain still gave a favorable realized noisy prediction; lack of deformation alone is not a sufficient failure label. [Saved-action diagnosis](v05-action-diagnosis.json).','',
  '## Reproduction and artifacts','','Raw FEM fields, images, depth, checkpoints and training inputs remain under `runs/phystwin-sequential-support-v06`. Code and derived summaries/videos are published. See [the implementation instructions](../../scripts/phystwin/README.md). The direct and RGB-D seals hash numerical inputs before held-out scoring. The source adapter and force-kernel check are published, and the original PhysTwin checkout remains unchanged.','',
  '![Sequential comparison](sequential-comparison.png)','',
  '[View Bayesian actual-versus-predicted shapes](actual-vs-predicted.mp4). [View the PhysTwin bridge](phystwin-held-force.mp4). Selection and visualization scope are recorded in [video-selection.json](video-selection.json).','']
 (OUT/'REPORT.md').write_text('\n'.join(lines));print('REPORT_COMPLETE',OUT,flush=True)
if __name__=='__main__':main()
