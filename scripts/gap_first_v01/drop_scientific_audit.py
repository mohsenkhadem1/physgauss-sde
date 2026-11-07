import json,hashlib,datetime
from pathlib import Path
import numpy as np
from scipy.special import softmax
R=Path('reports/gap-first-v01/drop');U=Path('runs/gap-first-v01/drop');p=json.loads((R/'protocol.json').read_text());ext=R/'native-reference-extension';checks=0;summary=[]
def eq(a,b):
 global checks
 assert np.allclose(a,b,rtol=1e-11,atol=1e-12),(a,b)
 checks+=1
for asset in p['assets']:
 d=json.loads((R/f'{asset}-analysis.json').read_text());bank=json.loads((R/f'{asset}-preobservation-bank.json').read_text());q=np.array(bank['target_bank']);m=np.array(bank['measurement_bank']);gap=np.sqrt(np.mean((q[0]-q[1])**2));eq(gap,d['changed_force_target_pair_rms_difference'])
 for i,k in enumerate('AB'):
  with np.load(U/'cases'/asset/f'changed-{k}'/'trajectory.npz') as z: eq(q[i],z['centroids'][18:25].reshape(-1))
 for r in d['noise_rows']:
  w=np.array([.5,.5]) if r['stage']=='before' else softmax(-.5*((r['calibration_observation']-m)/.002)**2)
  eq(w[0],r['posterior_A']);j=int(w.argmax());errors=np.sqrt(np.mean((q-q[j])**2,axis=1));eq(w@(errors>.01),r['posterior_probability_rms_error_above_tolerance']);eq(errors[r['world']],r['realized_rms_error']);eq(r['realized_rms_error'],r['point_refit_rms_error'])
 er=json.loads((ext/f'{asset}-result.json').read_text());pr=json.loads((ext/f'{asset}-preoutcome-risk.json').read_text());qh=np.array(pr['qhat']);errors=np.sqrt(np.mean((q-qh)**2,axis=1));eq(errors,er['candidate_errors']);eq(np.mean(errors>.01),pr['prior_risk'])
 with np.load(U/'native-reference-extension'/asset/'changed.npz') as z:eq(qh,z['centroids'][18:25].reshape(-1))
 for rr,r in zip(er['rows'],d['noise_rows']):
  w=np.array([r['posterior_A'],1-r['posterior_A']]);risk=w@(errors>.01);eq(rr['native_predicted_failure_probability'],risk);eq(rr['native_realized_coordinate_rms'],errors[r['world']]);eq(rr['native_brier'],(risk-float(errors[r['world']]>.01))**2);eq(rr['corrected_MAP_error'],r['point_refit_rms_error'])
 summary.append({'asset':asset,'force_coordinate_rms_gap':gap,'prior_MAP_risk':bank['prior_risk'],'named_native_prior_risk':pr['prior_risk'],'named_native_mean_error':float(errors.mean()),'MAP_before_mean_error':gap/2,'MAP_after_mean_error':float(np.mean([r['realized_rms_error'] for r in d['noise_rows'] if r['stage']=='after']))})
manifest=json.loads((ext/'protocol.json').read_text());assert hashlib.sha256(Path('scripts/gap_first_v01/drop_native_extension.py').read_bytes()).hexdigest()==manifest['source_sha256']
paths=['papers/cvpr2027-joint-identification/sections/gap-first-drop.tex','scripts/gap_first_v01/drop_native_extension.py','scripts/gap_first_v01/drop_analysis.py','scripts/gap_first_v01/drop_predict.py','runs/gap-first-v01/drop/vendor/ReconPhys/release_utils.py','reports/gap-first-v01/drop/protocol.json','reports/gap-first-v01/drop/native-reference-extension/protocol.json']
out={'passed':True,'reviewed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'independent_numeric_checks':checks,'summary':summary,'dataflow_review':['Native loader returns [3,T,H,W] in sequential frame order; [:,:,:20] on batched tensor restricts network to passive frames0..19.','Known force is a separate reset rollout; overlapping frame numbers do not expose held-force observations to predictor.','Named native parameter output is frozen before replay and never updated after calibration.','Reference target bank is prospective two-world model prediction; indexed true-world evaluation happens only after qhat/risk are saved.','Primary MAP risk .5 differs from named NN risk1 because they assess different fixed point predictions.','All600 noise trials come from three geometry/two-world combinations; no independent-object performance claim.'],'scope':['Coordinate RMS over21scalar coordinates, not point-Euclidean RMS; normalized scene coordinates, not mm.','Original GT-parameter replay is imperfect; no clean end-to-end accuracy superiority claim.','Runtime values are local wall-clock observations including synchronization/rendering/setup; not controlled speed comparison.','Native future image score is view0 pooled-MSE PSNR, whereas released-video control aggregates per-frame PSNR across4views; describe different protocols.'],'source_sha256':{f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in paths}}
(R/'independent-scientific-final-review.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
