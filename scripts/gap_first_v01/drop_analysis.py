"""Frozen image-observation and exact two-hypothesis decision diagnostic."""
from pathlib import Path
import hashlib,json,sys,cv2
import numpy as np
from scipy.special import logsumexp
from skimage.metrics import structural_similarity
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.gap_first_v01.quantiles import gaussian_mixture_quantiles
from scripts.gap_first_v01.drop_protocol import verify
BASE=ROOT/'runs/gap-first-v01/drop';REPORT=ROOT/'reports/gap-first-v01/drop'
def foreground_centroid(rgb):
    mask=np.max(rgb,axis=-1)>8
    if not np.any(mask):raise ValueError('No foreground at predefined threshold')
    return float(np.where(mask)[1].mean()/rgb.shape[1])
def trajectory(asset,case):
 p=BASE/'cases'/asset/case;status=json.loads((p/'result.json').read_text())
 if not status['complete'] or status['frames']!=30:raise RuntimeError('Incomplete case '+asset+'/'+case)
 d=np.load(p/'trajectory.npz')
 if not all(np.all(np.isfinite(d[k])) for k in d.files):raise RuntimeError('Nonfinite archive '+asset+'/'+case)
 return d
def evaluate(asset):
    if (REPORT/f'{asset}-analysis.json').exists() or (REPORT/f'{asset}-preobservation-bank.json').exists():raise RuntimeError('Analysis or bank alreadyexists; no silentoverwrite')
    a,b=[trajectory(asset,'passive-'+k) for k in 'AB'];pixel=[]
    for view in range(4):
      x,y=a['rgb'][:,view],b['rgb'][:,view];diff=x.astype(float)-y.astype(float);mse=float(np.mean(diff**2));pixel.append(dict(view=view,pixel_mse=mse,psnr_db=None if mse==0 else float(10*np.log10(255**2/mse)),exactly_equal=mse==0,max_pixel_difference=float(np.max(np.abs(diff))),changed_pixel_fraction=float(np.mean(np.any(diff!=0,axis=-1))),mean_ssim=float(np.mean([structural_similarity(xx,yy,channel_axis=-1,data_range=255) for xx,yy in zip(x,y)]))))
    calibration=np.array([foreground_centroid(trajectory(asset,'calibration-'+k)['rgb'][18,0])-foreground_centroid(trajectory(asset,'passive-'+k)['rgb'][18,0]) for k in 'AB'])
    targets=np.stack([trajectory(asset,'changed-'+k)['centroids'][18:25].reshape(-1) for k in 'AB']).astype(float)
    targets_delta=float(np.sqrt(np.mean((targets[0]-targets[1])**2)));prior_errors=np.sqrt(np.mean((targets-targets[0])**2,axis=1));bankrecord=dict(asset=asset,measurement_bank=calibration.tolist(),target_bank=targets.tolist(),measurement_target_bank_sha256=hashlib.sha256(calibration.tobytes()+targets.tobytes()).hexdigest(),prior=[.5,.5],prior_point_model=0,prior_risk=float(np.mean(prior_errors>.01)),created_before_noise_generation=True)
    (REPORT/f'{asset}-preobservation-bank.json').write_text(json.dumps(bankrecord,indent=2))
    rng=np.random.default_rng(90123);rows=[]
    # One predefined stream, both worlds retained. Observation noise is TOTAL delta noise.
    for world in range(2):
      for replicate in range(100):
       obs=float(calibration[world]+rng.normal(scale=.002));logweights=-.5*((obs-calibration)/.002)**2;weights=np.exp(logweights-logsumexp(logweights))
       for stage,w in [('before',np.array([.5,.5])),('after',weights)]:
        chosen=int(np.argmax(w));errors=np.sqrt(np.mean((targets-targets[chosen])**2,axis=1));risk=float(np.dot(w,errors>.01));real=float(errors[world]);lo,hi=gaussian_mixture_quantiles(w,targets,np.zeros_like(targets),(.05,.95));coverage=float(np.mean((targets[world]>=lo)&(targets[world]<=hi)))
        rows.append(dict(world=world,replicate=replicate,stage=stage,calibration_observation=obs,posterior_A=float(w[0]),selected_point_model=chosen,posterior_probability_rms_error_above_tolerance=risk,realized_rms_error=real,realized_failure=real>.01,brier=(risk-float(real>.01))**2,marginal90_coverage=coverage,mean_interval_width=float(np.mean(hi-lo)),point_refit_rms_error=real,point_refit_uses_same_evidence=True))
    summary=[]
    for stage in ['before','after']:
     r=[x for x in rows if x['stage']==stage];summary.append(dict(stage=stage,n=len(r),mean_realized_rms_error=float(np.mean([x['realized_rms_error'] for x in r])),mean_predicted_failure_probability=float(np.mean([x['posterior_probability_rms_error_above_tolerance'] for x in r])),empirical_failure_rate=float(np.mean([x['realized_failure'] for x in r])),brier=float(np.mean([x['brier'] for x in r])),marginal90_coverage=float(np.mean([x['marginal90_coverage'] for x in r])),mean_interval_width=float(np.mean([x['mean_interval_width'] for x in r]))))
    control=trajectory(asset,'known-mass');control_delta=float(np.sqrt(np.mean((a['controls']-control['controls'])**2)))
    doc=dict(asset=asset,passive_pixel_comparison=pixel,passive_control_rms_difference=float(np.sqrt(np.mean((a['controls']-b['controls'])**2))),passive_control_max_difference=float(np.max(np.abs(a['controls']-b['controls']))),known_mass_control_rms_difference=control_delta,calibration_means_normalized_image=calibration.tolist(),calibration_total_sigma=.002,changed_force_target_pair_rms_difference=targets_delta,tolerance_normalized_scene_units=.01,decision_summary=summary,noise_rows=rows,risk_metric='Scalar-coordinate RMS over21coordinates; not Euclidean pointRMS (differs bysqrt3).',noise_replicates_scope='100noise realizations perworld; only3physicalassetgeometries, not600 independentobjects.',limits=['Controlled native-mechanics two-world experiment, not nativepredictor calibration or failure proof.','Same correctiveevidence refits pointMAP baseline; identical selected predictions are expected. Uncertainty value is prospective risk, not an assumed pointaccuracy win.','No camera acquisition policy tested; four passiveviews quantify observational equivalence.','Native normalized scene units and per-anchor force; not calibrated real-object SI.'])
    (REPORT/f'{asset}-analysis.json').write_text(json.dumps(doc,indent=2,allow_nan=False));return doc
if __name__=='__main__':
 protocol=verify();assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==protocol['source_sha256'][str(Path(__file__).relative_to(ROOT))]
 for asset in protocol['assets']:
  if (REPORT/f'{asset}-analysis.json').exists():
   print(asset,'existinganalysis retained',flush=True);continue
  try:
   d=evaluate(asset);print(asset,d['passive_control_max_difference'],d['changed_force_target_pair_rms_difference'],d['decision_summary'],flush=True)
  except Exception as e:
   failure=dict(asset=asset,status='failed',error=repr(e),retained_in_declared_cohort=True);(REPORT/f'{asset}-analysis.json').write_text(json.dumps(failure,indent=2));print(failure,flush=True)
