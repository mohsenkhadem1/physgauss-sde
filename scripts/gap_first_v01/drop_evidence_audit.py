"""Independent final artifact/numeric audit; success never requires an advantage."""
from pathlib import Path
import sys,json,hashlib,datetime,pickle
import numpy as np
from scipy.special import logsumexp
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));from scripts.gap_first_v01.drop_protocol import verify
BASE=ROOT/'runs/gap-first-v01/drop';REPORT=ROOT/'reports/gap-first-v01/drop'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 protocol=verify();issues=[];checks=0;cases=[];analysis_summaries=[]
 def check(ok,label):
  nonlocal checks
  checks+=1
  if not ok:issues.append(label)
 camera=np.load(BASE/'initial/cameras.npz');expected=np.array([[0,2,0],[2,0,0],[0,-2,0],[-2,0,0]])
 check(np.allclose(np.linalg.inv(camera['extrinsics'])[:,:3,3],expected,atol=1e-5,rtol=0),'camera_order_positions')
 for asset in protocol['assets']:
  states={};firsthash=None
  for case in protocol['cases']:
   folder=BASE/'cases'/asset/case;d=json.loads((folder/'result.json').read_text());check(d['complete'] and d['frames']==30,asset+'/'+case+'/complete');check(d['source_sha256']==protocol['source_sha256']['scripts/gap_first_v01/drop_run.py'],asset+'/'+case+'/source');check(d['initial_sha256']==sha(BASE/'initial'/f'{asset}.pt'),asset+'/'+case+'/anchors')
   if case=='native-predicted':expected_params=json.loads((REPORT/f'{asset}-native-predictor.json').read_text())['params']
   elif case=='native-groundtruth':expected_params={k:v for k,v in pickle.load(open(BASE/'assets'/asset/'sample_00000/physics_params.pkl','rb')).items() if k in ['m','k','damp','fric_k']}
   elif case=='known-mass':expected_params=protocol['known_mass_control']
   else:expected_params=protocol['per_anchor_parameters'][case[-1]]
   check(d['params']==expected_params,asset+'/'+case+'/parameters')
   expected_force=([4.,0.,0.] if case.startswith('calibration') else ([0.,0.,20.] if case.startswith('changed') else None))
   check(d['force_per_anchor']==expected_force and d['force_frames']==([12,17] if expected_force else None),asset+'/'+case+'/force')
   if not d['complete']:continue
   x=np.load(folder/'trajectory.npz');states[case]=x;check(x['rgb'].shape==(30,4,512,512,3),asset+'/'+case+'/rgbshape');check(x['controls'].shape==(30,2048,3),asset+'/'+case+'/controlshape');check(all(np.all(np.isfinite(x[k])) for k in x.files),asset+'/'+case+'/finite');check(np.min(x['controls'][1:,:,2])>=-.200001,asset+'/'+case+'/floor');check(np.array_equal(x['controls'][0],x['initial_controls']),asset+'/'+case+'/initialcontrols')
   h=sha(folder/'gaussians-000.npy');firsthash=h if firsthash is None else firsthash;check(h==firsthash,asset+'/'+case+'/initialgaussians')
   cases.append(dict(asset=asset,case=case,complete=d['complete'],trajectory_sha256=sha(folder/'trajectory.npz')))
  for branch in ['calibration','changed']:
   for world in 'AB':check(np.array_equal(states[branch+'-'+world]['rgb'][:12],states['passive-'+world]['rgb'][:12]),asset+'/'+branch+'/'+world+'/before_force_identical')
  doc=json.loads((REPORT/f'{asset}-analysis.json').read_text());check(doc.get('status')!='failed',asset+'/analysiscomplete');bank=json.loads((REPORT/f'{asset}-preobservation-bank.json').read_text());q=np.stack([states['changed-'+w]['centroids'][18:25].reshape(-1) for w in 'AB']).astype(float)
  def center(rgb):return np.where(np.max(rgb,axis=-1)>8)[1].mean()/512
  mu=np.array([center(states['calibration-'+w]['rgb'][18,0])-center(states['passive-'+w]['rgb'][18,0]) for w in 'AB']);check(bank['measurement_target_bank_sha256']==hashlib.sha256(mu.tobytes()+q.tobytes()).hexdigest(),asset+'/preobservationbank');check(np.allclose(mu,doc['calibration_means_normalized_image'],atol=1e-14,rtol=0),asset+'/measurement')
  rows={(r['world'],r['replicate'],r['stage']):r for r in doc['noise_rows']};check(len(rows)==400,asset+'/noiserowcount');rng=np.random.default_rng(90123);maximum_error=0.
  for world in range(2):
   for rep in range(100):
    obs=mu[world]+rng.normal(scale=.002);lw=-.5*((obs-mu)/.002)**2;posterior=np.exp(lw-logsumexp(lw))
    for stage,w in [('before',np.array([.5,.5])),('after',posterior)]:
     row=rows[(world,rep,stage)];selected=int(np.argmax(w));error=np.sqrt(np.mean((q-q[selected])**2,axis=1));risk=float(w@(error>.01));maximum_error=max(maximum_error,abs(row['calibration_observation']-obs),abs(row['posterior_A']-w[0]),abs(row['posterior_probability_rms_error_above_tolerance']-risk),abs(row['realized_rms_error']-error[world]));check(row['selected_point_model']==selected and row['point_refit_rms_error']==row['realized_rms_error'],asset+'/same_evidence_MAP')
  check(maximum_error<1e-12,asset+'/independent_noise_risk_recomputation');analysis_summaries.append(dict(asset=asset,noise_recomputation_max_error=maximum_error,passive_control_max_difference=doc['passive_control_max_difference'],target_response_gap=doc['changed_force_target_pair_rms_difference'],decision_summary=doc['decision_summary']))
 out=dict(complete=True,integrity_pass=not issues,issues=issues,checks=checks,declared_assets=3,declared_cases=27,completed_cases=len(cases),protocol_sha256=sha(REPORT/'protocol.json'),updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),case_records=cases,analysis=analysis_summaries,limits=['No metric advantage or coverage threshold is required for integritypass.','600noise realizations represent3assetgeometries and2latentworlds each, not600independentobjects.','Corrective measurements use renderedforeground, nothiddencontrols; predictiontarget is3Dcentroid scalarcoordinateRMS.','Nativeparameterreplay and controlledscaleambiguity areseparateclaims.'])
 (REPORT/'independent-evidence-audit.json').write_text(json.dumps(out,indent=2));print({k:v for k,v in out.items() if k not in ['case_records','analysis']})
if __name__=='__main__':main()
