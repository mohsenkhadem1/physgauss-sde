"""Independent arithmetic/archive/provenance audit of the frozen native extension."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import imageio.v2 as imageio
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.gap_first_v01.drop_protocol import verify
REPORT=ROOT/'reports/gap-first-v01/drop';BASE=ROOT/'runs/gap-first-v01/drop';EXT=REPORT/'native-reference-extension';RUN=BASE/'native-reference-extension'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=verify();m=json.loads((EXT/'protocol.json').read_text());assert sha(ROOT/'scripts/gap_first_v01/drop_native_extension.py')==m['source_sha256']
for f,h in m['input_sha256'].items():assert sha(ROOT/f)==h
checks=0;rows=[]
for a in p['assets']:
 d=json.loads((EXT/f'{a}-result.json').read_text());pred=json.loads((EXT/f'{a}-frozen-prediction.json').read_text());risk=json.loads((EXT/f'{a}-preoutcome-risk.json').read_text());q=np.array(json.loads((REPORT/f'{a}-preobservation-bank.json').read_text())['target_bank']);primary=json.loads((REPORT/f'{a}-analysis.json').read_text());changed=np.load(RUN/a/'changed.npz');passive=np.load(RUN/a/'passive.npz');qhat=changed['centroids'][18:25].astype(float).reshape(-1);errors=np.linalg.norm(q-qhat,axis=1)/np.sqrt(21);np.testing.assert_allclose(errors,d['candidate_errors'],rtol=1e-12);np.testing.assert_array_equal(qhat,risk['qhat']);assert d['params']==pred['native_params'];assert pred['prediction_fixed_before_force_simulation'];checks+=5
 for condition in ['passive','changed']:assert sha(RUN/a/(condition+'.npz'))==d[condition+'_archive_sha256'];checks+=1
 for w in 'AB':
  movie=RUN/a/f'passive-{w}.mp4';assert sha(movie)==pred['input_movie_sha256'][w];np.testing.assert_array_equal(np.array(imageio.mimread(movie)),np.load(BASE/'cases'/a/f'passive-{w}'/'trajectory.npz')['rgb'][:,0]);checks+=2
 events=errors>.01
 assert len(d['rows'])==len(primary['noise_rows'])==400
 for r,pr in zip(d['rows'],primary['noise_rows']):
  assert (r['world'],r['stage'],r['replicate'])==(pr['world'],pr['stage'],pr['replicate']);prob=pr['posterior_A']*events[0]+(1-pr['posterior_A'])*events[1];np.testing.assert_allclose(r['native_predicted_failure_probability'],prob,atol=1e-15);np.testing.assert_allclose(r['native_realized_coordinate_rms'],errors[r['world']],rtol=1e-12);np.testing.assert_allclose(r['native_brier'],(prob-events[r['world']])**2,atol=1e-15);assert r['corrected_MAP_error']==pr['point_refit_rms_error'];checks+=5
 gt=np.load(BASE/'cases'/a/'passive-A/trajectory.npz')['rgb']
 for f,sl in zip(d['fit'],[slice(0,20),slice(20,30)]):
  mse=np.mean((gt[sl,0].astype(float)-passive['rgb'][sl,0].astype(float))**2);np.testing.assert_allclose(f['mse'],mse);np.testing.assert_allclose(f['psnr_db'],10*np.log10(255**2/mse));checks+=2
 rows.append(dict(asset=a,candidate_coordinate_rms=errors.tolist(),max_AB_parameter_difference=pred['max_AB_parameter_difference'],native_risk_before=float(np.mean(events)),native_fails_both=bool(events.all())))
out=dict(passed=True,checks=checks,assets=rows,protocol_sha256=sha(EXT/'protocol.json'),audit_source_sha256=sha(__file__),scope='Independent archive/metric/row/provenance recomputation; frozen sources and primary campaign unchanged. Two candidate worlds per each of three geometries, not 600 independent objects.',limits='Native model remains fixed after correction; corrected MAP refits. Risk=1 for both candidates is a correct warning, not broad calibration evidence.')
(EXT/'independent-audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
