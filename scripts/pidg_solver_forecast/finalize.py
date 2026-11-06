"""Collect complete-case results without choosing models from final metrics."""
import time,subprocess,json,shutil,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'runs/pidg-solver-forecast-v01';REPORT=ROOT/'reports/pidg-solver-forecast-v01';PY=Path.home()/'.local/share/physgauss-external/pidg-official-v01/env/bin/python';scenes=['balls_reaction','rubber_duck','motion_kuro']
while not all((BASE/s/'complete.json').exists() for s in scenes):time.sleep(15)
for s in scenes:
 if not (REPORT/s/'temporal-physics.json').exists():
  with (BASE/s/'physics-audit.log').open('w') as f:subprocess.run([str(PY),'scripts/pidg_solver_forecast/audit_predicted_physics.py','--scene',s,'--gpu','2'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
 for name in ['selection.json']:
  shutil.copy2(BASE/s/'evaluation'/name,REPORT/s/name)
 for arm in ['native-base','native-refined','hybrid']:shutil.copy2(BASE/s/arm/'run.json',REPORT/s/(arm+'-run.json'))
with (BASE/'report.log').open('w') as f:subprocess.run([str(ROOT/'runs/phystwin-env/bin/python'),'scripts/pidg_solver_forecast/report.py'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
subprocess.run(['python3','scripts/pidg_solver_forecast/audit.py'],cwd=ROOT,check=True)
lines=['PIDG: solver-driven future-frame qualification','============================================','', 'Training: original fitting frames through 60% of sequence time.','Model selection: forecast-validation frames at 60-70%.','Final evaluation: released test frames after 70%; 11/15/15 images.','These public datasets were previously exposed; this is a new chronological protocol.','Old full-sequence appearance checkpoints are excluded. Previous paper results are unchanged.','', 'OBSERVED-PREFIX APPEARANCE (foreground PSNR, higher is better)']
for s in scenes:
 d=json.loads((REPORT/s/'appearance-fit.json').read_text());lines.append(f"{s}: native prefix base {d['native_prefix_validation_foreground_psnr']:.3f} dB; bound/refined {d['best']['score']:.3f} dB")
lines+=['','FUTURE FRAME RESULTS','scene | method | completed | PSNR | foreground PSNR | SSIM | LPIPS | centroid px | flow disagreement px']
for s in scenes:
 d=json.loads((REPORT/s/'results.json').read_text());flow=json.loads((REPORT/s/'motion.json').read_text())
 for method,record in d['summary'].items():
  metrics=record['metrics'];values=[metrics.get(k) for k in ['psnr_db','foreground_psnr_db','ssim','lpips_alex','silhouette_centroid_error_px']]+[flow['summary'].get(method)];numbers=['NA' if x is None else f'{x:.4f}' for x in values];lines.append(' | '.join([s,method,f"{record['completed_frames']}/{record['expected_frames']}",*numbers]))
 lines.append('Native source selected on validation: '+d['native_selected'])
 if d['failures']:lines.append('Failures: '+json.dumps(d['failures']))
lines+=['','PHYSICAL DIAGNOSTICS (not ground-truth accuracy)']
for s in scenes:
 d=json.loads((REPORT/s/'results.json').read_text());rows=[r for r in d['rows'] if r['method']=='solver'];
 if rows:lines.append(f"{s}: sampled min J {min(r['minimum_J'] for r in rows):.6f}; max net internal force {max(r['net_internal_force'] for r in rows):.3g}; maximum sampled interbody depth/object extent {max(r.get('sampled_interbody_max_depth_normalized',0) for r in rows):.6f}")
 temporal=json.loads((REPORT/s/'temporal-physics.json').read_text());lines.append('Temporal-field Jacobian audit: '+json.dumps(temporal['summary']))
lines+=['','INTERPRETATION AND LIMITS','All methods use prefix-only appearance initialization and the same fitting observations.','Native PIDG chooses base/refined on validation. Hybrid has the previously fixed 1e-5 weight.','Solver adds static appearance fitting and RGB multiple-shooting physical identification.','A constant-velocity control shares the bound appearance and selects velocity filtering on validation.','Balls has six first-fitting-frame-assisted instances; the others use occupancy components.','Volumes and units are inferred proxies. No true 3D states, masses, materials or forces are supplied.','Gaussian fields represent unknown interactions; this adapter does not enforce explicit contact.','Centroid/flow are image-motion proxies. Positive Jacobians and conservative internal forces','are numerical sanity checks, not a certificate of real physical accuracy.','Temporal-field Jacobians and FEM Jacobians use different representations; do not turn them','into an unqualified physical-error leaderboard.','The first raw-velocity duck attempt failed; it is retained. Target-free diagnostics showed','that decreasing dt did not cure the problem; filtering nonrigid initial velocities did.','Changes were driven by fitting/validation and stability diagnostics, not final-test scores.','Final figures use the middle declared test frame and shared reference crops; videos show','all sparse recorded test frames. No deformation exaggeration or per-method alignment.','Raw datasets/checkpoints/trajectories stay local. No novel-view or new-force ground truth is invented.']
(REPORT/'REPORT.txt').write_text('\n'.join(lines)+'\n');(REPORT/'completed.json').write_text(json.dumps(dict(completed=True,scenes=scenes,finished=time.time(),report='REPORT.txt'),indent=2));print('Report and comparisons complete',flush=True)
