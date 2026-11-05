from pathlib import Path
import json,hashlib,numpy as np
root=Path('runs/pidg-mechanics-v04');r=json.loads((root/'evaluation/results.json').read_text());s=json.loads((root/'final-selection.json').read_text());assert s['checkpoint_sha256']==hashlib.sha256((root/s['checkpoint']).read_bytes()).hexdigest()
assert len(r['rows'])==114 and len(r['novel_views'])==21
assert all(np.isfinite([row[k] for k in ['psnr_db','foreground_psnr_db','ssim','lpips_alex']]).all() for row in r['rows'])
fit=json.loads(Path('runs/pidg-comparison-v01/data/balls_reaction/fit/transforms_train.json').read_text());test=json.loads(Path('runs/pidg-comparison-v01/data/balls_reaction/evaluation/transforms_test.json').read_text());assert not ({f['file_path'] for f in fit['frames']}&{f['file_path'] for f in test['frames']})
geom=np.load(root/'training-geometry.npz');np.testing.assert_allclose(geom['times'],[f['time'] for f in fit['frames']])
d=r['mechanics_diagnostics'];assert len(d)==228 and all(x['inverted_tetrahedra']==0 for x in d)
j=json.loads((root/'joint-selection.json').read_text());grads=j['gradient_records'];assert len(grads)==400 and max(x['material_gradient_norm'] for x in grads)>0 and max(x['boundary_gradient_norm'] for x in grads)>0
out={'passed':True,'fitting_images':len(fit['frames']),'test_images':len(test['frames']),'test_resolution_pairs':len(r['rows']),'novel_camera_times':21,'checkpoint_hash_verified':True,'fit_test_ids_disjoint':True,'training_control_clock_verified':True,'fem_tetrahedron_evaluations':sum(x['tetrahedra'] for x in d),'nonpositive_jacobians':0,'minimum_J':min(x['minimum_J'] for x in d),'maximum_solver_residual':max(x['solver']['residual'] for x in d),'material_gradient_max_norm':max(x['material_gradient_norm'] for x in grads),'boundary_gradient_max_norm':max(x['boundary_gradient_norm'] for x in grads),'limitations':'No independent novel-view targets; no full coupled collision/inference or physical-accuracy claim.'}
Path('reports/pidg-mechanics-v01/final-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
