"""POST-HOC secondary reference-resolution sensitivity of fixed predictions.
No refitting, new simulation, replacement primary scores, or gate relaxation.
"""
import json
from pathlib import Path
import numpy as np
from scripts.compositional_dynamics_v17_reporting.cohort_recovery_evaluation import committed
from scripts.compositional_dynamics_v17.forecast_metrics import compare_surface_forecast
from scripts.compositional_dynamics_v17.prototype import ROOT,REPORT,sha,write,face_grid
BASE=ROOT/'runs/compositional-dynamics-v17/fresh-cohort'
OUT=REPORT/'fresh-cohort-reference-robustness-posthoc.json'

def main():
 if OUT.exists():raise ValueError('Preserve previous secondary result')
 protocol=REPORT/'fresh-cohort-protocol.json';freeze=REPORT/'fresh-cohort-method-freeze.json'
 bindings=dict(protocol_path=str(protocol),freeze_path=str(freeze),approval_path=str(REPORT/'fresh-cohort-prefit-independent-audit.json'),protocol_sha=sha(protocol),freeze_sha=sha(freeze))
 p,predictions,evidence=committed(bindings);primary_path=REPORT/'fresh-cohort-geometry-recovery-results.json';primary=json.loads(primary_path.read_text());evidence[str(primary_path.relative_to(ROOT))]=sha(primary_path)
 rows=[];fields=('world_point_rms_m','shape_point_rms_m');role='forecast-80mm';dims=np.asarray(p['generator_settings']['dimensions_m'])
 for world in ('world-0','world-1'):
  meta_path=BASE/'public'/world/'metadata.json';meta=json.loads(meta_path.read_text());status_path=meta_path.with_name('complete.json');export=json.loads(status_path.read_text())
  if sha(meta_path)!=export['public_metadata_sha256']:raise ValueError('Public geometry changed')
  q=np.asarray(meta['material_queries']);rest=q*dims+[0,-.03,-.03];dense=face_grid(17);indices=[np.flatnonzero(np.all(dense==point,axis=1))[0]for point in q]
  for f in (meta_path,status_path):evidence[str(f.relative_to(ROOT))]=sha(f)
  for label,n,dt in [('n4_base',4,25e-6),('n6_base',6,25e-6),('n6_halfstep',6,12.5e-6)]:
   folder=BASE/'references'/world/role/f'n{n}-dt{dt:.8f}';status_path=folder/'complete.json';status=json.loads(status_path.read_text());evidence[str(status_path.relative_to(ROOT))]=sha(status_path)
   if status['world_id']!=world or status['role']!=role or status['n']!=n or status['dt_s']!=dt or status['protocol_sha256']!=sha(protocol)or status['method_freeze_sha256']!=sha(freeze):raise ValueError('Reference identity mismatch')
   row=dict(world_id=world,role=role,reference=label,reference_complete=status['completed'],methods={})
   if not status['completed']:row['error']=status.get('error');rows.append(row);continue
   source=folder/'trajectory.npz'
   if sha(source)!=status['trajectory_sha256']:raise ValueError('Reference trajectory changed')
   evidence[str(source.relative_to(ROOT))]=sha(source)
   with np.load(source)as f:t=f['times_s'][::10];x=f['surface_frames'][::10].reshape(len(t),-1,3)[:,indices]
   if len(t)!=31:raise ValueError('Full31frame reference required')
   for pred in predictions:
    if pred['world_id']!=world or pred['role']!=role:continue
    result=dict(completed=pred['record']['completed'],error=pred['record'].get('error'))
    if result['completed']:
     with np.load(pred['array'])as f:pt=f['times_s'];y=f['positions_m'if pred['method']=='ours'else'tracked_positions_m']
     score=compare_surface_forecast(rest,q,dims,t,x,pt,y)
     if not score['complete']:raise ValueError('Committed prediction incomplete')
     result.update({field:score[field]for field in fields})
     if label=='n4_base':
      old=next(r for r in primary['rows']if r['world_id']==world and r['role']==role)['methods'][pred['method']]['geometry']
      if any(not np.isclose(score[k],old[k],rtol=1e-12,atol=1e-15)for k in fields):raise ValueError('Primary n4 metrics not exactly reproduced')
    row['methods'][pred['method']]=result
   row['paired_complete']=all(row['methods'][m]['completed']for m in ('ours','native_phystwin'))
   if row['paired_complete']:row['ours_minus_native_m']={field:row['methods']['ours'][field]-row['methods']['native_phystwin'][field]for field in fields}
   rows.append(row)
 changes=[]
 for world in ('world-0','world-1'):
  selected=[r for r in rows if r['world_id']==world];valid=all(r.get('paired_complete',False)for r in selected)
  changes.append(dict(world_id=world,all_three_references_paired=valid,ranking_signs={field:{r['reference']:int(np.sign(r['ours_minus_native_m'][field]))for r in selected}for field in fields}if valid else None,scope='Negative means smaller error for ours; missing/failed predictions remain unavailable, never imputed.'))
 write(OUT,dict(posthoc_secondary=True,protocol_sha256=sha(protocol),method_freeze_sha256=sha(freeze),fixed_worlds=['world-0','world-1'],fixed_role=role,rows=rows,ranking_checks=changes,evidence_sha256=evidence,source_sha256={str(Path(__file__).relative_to(ROOT)):sha(__file__),**{str((ROOT/name).relative_to(ROOT)):sha(ROOT/name)for name in ['scripts/compositional_dynamics_v17/forecast_metrics.py','scripts/compositional_dynamics_v17/sequence_metrics.py','scripts/compositional_dynamics_v17_reporting/cohort_recovery_evaluation.py']}},scope='POST-HOC secondary reference-robustness diagnostic. Same sealed31frame98point predictions and rest-area metric at n4/n6/n6halfstep targets on both fixed sensitivity worlds. Primary scores and failed spatial qualification remain unchanged. Ranking stability does not establish continuum or real-world accuracy.'))
if __name__=='__main__':main()
