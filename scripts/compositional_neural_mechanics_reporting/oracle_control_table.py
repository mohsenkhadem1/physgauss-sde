"""Join unchanged primary oracle results and supplementary normalization controls."""
import json
from pathlib import Path
from scripts.compositional_neural_mechanics.stable_zero_control import ROOT,REPORT,sha,guard
from scripts.compositional_neural_mechanics.dataset_runner import write

def main():
 guard();paths=[REPORT/'oracle-reserved-forward-results.json',REPORT/'stable-zero-control-results.json',REPORT/'stable-zero-scoring-reference-audit.json']
 primary,control,audit=[json.loads(p.read_text())for p in paths]
 if not audit['passed']:raise ValueError('Reference provenance audit required')
 for p,h in audit['reference_evidence_sha256'].items():
  if sha(ROOT/p)!=h:raise ValueError('Scored reference evidence changed')
 rows=[]
 for source,report in [('frozen_primary',primary),('supplementary_source_audit_control',control)]:
  for row in report['summaries']:rows.append(dict(source=source,**row))
 if len(rows)!=14 or any(r['expected_commands']!=24 for r in rows):raise ValueError('Exact7methods by2cohorts required')
 out=dict(scope='ORACLE descriptors: geometry and physical material/support descriptors are supplied. Stable variants are post-freeze source-only preprocessing controls; retain original five methods and both controls, no selection.',rows=rows,evidence_sha256={str(p.relative_to(ROOT)):sha(p)for p in paths})
 write(REPORT/'oracle-plus-stable-comparison.json',out)
 lines=[out['scope'],'','Cohort | Method | Source | Scored/24 | Displacement mm | Force N | Torque Nm']
 for r in rows:
  m=r.get('object_mean_scores',{});values=[m.get(k)for k in ('displacement_coordinate_rmse_m','force_coordinate_rmse_N','torque_coordinate_rmse_Nm')]
  values=[('unavailable'if x is None else f'{x*(1000 if i==0 else 1):.6g}')for i,x in enumerate(values)]
  lines.append(' | '.join([r['cohort'],r['method'],r['source'],str(r['scored_commands'])+'/24',*values]))
 (REPORT/'ORACLE-PLUS-STABLE-COMPARISON.txt').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
