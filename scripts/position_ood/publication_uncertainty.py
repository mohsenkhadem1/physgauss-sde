"""Generate uncertainty tables from separately calibrated, frozen predictions."""
from pathlib import Path
import json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.position_ood.study import BASE,OUT,ROOT
from scripts.position_ood.seal import verify,digest
from scripts.position_ood.score import paired_interval
from scripts.phystwin.sequential_reference import write

def describe(rows):
 return dict(objects=len(rows),mean_rmse_mm=float(np.mean([r['rmse_mm'] for r in rows])),coordinate_coverage=float(np.mean([r['coordinate_coverage'] for r in rows])),covered_objects=sum(r['at_least_90_percent_coordinates'] for r in rows),mean_half_width_mm=float(np.mean([r['mean_half_width_mm'] for r in rows])))

def acquisition_sources(active):
 if not active['complete'] or not active['matched_frame_ablation']['complete']:
  raise RuntimeError('Complete acquisition and matched-frame cohorts required')
 if active['declared_objects']!=9 or [r['object'] for r in active['rows']]!=list(range(9)):
  raise RuntimeError('Expected all nine declared acquisition objects')
 calibration=BASE/'calibration/transfer.json';cal=json.loads(calibration.read_text())
 paths={BASE/'method-seal.json',calibration,OUT/'transfer_test-active-results.json'}
 for name,expected in cal['sources'].items():
  path=BASE/name
  if digest(path)!=expected:raise RuntimeError(f'Changed calibration input: {name}')
  paths.add(path)
 for row in active['rows']:
  i=row['object'];directory=BASE/f'inverse/transfer_test-{i}'
  choices=BASE/f'active-jobs/transfer_test-{i}-choices.json'
  if json.loads(choices.read_text())['actions']!=row['actions']:
   raise RuntimeError(f'Acquisition choices differ for object {i}')
  paths.update([choices,BASE/f'fields/transfer_test-{i}.npz'])
  tags=['p3']+[f'p4-0-1-2-{a}' for a in sorted(set(row['actions'].values()))]
  for tag in tags:
   paths.update([directory/f'extrapolated-n3-{tag}.npz',directory/f'extrapolated-n3-{tag}-uncertainty.npz'])
  for tag in ['p3',f'p4-0-1-2-{row["actions"]["adaptive"]}']:
   paths.add(directory/f'extrapolated_material_frame-n3-{tag}.npz')
 return {str(path.relative_to(ROOT)):digest(path) for path in sorted(paths)}

def main():
 verify();paper=ROOT/'papers'/BASE.name;local={}
 for split in ['test_id','test_material','test_environment']:
  d=json.loads((OUT/f'{split}-extrapolated-coverage.json').read_text());local[split]=describe(d['rows'])
 lines=['\\begin{table}[t]','\\centering\\small\\setlength{\\tabcolsep}{3pt}','\\begin{tabular}{lrrr}','\\toprule','Cohort & Coordinates & Objects & Half-width \\\\','\\midrule']
 for split,label in [('test_id','Familiar parameters'),('test_material','Material shift'),('test_environment','Environment shift')]:
  d=local[split];lines.append(f'{label} & {100*d["coordinate_coverage"]:.1f}\\% & {d["covered_objects"]}/{d["objects"]} & {d["mean_half_width_mm"]:.3f} \\\\')
 lines+=['\\bottomrule','\\end{tabular}','\\caption{Prediction-interval coverage for held local actions. Coordinates is mean coordinate coverage; Objects counts cases with at least 90\\% of coordinates covered. Half-width is in mm. One fixed scale is fitted on twelve independent calibration objects.}','\\label{tab:coverage}','\\end{table}'];(paper/'generated-coverage-table.tex').write_text('\n'.join(lines)+'\n')
 active=json.loads((OUT/'transfer_test-active-results.json').read_text())
 sources=acquisition_sources(active)
 policies={'before':describe([r['before'] for r in active['rows']])}
 policies.update({p:describe([r['after'][p] for r in active['rows']]) for p in ['fixed','random','adaptive']})
 lines=['\\begin{table}[t]','\\centering\\small\\setlength{\\tabcolsep}{3pt}','\\begin{tabular}{lrrr}','\\toprule','Acquisition & Error & Coverage & Half-width \\\\','\\midrule']
 for name,d in policies.items():lines.append(f'{name.capitalize()} & {d["mean_rmse_mm"]:.3f} & {100*d["coordinate_coverage"]:.1f}\\% & {d["mean_half_width_mm"]:.3f} \\\\')
 lines+=['\\bottomrule','\\end{tabular}','\\caption{Held spatial target before and after one acquired response on nine objects. Error is coordinate RMSE and widths are in mm. Each policy uses a separately fitted scale from twelve calibration objects; coverage and width are assessed together.}','\\label{tab:active}','\\end{table}'];(paper/'generated-active-table.tex').write_text('\n'.join(lines)+'\n')
 write(OUT/'uncertainty-publication-summary.json',dict(local=local,active=policies,adaptive_advantage_mm=active['adaptive_advantage_mm'],matched_frame_ablation=active['matched_frame_ablation'],acquisition_source_hashes=sources))
if __name__=='__main__':main()
