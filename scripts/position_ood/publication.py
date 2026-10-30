"""Generate manuscript tables from the frozen, declared evaluation, never labels-as-input."""
from pathlib import Path
import json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.position_ood.study import BASE,OUT,ROOT,protocol
from scripts.position_ood.runtime import CONFIG
from scripts.position_ood.seal import verify,digest
from scripts.position_ood.aggregate import summarize
from scripts.position_ood.score import run as score
from scripts.phystwin.sequential_reference import write

SPLITS=['test_id','test_material','test_environment']
GROUPS=['interpolation','new_interaction','larger_amplitude']
LABELS={'extrapolated-n3-p3':'Spatial inverse (ours)','extrapolated_material_frame-n3-p3':'Material-indexed load','extrapolated_nominal-n3-p3':'Nominal environment','neural-selected-id':'Neural (ID actions)','neural-selected-all':'Neural (all actions)','undeformed':'Undeformed','last_full_state_oracle_persistence':'Full-state persistence (oracle)'}

def reference_summary():
 qualification=[];missing=[]
 counts={split:len(protocol()[split]) for split in ['development','calibration',*SPLITS]}
 counts.update(transfer_development=3,transfer_calibration=12,transfer_test=9)
 for split,count in counts.items():
  for i in range(count):
   p=BASE/f'qualification/{split}-{i}.json'
   if not p.exists():missing.append(f'{split}-{i}');continue
   q=json.loads(p.read_text());f=BASE/f'fields/{split}-{i}.npz'
   with np.load(f) as z:
    qualification.append(dict(split=split,object=i,mesh=q['mesh'],spatial_rms_um=1e6*max(q['spatial'][-1]['rms_m']),spatial_maximum_um=1e6*max(q['spatial'][-1]['maximum_m']),quadrature_rms_um=1e6*max(q['quadrature']['rms_m']),minimum_J=float(z['minimum_J'].min()),residual_N=float(z['residual_N'].max()),reaction_balance_N=float(z['reaction_balance_N'].max()),qualified=q['qualified']))
 report=dict(declared=sum(counts.values()),rows=qualification,missing=missing)
 write(OUT/'reference-summary.json',report)
 return report

def main():
 seal=verify();settings=seal['settings'];config=json.loads(CONFIG.read_text());paper=ROOT/'papers'/BASE.name;paper.mkdir(parents=True,exist_ok=True)
 pt='phystwin-'+settings['phystwin']['tag'];mpm='physgaussian-'+settings['mpm']['tag'];LABELS.update({pt:'PhysTwin core adaptation',mpm:'PhysGaussian MPM adaptation'})
 main_methods=['extrapolated-n3-p3','neural-selected-id','neural-selected-all'];physical_methods=main_methods+[pt,mpm,'extrapolated_material_frame-n3-p3','extrapolated_nominal-n3-p3'];reports={}
 for split in SPLITS:
  reports[split]={}
  for name,methods,indices in [('main',main_methods,None),('matched',physical_methods,config['expensive_baseline_indices'][split])]:
   r=summarize(split,methods,indices);reports[split][name]=r;write(OUT/f'{split}-{name}-aggregate.json',r)
 transfer=summarize('transfer_test',main_methods+[pt,mpm,'extrapolated_material_frame-n3-p3'],config['transfer_physical_baseline_indices']);write(OUT/'transfer_test-matched-aggregate.json',transfer)
 transfer_lines=['\\begin{table}[t]','\\centering\\small\\setlength{\\tabcolsep}{3pt}','\\begin{tabular}{lrr}','\\toprule','Method & Translations & Target \\\\','\\midrule']
 for entry in transfer['rows']:
  cells=[]
  for group in ['spatial_transfer','held_target']:
   g=entry['groups'][group];v=g['summary'];cell='--' if v is None else f'{v["mean"]:.3f}'
   if not entry['complete']:cell+=f' ({len(g["objects"])}/{entry["declared_objects"]})'
   cells.append(cell)
  transfer_lines.append(LABELS[entry['method']]+' & '+' & '.join(cells)+' \\\\')
 transfer_lines+=['\\bottomrule','\\end{tabular}','\\caption{Spatial-transfer coordinate RMSE (mm) before acquisition, on six matched objects. Every model reuses its original three-observation fit. The held target is the $-40$ mm $y$ translation.}','\\label{tab:transfer}','\\end{table}']
 (paper/'generated-transfer-table.tex').write_text('\n'.join(transfer_lines)+'\n')
 reference_summary()
 # Counts stay visible: partial completion can never silently become a full cohort.
 tables=[]
 for kind,methods,metric in [('main',main_methods,'coordinate'),('matched',physical_methods,'coordinate'),('incremental',physical_methods,'incremental_coordinate_rmse_mm')]:
  first_line=len(tables)
  tables+=['\\begin{table*}[t]','\\centering\\small\\setlength{\\tabcolsep}{3pt}','\\begin{tabular}{lrrrrrr}','\\toprule','& \\multicolumn{2}{c}{Familiar parameters} & \\multicolumn{2}{c}{Material shift} & \\multicolumn{2}{c}{Environment shift} \\\\', 'Method & New motions & $2\\times$ amplitude & New motions & $2\\times$ amplitude & New motions & $2\\times$ amplitude \\\\','\\midrule']
  for method in methods:
   cells=[]
   for split in SPLITS:
    entry=next(r for r in reports[split]['main' if kind=='main' else 'matched']['rows'] if r['method']==method)
    for group in ['new_interaction','larger_amplitude']:
     g=entry['groups'][group];s=g['summary'] if metric=='coordinate' else g[metric]['summary'];value='--' if s is None else f'{s["mean"]:.3f}';value+=('' if entry['complete'] else f' ({len(g["objects"])}/{entry["declared_objects"]})');cells.append(value)
   tables.append(LABELS[method]+' & '+' & '.join(cells)+' \\\\')
  caption=('All 42 declared test objects.' if kind=='main' else 'Matched physical comparison on 18 objects selected before evaluation (six per cohort). Native physical components use our shared-interface adapters.')
  if kind=='incremental':caption="Incremental response error after subtracting each method's nominal-pose prediction, on the same matched 18-object subset."
  tables+=['\\bottomrule','\\end{tabular}',f'\\caption{{Coordinate RMSE (mm), lower is better. {caption} Object-level bootstrap intervals and familiar-direction queries are provided in the accompanying report.}}',f'\\label{{tab:{kind}}}','\\end{table*}']
  (paper/f'generated-{kind}-table.tex').write_text('\n'.join(tables[first_line:])+'\n')
 (paper/'generated-tables.tex').write_text('\n'.join(tables)+'\n')
 sources=[BASE/'method-seal.json',BASE/'neural-selection.json']+list(OUT.glob('*-aggregate.json'))+[OUT/'reference-summary.json']
 raw=[]
 for pattern in ['inverse/test_*/*.npz','inverse/transfer_test-*/*.npz','phystwin/test_*/*/predictions.npz','phystwin/transfer_test-*/*/predictions.npz','mpm*/test_*/predictions.npz','mpm*/transfer_test-*/predictions.npz','neural-predictions/test_*.npz','neural-predictions/transfer_test-*.npz','calibration/*.json']:
  raw.extend(BASE.glob(pattern))
 write(OUT/'publication-manifest.json',dict(seal_created=seal['created_utc'],sources={str(p.relative_to(ROOT)):digest(p) for p in sources},local_prediction_hashes={str(p.relative_to(ROOT)):digest(p) for p in sorted(set(raw))},scope='Generated from stored frozen predictions. Reference labels are read only for scoring. Missing results are retained.'))
 print(json.dumps({s:{k:all(r['complete'] for r in v['rows']) for k,v in d.items()} for s,d in reports.items()}))
if __name__=='__main__':main()
