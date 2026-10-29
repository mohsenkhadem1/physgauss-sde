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

def main():
 seal=verify();settings=seal['settings'];config=json.loads(CONFIG.read_text());paper=ROOT/'papers'/BASE.name;paper.mkdir(parents=True,exist_ok=True)
 pt='phystwin-'+settings['phystwin']['tag'];mpm='physgaussian-'+settings['mpm']['tag'];LABELS.update({pt:'PhysTwin core adaptation',mpm:'PhysGaussian MPM adaptation'})
 main_methods=['extrapolated-n3-p3','neural-selected-id','neural-selected-all'];physical_methods=main_methods+[pt,mpm,'extrapolated_material_frame-n3-p3','extrapolated_nominal-n3-p3'];reports={}
 for split in SPLITS:
  reports[split]={}
  for name,methods,indices in [('main',main_methods,None),('matched',physical_methods,config['expensive_baseline_indices'][split])]:
   r=summarize(split,methods,indices);reports[split][name]=r;write(OUT/f'{split}-{name}-aggregate.json',r)
 qualification=[];missing=[]
 for split in ['development','calibration',*SPLITS]:
  for i in range(len(protocol()[split])):
   p=BASE/f'qualification/{split}-{i}.json'
   if not p.exists():missing.append(f'{split}-{i}');continue
   q=json.loads(p.read_text());f=BASE/f'fields/{split}-{i}.npz'
   with np.load(f) as z:
    qualification.append(dict(split=split,object=i,mesh=q['mesh'],spatial_rms_um=1e6*max(q['spatial'][-1]['rms_m']),quadrature_rms_um=1e6*max(q['quadrature']['rms_m']),minimum_J=float(z['minimum_J'].min()),residual_N=float(z['residual_N'].max()),reaction_balance_N=float(z['reaction_balance_N'].max())))
 write(OUT/'reference-summary.json',dict(rows=qualification,missing=missing))
 # Counts stay visible: partial completion can never silently become a full cohort.
 tables=[]
 for kind,methods in [('main',main_methods),('matched',physical_methods)]:
  tables+=['\\begin{table*}[t]','\\centering\\small','\\begin{tabular}{lrrrrrr}','\\toprule','& \\multicolumn{2}{c}{Familiar parameters} & \\multicolumn{2}{c}{Material shift} & \\multicolumn{2}{c}{Environment shift} \\\\', 'Method & New motions & $2\\times$ amplitude & New motions & $2\\times$ amplitude & New motions & $2\\times$ amplitude \\\\','\\midrule']
  for method in methods:
   cells=[]
   for split in SPLITS:
    entry=next(r for r in reports[split][kind]['rows'] if r['method']==method)
    for group in ['new_interaction','larger_amplitude']:
     g=entry['groups'][group];s=g['summary'];value='--' if s is None else f'{s["mean"]:.3f}';value+=('' if entry['complete'] else f' ({len(g["objects"])}/{entry["declared_objects"]})');cells.append(value)
   tables.append(LABELS[method]+' & '+' & '.join(cells)+' \\\\')
  caption=('All 42 declared test objects.' if kind=='main' else 'Matched physical comparison on 18 objects selected before evaluation (six per cohort). Native physical components use our shared-interface adapters.')
  tables+=['\\bottomrule','\\end{tabular}',f'\\caption{{Coordinate RMSE (mm), lower is better. {caption} Missing or incomplete results retain explicit counts. Object-level bootstrap intervals and familiar-direction queries are provided in the accompanying report.}}',f'\\label{{tab:{kind}}}','\\end{table*}']
 (paper/'generated-tables.tex').write_text('\n'.join(tables)+'\n')
 sources=[BASE/'method-seal.json',BASE/'neural-selection.json']+list(OUT.glob('*-aggregate.json'))+[OUT/'reference-summary.json']
 write(OUT/'publication-manifest.json',dict(seal_created=seal['created_utc'],sources={str(p.relative_to(ROOT)):digest(p) for p in sources},scope='Generated from stored frozen predictions. Reference labels are read only for scoring. Missing results are retained.'))
 print(json.dumps({s:{k:all(r['complete'] for r in v['rows']) for k,v in d.items()} for s,d in reports.items()}))
if __name__=='__main__':main()
