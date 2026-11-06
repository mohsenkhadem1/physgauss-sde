"""Aggregate every declared case, paired comparisons, figures and paper tables."""
from pathlib import Path
import json,csv,hashlib
import numpy as np
from . import surface as bank
ROOT=bank.ROOT;REPORT=ROOT/'reports/joint-field-surface-v04';OUT=ROOT/'output/joint-field-surface-v04';PAPER=ROOT/'papers/cvpr2027-joint-identification'
POLICIES=['active','fixed','fixed-view','fixed-probe','random'];POP=['matched','continuous-coarse','offgrid-fine','force-prior']
METRICS=['rmse_mm','hidden_rmse_mm','changed_load_rmse_mm','material_relative_rmse','force_rmse_N','force_profile_rmse_N_per_m','support_center_error_mm','load_center_error_mm','coverage90','hidden_coverage90','changed_coverage90','material_coverage90','force_coverage90','interval_width_mm']

def paired(rows,pop,a,b,key='rmse_mm',step=4):
    def subset(policy):return {(r['shape'],r['case']):r[key] for r in rows if r['population']==pop and r['policy']==policy and r['step']==step and key in r}
    aa,bb=subset(a),subset(b);ids=sorted(set(aa)&set(bb));x=np.array([aa[i]-bb[i] for i in ids]);rng=np.random.default_rng(570001)
    if not len(x):return dict(n=0)
    boot=x[rng.integers(len(x),size=(2000,len(x)))].mean(1)
    return dict(n=len(x),mean_difference=float(x.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),a_mean=float(np.mean([aa[i] for i in ids])),b_mean=float(np.mean([bb[i] for i in ids])),fraction_a_better=float(np.mean(x<0)))

def main():
    OUT.mkdir(exist_ok=True);protocol=json.loads((REPORT/'protocol.json').read_text());rows=[];counts={};actions=[]
    for shape in protocol['shapes']:
      for pop in POP:
        docs=[json.loads(p.read_text()) for p in sorted((REPORT/shape/pop).glob('case-*.json'))]
        counts[f'{shape}/{pop}']=dict(completed=len(docs),expected=protocol['case_counts'][pop])
        for d in docs:
          for r in d['rows']:rows.append(dict(shape=shape,population=pop,case=d['case'],**r))
          for a in d['acquisition']:actions.append(dict(shape=shape,population=pop,case=d['case'],step=a['step'],action=a['chosen'],kind='probe' if a['chosen']>=10 else 'view'))
    fields=sorted(set().union(*(r.keys() for r in rows)))
    with (REPORT/'all-case-metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    summary=[]
    for pop in POP:
      for policy in POLICIES+['marginal-map','joint-map','uniform-material','fixed-interaction','independent-moments']:
       for step in range(5):
        rr=[r for r in rows if r['population']==pop and r['policy']==policy and r['step']==step]
        if not rr:continue
        item=dict(population=pop,policy=policy,step=step,n=len(rr))
        for key in METRICS:
            values=[r[key] for r in rr if key in r]
            if values:item[key]=float(np.mean(values))
        summary.append(item)
    comparisons=[]
    for pop in POP:
      for other in POLICIES[1:]:
       for metric in ['rmse_mm','hidden_rmse_mm','changed_load_rmse_mm','material_relative_rmse','force_rmse_N']:
        comparisons.append(dict(population=pop,first='active',second=other,metric=metric,**paired(rows,pop,'active',other,metric)))
    doc=dict(complete=all(v['completed']==v['expected'] for v in counts.values()),counts=counts,summary=summary,comparisons=comparisons,action_counts={pop:{kind:sum(a['population']==pop and a['kind']==kind for a in actions) for kind in ['view','probe']} for pop in POP},scope='Means of per-case coordinate RMSE. Bootstrap resamples paired cases across two fixed geometries; no claim of unseen-geometry population inference. Exact posterior only within finite dictionary. Continuous results separate.')
    (REPORT/'summary.json').write_text(json.dumps(doc,indent=2));(REPORT/'selected-actions.json').write_text(json.dumps(actions,indent=2))
    continuous=[]
    for p in sorted((REPORT/'continuous').glob('*.json')):
        d=json.loads(p.read_text())
        if d.get('complete') and 'job' in d and d['job']['population']!='development':continuous.append(d)
    (REPORT/'continuous-index.json').write_text(json.dumps([dict(path=str(p.relative_to(ROOT))) for p in sorted((REPORT/'continuous').glob('*.json')) if 'case' in p.name],indent=2))
    print(json.dumps(dict(complete=doc['complete'],counts=counts,continuous_jobs=len(continuous))),flush=True)
    return doc

if __name__=='__main__':
    import argparse,time,datetime
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    deadline=datetime.datetime.fromisoformat(json.loads((REPORT/'protocol.json').read_text())['deadline_utc'])
    while True:
        doc=main()
        if not args.watch or doc['complete'] or datetime.datetime.now(datetime.timezone.utc)>=deadline:break
        time.sleep(60)
