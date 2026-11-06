"""Complete paired fine-reference numerical report; never pools surface data."""
from pathlib import Path
import json,csv
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; REPORT=ROOT/'reports/joint-field-fine-v03'

def main():
    rows=[]
    for shape in ['block','step']:
      for model in ['coarse8','fine24']:
       for case in range(16):
        p=REPORT/'evaluation'/shape/model/f'case-{case:03d}.json'
        if not p.exists():raise RuntimeError(f'Incomplete: {p}')
        d=json.loads(p.read_text())
        for r in d['rows']:rows.append(dict(shape=shape,model=model,case=case,**r))
    fields=sorted(set().union(*(r.keys() for r in rows)))
    with (REPORT/'all-case-metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
    metrics=['rmse_mm','hidden_rmse_mm','changed_load_rmse_mm','material_relative_rmse','force_rmse_N','coverage90','material_coverage90','force_coverage90','interval_width_mm'];summary=[];paired=[]
    for model in ['coarse8','fine24']:
      for policy in ['active','fixed','fixed-view','fixed-probe','random']:
       for step in [0,4]:
        rr=[r for r in rows if r['model']==model and r['policy']==policy and r['step']==step]
        assert len(rr)==32
        summary.append(dict(model=model,policy=policy,step=step,n=32,**{k:float(np.mean([r[k] for r in rr])) for k in metrics}))
    rng=np.random.default_rng(190001)
    samples=np.concatenate([rng.integers(16,size=(4000,16)),16+rng.integers(16,size=(4000,16))],1)
    for policy in ['active','fixed','fixed-view','fixed-probe','random']:
      for step in [0,4]:
       for metric in metrics:
        rr={m:sorted([r for r in rows if r['model']==m and r['policy']==policy and r['step']==step],key=lambda r:(r['shape'],r['case'])) for m in ['coarse8','fine24']}
        delta=np.array([a[metric]-b[metric] for a,b in zip(rr['fine24'],rr['coarse8'])])
        paired.append(dict(policy=policy,step=step,metric=metric,fine_minus_coarse=float(delta.mean()),ci95=np.quantile(delta[samples].mean(1),[.025,.975]).tolist()))
    out=dict(complete=True,cases=32,summary=summary,paired=paired,scope='Interior material sensor reference, not RGB surface study. Paired bootstrap within two fixed shapes. Fresh32mesh truth; discrete5^5prior in both inverse models. No continuum convergence claim.')
    (REPORT/'summary.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(out['summary'][:4],indent=2))
if __name__=='__main__':main()
