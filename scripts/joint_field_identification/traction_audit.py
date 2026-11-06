"""Independent record/aggregation check of posthoc surface traction diagnostic."""
from pathlib import Path
import hashlib,json
import numpy as np
R=Path(__file__).resolve().parents[2]/'reports/joint-field-surface-v04/traction'
def main():
    s=json.loads((R/'summary.json').read_text());ph=hashlib.sha256((R/'protocol.json').read_bytes()).hexdigest();rows={}
    for shape in ['block','step']:
      area=s['checks'][shape]['surface_areas_m2'];total=sum(area.values())
      for case in range(64):
        d=json.loads((R/f'{shape}-case-{case:03d}.json').read_text());assert d['protocol_sha256']==ph;assert len(d['rows'])==6
        for r in d['rows']:
          k=(shape,case,r['stage'],r['estimator']);assert k not in rows;rows[k]=r
          assert abs(r['traction_rmse_Pa']**2-(area['top']*r['top_rmse_Pa']**2+area['side']*r['side_rmse_Pa']**2)/total)<1e-9
    maxerr=0.
    for a in s['summary']:
        raw=[rows[(shape,case,a['stage'],a['estimator'])] for shape in ['block','step'] for case in range(64)]
        for metric in ['traction_rmse_Pa','top_rmse_Pa','side_rmse_Pa']:maxerr=max(maxerr,abs(a[metric]-sum(r[metric] for r in raw)/128))
    assert maxerr<1e-12
    rng=np.random.default_rng(193001);i=rng.integers(64,size=(4000,64));j=rng.integers(64,size=(4000,64));maxci=0.
    def paired(stage1,estimate1,stage2,estimate2):
        d={shape:np.array([rows[(shape,k,stage1,estimate1)]['traction_rmse_Pa']-rows[(shape,k,stage2,estimate2)]['traction_rmse_Pa'] for k in range(64)]) for shape in ['block','step']}
        bs=.5*(d['block'][i].mean(1)+d['step'][j].mean(1));return dict(mean=.5*(d['block'].mean()+d['step'].mean()),ci95=np.quantile(bs,[.025,.975]).tolist())
    for p in s['paired']:
        calc=paired(p['stage'],'joint',p['stage'],'independent');assert abs(calc['mean']-p['mean'])<1e-12;maxci=max(maxci,float(np.max(abs(np.array(calc['ci95'])-p['ci95']))))
    assert maxci<1e-12
    out=dict(passed=True,cases=128,unique_rows=len(rows),max_aggregate_error_Pa=maxerr,max_bootstrap_CI_error_Pa=maxci,area_weighting_identity_verified=True,summary_sha256=hashlib.sha256((R/'summary.json').read_bytes()).hexdigest(),additional_descriptive_paired_contrasts_Pa={'active_minus_initial':paired('active','joint','initial','joint'),'fixed_minus_initial':paired('fixed','joint','initial','joint'),'active_minus_fixed':paired('active','joint','fixed','joint')},scope='Independent aggregation and unit consistency audit of128fixedcase records; additional contrasts are descriptive posthoc statistics, not new confirmatory endpoints.')
    (R/'aggregation-audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
