"""Assemble complete surface hierarchy reports and figures without changing fits."""
import argparse,hashlib,json,math,time
import numpy as np
from pathlib import Path
from .hierarchy_surface import REPORT,verify_protocol,summarize
from .hierarchy_surface_figure import main as figures


def main(wait_seconds=0):
    deadline=time.time()+wait_seconds
    while not all(len(list((REPORT/s).glob('case-*.json')))==64 for s in ['block','step']):
        if time.time()>=deadline:raise RuntimeError('Full128case cohort not yet complete')
        time.sleep(20)
    time.sleep(2);protocol,sha=verify_protocol();docs=[];compared=0;maximum=0.
    for shape in ['block','step']:
        for i in range(64):
            d=json.loads((REPORT/shape/f'case-{i:03d}.json').read_text());assert d['protocol_sha256']==sha
            assert len(d['rows'])==12 and len({(r['model'],r['condition']) for r in d['rows']})==12
            for r in d['rows']:
                for key,value in r.items():
                    if isinstance(value,float):assert math.isfinite(value),(shape,i,key)
            if d['truth_tau']==1.:
                for condition in ['initial','views','probes']:
                    a=next(r for r in d['rows'] if r['model']=='fixed-nominal' and r['condition']==condition)
                    b=next(r for r in d['rows'] if r['model']=='oracle-scale' and r['condition']==condition)
                    for key in ['rmse_mm','material_relative_rmse','force_rmse_N','coverage90','material_coverage90','force_coverage90']:
                        delta=abs(a[key]-b[key]);maximum=max(maximum,delta);assert delta==0.
                    compared+=1
            docs.append(d)
    audit=dict(passed=True,cases=len(docs),metric_rows=sum(len(d['rows']) for d in docs),
               finite_scalar_metrics=True,protocol_hash_all_cases=sha,
               oracle_tau1_vs_nominal_comparisons=compared,maximum_difference=maximum,
               frozen_source_hashes_verified=True,source_sha256=protocol['source_sha256'])
    (REPORT/'completion-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    summarize()
    rng=np.random.default_rng(63001)
    indices=np.concatenate([rng.integers(0,64,(2048,64)),64+rng.integers(0,64,(2048,64))],1)
    rows=[]
    for model in ['hierarchical','fixed-nominal','variance-matched','oracle-scale']:
        for after,before in [('views','initial'),('probes','initial'),('probes','views')]:
            for metric in ['rmse_mm','hidden_rmse_mm','changed_load_rmse_mm','material_relative_rmse','force_rmse_N','coverage90','material_coverage90','force_coverage90','material_interval_relative_width']:
                difference=np.array([next(r[metric] for r in d['rows'] if r['model']==model and r['condition']==after)-next(r[metric] for r in d['rows'] if r['model']==model and r['condition']==before) for d in docs])
                ci=np.quantile(difference[indices].mean(1),[.025,.975])
                rows.append(dict(model=model,after=after,before=before,metric=metric,
                    mean_difference=float(difference.mean()),paired_bootstrap95=ci.tolist()))
    comparison=dict(cases=128,method='2048pairedcasebootstrap within each fixedgeometry,seed63001; after-minus-before; all predeclared conditions retained.',rows=rows)
    (REPORT/'condition-comparisons.json').write_text(json.dumps(comparison,indent=2)+'\n')
    figures();verify_protocol()
    (REPORT/'finalizer.json').write_text(json.dumps(dict(completed=True,protocol_sha256=sha,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Reports/figures assembled; separate rendered-PDF visual QA still required.'),indent=2)+'\n')
    print('Surface128case reports and figures complete',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--wait-seconds',type=int,default=0);args=ap.parse_args();main(args.wait_seconds)
