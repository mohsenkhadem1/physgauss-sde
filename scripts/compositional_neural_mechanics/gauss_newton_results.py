"""Additive reporting of the fixed optimizer-only development comparison."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'reports/compositional-neural-mechanics-v16';OUT=ROOT/'runs/compositional-neural-mechanics-v16/gauss-newton-development-comparison'

def main():
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    plan_path=R/'gauss-newton-development-comparison-plan.json';plan=json.loads(plan_path.read_text());status_path=R/'gauss-newton-development-status.json';status=json.loads(status_path.read_text());snapshot_path=OUT/'runtime-snapshot.json';snapshot=json.loads(snapshot_path.read_text())
    if status['plan_sha256']!=sha(plan_path) or snapshot['plan_sha256']!=sha(plan_path):raise ValueError('Plan binding')
    for mapping in (plan['source_sha256'],plan['artifact_sha256'],snapshot['original_sha256'],snapshot['snapshot_sha256']):
        if any(sha(ROOT/p)!=h for p,h in mapping.items()):raise ValueError('Changed pinned source/input/runtime snapshot')
    rows=[];inputs=[plan_path,status_path,snapshot_path]
    for i in (0,1):
        oldpath=OUT/f'original-world{i}.json';old=json.loads(oldpath.read_text());inputs.append(oldpath)
        newpath=R/f'gauss-newton-development-world{i}.json';new=json.loads(newpath.read_text()) if newpath.exists() else {'fits':[]}
        if newpath.exists():
            inputs.append(newpath)
            if new['runtime_snapshot_sha256']!=sha(snapshot_path) or new['original_report_sha256']!=sha(oldpath):raise ValueError('Fit report provenance')
        for name in ('analytic','learned'):
            a=next((x for x in old['fits'] if x['model']==name),None);b=next((x for x in new['fits'] if x['model']==name),None)
            row=dict(world=i,model=name,original={},gauss_newton={},new_report_present=b is not None)
            for key,fit in [('original',a),('gauss_newton',b)]:
                if fit is None:continue
                held=fit.get('heldout_errors',[])
                summary={k:fit.get(k) for k in ('loss','optimizer_success','incomplete','message','elapsed_seconds','optimality')}
                summary.update(evaluation_count=len(fit.get('evaluations',[])),heldout_errors=held,packed_latent_error=fit.get('packed_latent_error'))
                summary['all_two_forecasts_complete']=len(held)==2 and all('failure' not in h for h in held)
                if summary['all_two_forecasts_complete']:
                    summary['mean_heldout_errors']={k:float(np.mean([h[k] for h in held])) for k in held[0]}
                row[key]=summary
            if row['original'].get('all_two_forecasts_complete') and row['gauss_newton'].get('all_two_forecasts_complete'):
                row['heldout_difference_new_minus_original']={k:row['gauss_newton']['mean_heldout_errors'][k]-v for k,v in row['original']['mean_heldout_errors'].items()}
            rows.append(row)
    report=dict(scope='Two development scenes only; optimizer comparison, not a new benchmark or uncertainty result',input_sha256={str(p.relative_to(ROOT)):sha(p)for p in inputs},report_source_sha256=sha(Path(__file__)),status=status,rows=rows,
                optimizer_converged_original=sum(bool(x['original'].get('optimizer_success')) for x in rows),optimizer_converged_gauss_newton=sum(bool(x['gauss_newton'].get('optimizer_success')) for x in rows),
                budgets='Same900s/model ceiling; GN40 versus original160 evaluation ceiling. Different derivative work per evaluation, not equal compute per step.',
                caveat='Small prediction errors do not establish unique latent recovery. All failed or incomplete fits retained, no model/per-world selection.')
    path=R/'gauss-newton-development-results.json';path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
