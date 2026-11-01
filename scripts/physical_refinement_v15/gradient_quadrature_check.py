"""Compare saved continuous-gradient quadratures without rereading observations."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'runs/physical-refinement-v15/development/posterior-quadrature'


def main():
    p=argparse.ArgumentParser();p.add_argument('--coarse',type=int,default=5);p.add_argument('--fine',type=int,default=9);a=p.parse_args()
    records=[];evidence={}
    for count in (a.coarse,a.fine):
        folder=BASE/f'g{count}-s49-97'
        meta=json.loads((folder/'s97.json').read_text())
        with np.load(folder/'s97.npz') as data:mean=data['mean'].copy();sd=data['sd'].copy()
        records.append((meta,mean,sd))
        for f in ('s97.json','s97.npz'):
            path=folder/f;evidence[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    (x,xm,xs),(y,ym,ys)=records
    identity=lambda r:{k:v for k,v in r.items() if k not in ('scale','log_evidence')}
    if list(map(identity,x['rows']))!=list(map(identity,y['rows'])) or x['evidence_sha256']!=y['evidence_sha256']:
        raise ValueError('Must compare identical observed histories')
    d=abs(xm-ym).reshape(-1,3,249);s=abs(xs-ys).reshape(-1,3,249);r=ys.reshape(-1,3,249)
    checks=dict(shape_mean=bool((d[:,:,:243]<=.00005).all()),shape_sd=bool((s[:,:,:243]<=.00001+.05*r[:,:,:243]).all()),
                force_mean=bool((d[:,:,243:246]<=.001).all()),torque_mean=bool((d[:,:,246:]<=.0001).all()),
                force_sd=bool((s[:,:,243:246]<=.0001+.05*r[:,:,243:246]).all()),
                torque_sd=bool((s[:,:,246:]<=.00001+.05*r[:,:,246:]).all()),
                log_evidence=bool(max(abs(u['log_evidence']-v['log_evidence']) for u,v in zip(x['rows'],y['rows']))<=.01),
                scale_mean=bool(max(abs(u['scale']['mean']-v['scale']['mean']) for u,v in zip(x['rows'],y['rows']))<=.005),
                scale_sd=bool(max(abs(u['scale']['sd']-v['scale']['sd']) for u,v in zip(x['rows'],y['rows']))<=.005),
                scale_quantiles=bool(max(np.max(abs(np.array(u['scale']['quantiles_05_50_95'])-v['scale']['quantiles_05_50_95'])) for u,v in zip(x['rows'],y['rows']))<=.01))
    result=dict(scope='Development posterior gradient-integration refinement, fixed97scale quadrature; not policy stability or physical accuracy evidence',
                gradient_counts=[a.coarse,a.fine],histories=len(x['rows']),checks=checks,passed=all(checks.values()),
                maximum_shape_mean_difference_mm=float(1000*d[:,:,:243].max()),maximum_force_mean_difference_N=float(d[:,:,243:246].max()),
                maximum_torque_mean_difference_Nm=float(d[:,:,246:].max()),
                maximum_log_evidence_difference=max(abs(u['log_evidence']-v['log_evidence']) for u,v in zip(x['rows'],y['rows'])),evidence_sha256=evidence)
    path=ROOT/'reports/physical-refinement-v15'/f'posterior-gradient-{a.coarse}-{a.fine}.json'
    if path.exists():raise ValueError('Do not overwrite prior refinement evidence')
    path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
