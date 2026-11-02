"""Read-only whole-body force/angular-wrench balance for completed zero-load pilots."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    out=ROOT/'reports/compositional-neural-mechanics-v16/nonlinear-wrench-balance-audit.json'
    if out.exists():raise ValueError('Preserve previous audit')
    evidence={str(Path(__file__).relative_to(ROOT)):sha(__file__)};rows=[]
    for name in ('nonlinear-range-development','crossed-soft-saturated-development'):
        folder=ROOT/'runs/compositional-neural-mechanics-v16'/name;manifest=json.loads((folder/'manifest.json').read_text());report=json.loads((folder/'report.json').read_text())
        if any(np.linalg.norm(s['gravity'])!=0 for s in manifest['development_plan']['cases']):raise ValueError('This zero-body-load audit does not omit nonzero gravity torque')
        for p in (folder/'manifest.json',folder/'report.json'):evidence[str(p.relative_to(ROOT))]=sha(p)
        for r in report['rows']:
            if r['state']!='complete':rows.append(dict(case=r['case'],model=r['model'],n=r['n'],action=r['action'],passed=False,reason='Failed solve retained'));continue
            p=ROOT/r['artifact']
            if sha(p)!=r['sha256']:raise ValueError('Completed state changed')
            evidence[str(p.relative_to(ROOT))]=sha(p)
            with np.load(p) as data:
                force=data['forces_N'];torque=data['torques_Nm'];origins=data['origins_world_m']
            f=float(np.linalg.norm(force.sum(0)));t=float(np.linalg.norm((torque+np.cross(origins,force)).sum(0)))
            rows.append(dict(case=r['case'],model=r['model'],n=r['n'],action=r['action'],net_force_N=f,net_world_torque_Nm=t,passed=f<=1e-7 and t<=1e-8))
    result=dict(scope=__doc__,source_evidence_sha256=evidence,rows=rows,all_gates_pass=all(r['passed'] for r in rows),
                maximum_net_force_N=max(r.get('net_force_N',0) for r in rows),maximum_net_world_torque_Nm=max(r.get('net_world_torque_Nm',0) for r in rows),
                limitation='Quasi-static prescribed-tool reactions, not movingrigidbody dynamics or contactvalidation')
    out.write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k not in ('rows','source_evidence_sha256')})
if __name__=='__main__':main()
