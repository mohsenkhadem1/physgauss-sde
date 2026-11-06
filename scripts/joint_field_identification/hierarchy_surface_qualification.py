"""Evaluation-only strain audit of every fresh hierarchy truth; no filtering."""
from pathlib import Path
import hashlib,json,time
import numpy as np
import torch
from .hierarchy_surface import BASE,REPORT,ROOT,verify_protocol
from .mechanics import LinearGaussianBody


def main():
    torch.set_num_threads(1);_,sha=verify_protocol();rows=[];began=time.time();hashes={}
    for shape in ['block','step']:
        source=BASE/shape/'truth.npz';truth=dict(np.load(source));assert str(truth['protocol_sha256'])==sha
        hashes[str(source.relative_to(ROOT))]=hashlib.sha256(source.read_bytes()).hexdigest()
        body=LinearGaussianBody(shape,(8,4,4),device='cpu')
        for start in range(0,64,4):
            with torch.no_grad():
                unit=body.solve(truth['theta'][start:start+4])['unit_response']
                force=torch.as_tensor(truth['force'][start:start+4].reshape(-1,8,2),dtype=torch.float64)
                displacement=torch.einsum('bncl,btl->btnc',unit,force)
                strain=body.strain(displacement)
                maximum=torch.linalg.eigvalsh(strain).abs().flatten(1).max(1).values
                grad=torch.einsum('bteai,eaj->bteij',displacement[:,:,body.elements,:],body.grad)
                minJ=torch.linalg.det(torch.eye(3,dtype=grad.dtype)+grad).flatten(1).min(1).values
                maxu=displacement.norm(dim=-1).flatten(1).max(1).values
                for j in range(len(unit)):
                    rows.append(dict(shape=shape,case=start+j,tau=float(truth['tau'][start+j]),
                        maximum_absolute_principal_strain=float(maximum[j]),
                        minimum_deformation_determinant=float(minJ[j]),
                        maximum_nodal_displacement_mm=float(maxu[j]*1000),
                        maximum_absolute_force_mode_N=float(force[j].abs().max())))
        print('Qualified',shape,flush=True)
    verify_protocol();strain=np.array([r['maximum_absolute_principal_strain'] for r in rows])
    report=dict(protocol_sha256=sha,cases=128,rows=rows,
        maximum_absolute_principal_strain=float(strain.max()),
        median_maximum_principal_strain=float(np.median(strain)),p95_maximum_principal_strain=float(np.quantile(strain,.95)),
        cases_above_5percent=int((strain>.05).sum()),cases_above_10percent=int((strain>.10).sum()),
        minimum_deformation_determinant=min(r['minimum_deformation_determinant'] for r in rows),
        seconds=time.time()-began,truth_sha256=hashes,
        source_sha256={str(Path(__file__).relative_to(ROOT)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        scope='All128actual hierarchy truth force histories on declared coarse linear FEM. Evaluation-only; no case discarded/replaced and no protocol changed. Positive determinant or modest strain is not a nonlinear continuum accuracy certificate.')
    (REPORT/'truth-strain-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print({k:v for k,v in report.items() if k not in ['rows','truth_sha256','source_sha256']},flush=True)


if __name__=='__main__':main()
