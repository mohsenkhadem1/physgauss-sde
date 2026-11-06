"""Post-generation qualification of all320 declared truth histories; no exclusions."""
from pathlib import Path
import json,hashlib,time
import numpy as np
import torch
from .mechanics import LinearGaussianBody
ROOT=Path(__file__).resolve().parents[2];REPORT=ROOT/'reports/joint-field-surface-v04';RAW=ROOT/'runs/joint-field-surface-v04'
def main():
    torch.set_num_threads(1);cfg=json.loads((REPORT/'protocol.json').read_text());rows=[];hashes={};start=time.time()
    for shape in cfg['shapes']:
      for pop,count in cfg['case_counts'].items():
        p=RAW/shape/f'{pop}-truth.npz';truth=dict(np.load(p));hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
        assert len(truth['theta'])==count
        body=LinearGaussianBody(shape,(12,6,6) if pop=='offgrid-fine' else (8,4,4),device='cpu')
        for first in range(0,count,4):
          with torch.no_grad():
            unit=body.solve(truth['theta'][first:first+4])['unit_response'];force=torch.as_tensor(truth['force'][first:first+4].reshape(-1,8,2),dtype=torch.float64)
            u=torch.einsum('bncl,btl->btnc',unit,force);strain=body.strain(u);maximum=torch.linalg.eigvalsh(strain).abs().flatten(1).max(1).values
            grad=torch.einsum('bteai,eaj->bteij',u[:,:,body.elements,:],body.grad);jac=torch.linalg.det(torch.eye(3,dtype=grad.dtype)+grad).flatten(1).min(1).values
            maxu=u.norm(dim=-1).flatten(1).max(1).values
            for j in range(len(unit)):rows.append(dict(shape=shape,population=pop,case=first+j,maximum_absolute_principal_strain=float(maximum[j]),minimum_deformation_determinant=float(jac[j]),maximum_nodal_displacement_mm=float(maxu[j]*1000)))
        print(shape,pop,'qualified',count,flush=True)
    values=np.array([r['maximum_absolute_principal_strain'] for r in rows]);out=dict(complete=len(rows)==320,cases=len(rows),rows=rows,maximum_absolute_principal_strain=float(values.max()),median_maximum_principal_strain=float(np.median(values)),p95_maximum_principal_strain=float(np.quantile(values,.95)),cases_above_5percent=int(sum(values>.05)),cases_above_10percent=int(sum(values>.1)),minimum_deformation_determinant=min(r['minimum_deformation_determinant'] for r in rows),truth_sha256=hashes,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),seconds=time.time()-start,scope='All original truth states, no filtering or retuning. Posthoc mechanical qualification, not evidence of nonlinear physical accuracy.')
    (REPORT/'truth-strain-audit.json').write_text(json.dumps(out,indent=2));print({k:v for k,v in out.items() if k not in ['rows','truth_sha256']},flush=True)
if __name__=='__main__':main()
