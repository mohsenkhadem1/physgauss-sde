"""Separate v06 actual-deformed visibility and all-history strain audit."""
from pathlib import Path
import json,time
import numpy as np
import torch
from . import temporal_hierarchy as campaign
from . import surface_visibility as visibility
from .mechanics import LinearGaussianBody

def main():
    torch.set_num_threads(1);protocol,ph=campaign.verify();reports=[];rows=[];hashes={};start=time.time()
    for shape in ['block','step']:
        path=campaign.BASE/shape/'truth.npz';truth=dict(np.load(path));assert len(truth['theta'])==32;assert str(truth['protocol_sha256'])==ph
        hashes[str(path.relative_to(campaign.ROOT))]=visibility.sha(path)
        reports.append(visibility.audit_truth(path,shape,'temporal-v06',campaign.REPORT/shape/'visibility.json'))
        body=LinearGaussianBody(shape,(8,4,4))
        for case,theta in enumerate(truth['theta']):
            with torch.no_grad():
                out=body.solve(theta);unit=out['unit_response'][0];probes=.08*out['probe_response'][0].permute(2,0,1)
                force=torch.as_tensor(truth['force'][case].reshape(8,2));u=torch.einsum('ncl,tl->tnc',unit,force)
                record=dict(shape=shape,case=case,tau=float(truth['tau'][case]),maximum_absolute_force_N=float(force.abs().max()))
                for name,disp in [('original',u),('known_probes',probes)]:
                    strain=body.strain(disp);eigen=np.linalg.eigvalsh(strain.numpy());grad=torch.einsum('teai,eaj->teij',disp[:,body.elements,:],body.grad).numpy();det=np.linalg.det(np.eye(3)+grad)
                    record[name]=dict(maximum_absolute_principal_strain=float(abs(eigen).max()),minimum_deformation_determinant=float(det.min()),maximum_nodal_displacement_mm=float(disp.norm(dim=-1).max()*1000))
                rows.append(record)
    campaign.verify();summary=dict(complete=True,passed=all(r['failed_checks']==0 for r in reports),cases=64,checks=sum(r['checks'] for r in reports),failed_cases=sum(r['failed_cases'] for r in reports),failed_checks=sum(r['failed_checks'] for r in reports),protocol_sha256=ph,truth_sha256=hashes,scope='Separate v06all64cases,8originaltimes and6knownprobes; suppliedmaterialmarkers, notRGBtracking. Existingv04/v05audit unchanged.')
    (campaign.REPORT/'visibility-summary.json').write_text(json.dumps(summary,indent=2))
    qualification=dict(complete=True,cases=64,protocol_sha256=ph,truth_sha256=hashes,rows=rows,seconds=time.time()-start,source_sha256=visibility.sha(__file__),scope='Alloriginaltruthstates and6knownprobestates, no exclusions/retuning. Linear small-strain FEM qualification doesnotcertify nonlinearaccuracy.')
    for mode in ['original','known_probes']:
        vals=np.array([r[mode]['maximum_absolute_principal_strain'] for r in rows]);qualification[mode]=dict(maximum_absolute_principal_strain=float(vals.max()),p95_maximum_principal_strain=float(np.quantile(vals,.95)),median_maximum_principal_strain=float(np.median(vals)),cases_above_5percent=int((vals>.05).sum()),cases_above_10percent=int((vals>.1).sum()),minimum_deformation_determinant=min(r[mode]['minimum_deformation_determinant'] for r in rows))
    (campaign.REPORT/'truth-strain-audit.json').write_text(json.dumps(qualification,indent=2));print(summary,flush=True);print({k:v for k,v in qualification.items() if k not in ['rows','truth_sha256']},flush=True)
if __name__=='__main__':main()
