"""Read-only nonlinear strain and two-family response signal on n6 pilot states."""
from pathlib import Path
import hashlib,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from numpy.polynomial.legendre import leggauss
from scripts.phystwin.tensor_fixture import basis
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'runs/compositional-neural-mechanics-v16/mechanics-reference-fine'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def statistics(rest,u):
    cells=np.array([24,6,6]);shape=tuple(2*cells+1)
    cell=np.stack(np.meshgrid(*[np.arange(k) for k in cells],indexing='ij'),-1).reshape(-1,3)
    offset=np.stack(np.meshgrid(*[np.arange(3)]*3,indexing='ij'),-1).reshape(-1,3)
    elements=np.ravel_multi_index((2*cell[:,None]+offset[None]).transpose(2,0,1),shape)
    q,w=leggauss(3);quad=np.stack(np.meshgrid(q,q,q,indexing='ij'),-1).reshape(-1,3);_,G=basis(quad)
    jac=np.einsum('eai,qaj->qeij',rest[elements],G);grad=np.einsum('qaJ,qeJi->qeai',G,np.linalg.inv(jac))
    F=np.eye(3)+np.einsum('eai,qeaj->qeij',u[elements],grad);J=np.linalg.det(F)
    q1=J**(-2/3)*(F*F).sum((-1,-2))-3
    weights=np.linalg.det(jac)*np.prod(np.stack(np.meshgrid(w,w,w,indexing='ij'),-1),axis=-1).ravel()[:,None];weights/=weights.sum()
    stretch=np.linalg.svd(F,compute_uv=False)
    return dict(q1=dict(min=float(q1.min()),max=float(q1.max()),volume_weighted_mean=float((weights*q1).sum()),quadrature_unweighted_percentiles50_95_99=np.quantile(q1,[.5,.95,.99]).tolist()),
                J=dict(min=float(J.min()),max=float(J.max()),volume_weighted_mean=float((weights*J).sum())),principal_stretch_min_max=[float(stretch.min()),float(stretch.max())])


def main():
    output=ROOT/'reports/compositional-neural-mechanics-v16/family-signal-qualification.json'
    if output.exists():raise ValueError('Preserve previous diagnostic')
    evidence={str(Path(__file__).relative_to(ROOT)):sha(__file__),'scripts/phystwin/tensor_fixture.py':sha(ROOT/'scripts/phystwin/tensor_fixture.py')};rows=[];strains=[]
    for geometry in ('rectangular','tapered_arch'):
        report_path=BASE/geometry/'report.json';report=json.loads(report_path.read_text());evidence[str(report_path.relative_to(ROOT))]=sha(report_path)
        for action in ('stretch','opposed_twist','right_bend'):
            fields=[]
            for family in ('mooney_rivlin','yeoh'):
                path=BASE/geometry/family/f'{action}.npz';saved=next(r for r in report['rows'] if r['family']==family and r['action']==action)
                if sha(path)!=saved['sha256']:raise ValueError('Prior n6 state changed')
                evidence[str(path.relative_to(ROOT))]=sha(path)
                with np.load(path) as data:values={k:data[k].copy() for k in data.files}
                fields.append(values);strains.append(dict(geometry=geometry,family=family,action=action,**statistics(values['rest'],values['displacement'])))
            a,b=fields
            rows.append(dict(geometry=geometry,action=action,field_difference_rms_mm=float(1000*np.sqrt(np.mean((a['field']-b['field'])**2))),
                             force_difference_N=np.linalg.norm(a['forces_N']-b['forces_N'],axis=1).tolist(),torque_difference_Nm=np.linalg.norm(a['torques_Nm']-b['torques_Nm'],axis=1).tolist()))
    output.write_text(json.dumps(dict(scope=__doc__,source_evidence_sha256=evidence,n6_strain_statistics=strains,family_response_differences=rows,
        interpretation='Visualfamily differences oftencomparableton4/n6meshdiscrepancy; stretchreactionseparationmuchlarger. NoidentifiabilityclaimfromRMSalone; nonlinear-support range needsseparatepilot.'),indent=2)+'\n')
    print(output)
if __name__=='__main__':main()
