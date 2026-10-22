"""Read-only motion diagnostic on the completed14 development mechanics cases.

Positive exact polynomial quadrature; no physics, observations or object draws.
These selected corners are not a random population or sensor experiment.
"""
from pathlib import Path
import hashlib,json
import numpy as np
from experiments.campaign_p2_keast_quadrature import positive_keast24
from experiments.campaign_static_p2_block import shapes


def motion_components(reference,deformed,weights):
    X=np.asarray(reference,dtype=float);Y=np.asarray(deformed,dtype=float);w=np.asarray(weights,dtype=float)
    if X.ndim!=2 or X.shape[1]!=3 or Y.shape!=X.shape or w.shape!=(len(X),) or not all(np.isfinite(a).all() for a in (X,Y,w)) or np.any(w<=0):raise ValueError('Finite matched material samples and positive volume weights required')
    w=w/w.sum();cx=w@X;cy=w@Y
    U,_,V=np.linalg.svd(((X-cx)*w[:,None]).T@(Y-cy));Q=U@np.diag([1.,1.,np.linalg.det(U@V)])@V
    residual=Y-((X-cx)@Q+cy);motion=Y-X
    total=float(np.sqrt(np.sum(w[:,None]*motion**2)/3));nonrigid=float(np.sqrt(np.sum(w[:,None]*residual**2)/3))
    return dict(total_coordinate_motion_RMS_m=total,nonrigid_coordinate_RMS_m=nonrigid,nonrigid_to_total_RMS_ratio=None if total==0 else nonrigid/total)


def main():
    root=Path(__file__).resolve().parents[1];base=root/'runs/operator-dev/c38-static-motion-diagnostic-v02';base.mkdir(parents=True,exist_ok=False)
    matrix=root/'runs/operator-dev/c38-static-prequalification-v01/completed-summary.json';d=json.loads(matrix.read_text());assert d['completed'] and d['cases_completed']==14
    bary,weights=positive_keast24();N=shapes(bary);rows=[];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for entry in d['rows']:
        # Full localresult is bound by the completed matrix's case result hash.
        resultrecord=entry['result'];path=(root/resultrecord['path']).resolve()
        if not path.is_relative_to(root) or sha(path)!=resultrecord['sha256']:raise ValueError('Changed matrix-bound case result')
        state=json.loads(path.read_text())
        if state['case']!=entry['case'] or not state['completed']:raise ValueError('Changed declared completed case')
        row=state['meshes'][-1];capture=(root/row['capture']['path']).resolve()
        if not capture.is_relative_to(root):raise ValueError('Outside repository capture')
        if sha(capture)!=row['capture']['sha256']:raise ValueError('Changed finite-element field')
        with np.load(capture,allow_pickle=False) as z:X=z['rest_m'];q=z['positions_m'];e=z['elements'];t=z['tets']
        tet=X[t];vol=np.linalg.det(np.swapaxes(tet[:,1:]-tet[:,:1],-1,-2))/6
        if np.any(vol<=0):raise ValueError('Positive reference volumes required')
        ref=np.einsum('qa,eai->eqi',N,X[e]).reshape(-1,3);deformed=np.einsum('qa,eai->eqi',N,q[e]).reshape(-1,3);w=(vol[:,None]*weights[None]).ravel();m=motion_components(ref,deformed,w);difference=abs(m['nonrigid_coordinate_RMS_m']-row['diagnostics']['bulk_nonrigid_coordinate_RMS_m'])
        if difference>1e-12:raise ValueError('Independent rigid removal differs from existing diagnostic')
        rows.append(dict(case_id=entry['case']['id'],capture=row['capture'],result=dict(path=str(path.relative_to(root)),sha256=sha(path)),**m,original_bulk_RMS_reproduction_difference_m=difference,spatial_indicator_passed=state['passed']))
    out=dict(schema='C38-supported-motion-readonly-v01',completed=True,inputs=dict(completed_matrix=dict(path=str(matrix.relative_to(root)),sha256=sha(matrix)),source=dict(path=str(Path(__file__).relative_to(root)),sha256=sha(Path(__file__)))),method='Exact24positive polynomial-volume samples; proper weighted rigid registration, coordinate RMS /3. Ratio is RMS not fraction of energy.',rows=rows,nonrigid_to_total_RMS_ratio_range=[min(r['nonrigid_to_total_RMS_ratio'] for r in rows),max(r['nonrigid_to_total_RMS_ratio'] for r in rows)],limits=['14 fixed development corners/interiors, not random population','No dataset admission:7 of14 displacement indicators fail','Rigid removal does not establish identification or calibration','No new forward solves/observations/rendered images'])
    target=base/'summary.json';target.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ['completed','nonrigid_to_total_RMS_ratio_range']}))
if __name__=='__main__':main()
