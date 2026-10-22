"""Sufficient whole-element positive-J bound at recorded trajectory frames."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from experiments.campaign_static_p2_block import shape_gradients


def frame_bounds(rest,tets,elements,positions):
    X=np.asarray(rest);T=np.asarray(tets);E=np.asarray(elements);Q=np.asarray(positions)
    if X.ndim!=2 or X.shape[1]!=3 or T.shape!=(len(E),4) or E.shape[1]!=10 or Q.ndim!=3 or Q.shape[1:]!=X.shape:raise ValueError('Compatible P2 nodal trajectory required')
    corners=X[T];Dm=np.swapaxes(corners[:,1:]-corners[:,:1],-1,-2)
    if not np.isfinite(Q).all() or np.any(np.linalg.det(Dm)<=0):raise ValueError('Finite states and positively oriented reference tets required')
    inverse=np.linalg.inv(Dm);g0=np.concatenate([-inverse.sum(1)[:,None],inverse],1);grad=[shape_gradients(b[None],g0) for b in np.eye(4)];rows=[]
    for k,q in enumerate(Q):
        ue=(q-X)[E];norm=0.
        for g in grad:
            G=np.einsum('eai,eaj->eij',ue,g);norm=max(norm,float(np.linalg.norm(G,ord=2,axis=(-2,-1)).max()))
        rows.append(dict(frame_index=k,maximum_vertex_displacement_gradient_spectral_norm=norm,sufficient_whole_element_minimum_J_bound=(1-norm)**3 if norm<1 else None,positive_whole_element_certificate=norm<1))
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('capture');p.add_argument('output');a=p.parse_args();source=Path(a.capture);target=Path(a.output)
    if target.exists():raise FileExistsError('Preserve diagnosis')
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    with np.load(source) as z:rows=frame_bounds(z['rest_m'],z['tets'],z['elements'],z['positions_m']);times=z['times_s'].tolist()
    result=dict(source=dict(path=str(source),sha256=digest),frames=rows,times_s=times,passed=all(r['positive_whole_element_certificate'] for r in rows),scope='All material points inside each P2 element at captured frames only; unsaved integration states/continuous time not certified',proof='P2 displacement gradient is affine in material coordinates; spectral norm bounded by maximum at 4 tet vertices. If norm<1, det(I+G)>=(1-norm)^3>0.',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(passed=result['passed'],frames=len(rows),minimum_sufficient_bound=min(r['sufficient_whole_element_minimum_J_bound'] or 0 for r in rows))))
