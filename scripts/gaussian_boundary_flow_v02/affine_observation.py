"""Initial-geometry affine-complete local interpolation/extrapolation.

No observed motion, labels or evaluation targets enter. Correct inverse-distance
weights by the minimum Euclidean-norm change subject to sum(w)=1 and sum(w*X)=p.
Negative weights are allowed (necessary for points outside the reduced hull), never
clipped. Their L1/L2 norms disclose worst-case error and IID-noise amplification.
This is a fixed material observation map, not inferred contact or geometry truth.
"""
import numpy as np
from scipy.spatial import cKDTree


def affine_observation_matrix(nodes,points,*,neighbors=16,max_l1=8.,rtol=1e-12):
    nodes=np.asarray(nodes,dtype=float);points=np.asarray(points,dtype=float)
    if nodes.ndim!=2 or nodes.shape[1]!=3 or len(nodes)<1 or points.ndim!=2 or points.shape[1]!=3:
        raise ValueError('Finite Nx3 nodes and Px3 points required')
    if not np.isfinite(nodes).all() or not np.isfinite(points).all():raise ValueError('Finite geometry required')
    if not isinstance(neighbors,int) or neighbors<1:raise ValueError('Positive integer neighbors required')
    if not np.isfinite(max_l1) or max_l1<1 or not np.isfinite(rtol) or not 0<rtol<1:
        raise ValueError('Finite max_l1>=1 and 0<rtol<1 required')
    extent=float(np.linalg.norm(np.ptp(nodes,axis=0)))
    if extent==0:extent=1.
    tree=cKDTree(nodes);matrix=np.zeros((len(points),len(nodes)));rows=[]
    counts=sorted(set([min(neighbors*2**i,len(nodes)) for i in range(1+int(np.ceil(np.log2(max(1,len(nodes)/neighbors)))))] + [len(nodes)]))
    for point_index,point in enumerate(points):
        nearest_distance,nearest=tree.query(point,k=1)
        if nearest_distance<=32*np.finfo(float).eps*extent:
            matrix[point_index,nearest]=1.
            rows.append(dict(neighbors=1,rank=1,condition_nonzero=1.,constraint_error=0.,l1=1.,l2=1.,negative_weights=0,exact_node=True))
            continue
        accepted=None
        for count in counts:
            distance,ids=tree.query(point,k=count);distance=np.atleast_1d(distance);ids=np.atleast_1d(ids)
            delta=nodes[ids]-point;scale=max(float(np.linalg.norm(delta,axis=1).max()),extent*np.finfo(float).eps)
            C=np.vstack([np.ones(count),(delta/scale).T]);target=np.array([1.,0.,0.,0.])
            U,s,Vh=np.linalg.svd(C,full_matrices=False);keep=s>rtol*s[0]
            prior=1/np.maximum(distance,extent*1e-12)**2;prior/=prior.sum()
            correction=Vh[keep].T@((U[:,keep].T@(target-C@prior))/s[keep])
            weights=prior+correction
            error=float(np.max(abs(C@weights-target)));l1=float(abs(weights).sum())
            if error<=1e-10 and l1<=max_l1:
                accepted=(ids,weights,dict(neighbors=count,rank=int(keep.sum()),condition_nonzero=float(s[0]/s[keep][-1]),constraint_error=error,l1=l1,l2=float(np.linalg.norm(weights)),negative_weights=int((weights < -1e-14).sum()),exact_node=False));break
        if accepted is None:
            raise ValueError(f'Point {point_index} has no affine-complete map within max_l1={max_l1}; outside representable affine span or ill-conditioned/excessive extrapolation')
        ids,weights,diagnostic=accepted;matrix[point_index,ids]=weights;rows.append(diagnostic)
    diagnostic={'method':'Local inverse-distance prior plus minimum-norm affine correction','neighbors_requested':neighbors,'maximum_allowed_l1':max_l1,'negative_weights_allowed':True,'rows':rows,'maximum_row_l1':max((r['l1']for r in rows),default=0.),'maximum_row_l2':max((r['l2']for r in rows),default=0.),'maximum_geometry_error':float(np.max(abs(matrix@nodes-points))) if len(points) else 0.,'maximum_partition_error':float(np.max(abs(matrix.sum(1)-1)))if len(points)else 0.}
    return matrix,diagnostic
