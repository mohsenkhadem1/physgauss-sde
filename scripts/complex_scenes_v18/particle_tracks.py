"""Fixed trilinear material-coordinate binding for native MPM visible tracks.

Particles occupy cell centers; surface queries extrapolate at most half a particle
spacing. Geometry-only weights are fixed before observing motion or fitting.
"""
import numpy as np


def binding(queries,counts=(12,6,6)):
    q=np.asarray(queries,float);counts=np.asarray(counts,int)
    if q.ndim!=2 or q.shape[1]!=3 or np.any(q<0)or np.any(q>1)or np.any(counts<2):raise ValueError('Unit material coordinates and valid grid required')
    s=q*counts-.5;lo=np.minimum(np.maximum(np.floor(s).astype(int),0),counts-2);f=s-lo
    corners=np.array([[i,j,k]for i in (0,1)for j in (0,1)for k in (0,1)])
    indices=np.ravel_multi_index((lo[:,None]+corners).transpose(2,0,1),tuple(counts))
    weights=np.prod(np.where(corners[None]==1,f[:,None],1-f[:,None]),axis=-1)
    return indices.astype(np.int32),weights.astype(np.float32)
