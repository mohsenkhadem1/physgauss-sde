"""Method-independent orientation of a fixed material-grid tetrahedralization.

All methods first sample their physical state onto the same k^3 control grid.
Signed tetrahedron volume ratios detect sampled folds; they do not certify that
the continuous deformation is globally injective or that contact is valid.
"""
import numpy as np
from scripts.complex_scenes_v18.appearance_controls import controls


def ratios(positions,k):
    positions=np.asarray(positions)
    if positions.ndim!=4 or positions.shape[2:]!=(k**3,3):raise ValueError('Common grid shape required')
    index=np.arange(k**3).reshape(k,k,k);corners=np.array([[i,j,l]for i in (0,1)for j in (0,1)for l in (0,1)])
    cells=np.stack([index[i:i+k-1,j:j+k-1,l:l+k-1].ravel()for i,j,l in corners],-1)
    tetra=np.array([[0,1,3,7],[0,3,2,7],[0,2,6,7],[0,6,4,7],[0,4,5,7],[0,5,1,7]])
    ids=cells[:,tetra].reshape(-1,4)
    q=np.stack(np.meshgrid(*[np.linspace(0,1,k)]*3,indexing='ij'),-1).reshape(-1,3)
    rest=q*np.array([.12,.06,.06]);r=rest[ids]
    denom=np.linalg.det(np.stack([r[:,i]-r[:,0]for i in (1,2,3)],-1))
    x=positions[:,:,ids];det=np.linalg.det(np.stack([x[:,:,:,i]-x[:,:,:,0]for i in (1,2,3)],-1))
    return det/denom


def audit(folder,k=7):
    times,positions=controls(folder,k=k);volume=ratios(positions,k);bad=volume<=0
    return dict(scope='Common material-grid signed tetrahedron volume ratios; sampled fold diagnostic, not proof of continuous physical validity',control_grid_per_axis=k,tetrahedra_per_body=volume.shape[-1],minimum_volume_ratio=float(volume.min()),frames_with_nonpositive_volume=int(bad.any(axis=(1,2)).sum()),fraction_nonpositive_samples=float(bad.mean()),first_nonpositive_time_s=float(times[np.flatnonzero(bad.any(axis=(1,2)))[0]])if bad.any()else None)
