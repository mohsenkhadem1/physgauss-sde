"""Common sampled-surface physical diagnostics, independent of model internals.
These do not certify continuum conservation, interior Jacobians or contact impulses.
"""
import numpy as np
from scripts.compositional_dynamics_v17.forecast_metrics import reference_surface_weights


def closed_triangles(queries):
    q=np.asarray(queries,float);reference_surface_weights(q,[1,1,1]);triangles=[]
    for axis in range(3):
        other=[j for j in range(3)if j!=axis]
        for side in (0.,1.):
            grid=np.empty((5,5),int)
            for i,a in enumerate(np.linspace(0,1,5)):
                for j,b in enumerate(np.linspace(0,1,5)):
                    point=np.zeros(3);point[axis]=side;point[other]=[a,b]
                    grid[i,j]=np.flatnonzero(np.all(q==point,axis=1))[0]
            for i in range(4):
                for j in range(4):
                    a,b,c,d=grid[i,j],grid[i+1,j],grid[i+1,j+1],grid[i,j+1]
                    for tri in ([a,b,c],[a,c,d]):
                        x=q[tri];normal=np.cross(x[1]-x[0],x[2]-x[0])
                        if normal[axis]*(2*side-1)<0:tri=[tri[0],tri[2],tri[1]]
                        triangles.append(tri)
    return np.asarray(triangles)


def diagnostics(times,positions,queries,dimensions,floor_z,contact_tolerance_m=1e-6):
    t=np.asarray(times,float);x=np.asarray(positions,float)
    if x.shape!=(len(t),98,3)or not np.isfinite(x).all()or not np.isfinite(t).all()or np.any(np.diff(t)<=0):raise ValueError('Finite matched98point frames required')
    if not len(t):return dict(frames=0,first_contact_s=None,first_separation_s=None,first_rebound_apex_s=None,first_rebound_apex_surface_centroid_z_m=None)
    triangles=closed_triangles(queries);w=reference_surface_weights(queries,dimensions);w/=w.sum();center=np.einsum('tnc,n->tc',x,w)
    local=x-center[:,None,:];v=local[:,triangles]
    volume=np.einsum('tni,tni->t',v[:,:,0],np.cross(v[:,:,1],v[:,:,2]))/6
    gap=x[:,:,2].min(axis=1)-float(floor_z);contact=gap<=contact_tolerance_m;hits=np.flatnonzero(contact)
    first=int(hits[0])if len(hits)else None;sep=None;apex=None
    if first is not None:
        separation=np.flatnonzero(~contact[first+1:])
        if len(separation):
            sep=first+1+int(separation[0]);nextcontact=np.flatnonzero(contact[sep+1:]);end=sep+1+int(nextcontact[0])if len(nextcontact)else len(t)
            candidate=sep+int(np.argmax(center[sep:end,2]))
            # An endpoint maximum is censored, not an observed rebound apex.
            if sep<candidate<end-1:apex=candidate
    return dict(frames=len(t),signed_closed_surface_volume_m3=volume.tolist(),volume_over_rest=(volume/np.prod(dimensions)).tolist(),surface_centroid_z_m=center[:,2].tolist(),minimum_sampled_gap_m=gap.tolist(),maximum_sampled_penetration_m=float(max(0,-gap.min())),first_contact_s=float(t[first])if first is not None else None,first_separation_s=float(t[sep])if sep is not None else None,first_rebound_apex_s=float(t[apex])if apex is not None else None,first_rebound_apex_surface_centroid_z_m=float(center[apex,2])if apex is not None else None,
                scope='Closed piecewise-linear98point surface; signed volume can miss interior folding. Penetration uses sampledvertices and may underestimate curved-surface penetration. Contact events use1um gap tolerance at recorded100Hz; no subframe interpolation. Surface centroid is not massCOM. Missing endpoint apex is censored.')


def compare_diagnostics(reference,prediction):
    n=prediction['frames'];out=dict(reference=reference,prediction=prediction,complete=n==reference['frames'])
    if n:
        out['volume_ratio_rmse']=float(np.sqrt(np.mean((np.array(reference['volume_over_rest'][:n])-prediction['volume_over_rest'])**2)))
    for key in ('first_contact_s','first_separation_s','first_rebound_apex_s','first_rebound_apex_surface_centroid_z_m'):
        a,b=reference.get(key),prediction.get(key);out[key+'_signed_error']=b-a if a is not None and b is not None else None
    return out
