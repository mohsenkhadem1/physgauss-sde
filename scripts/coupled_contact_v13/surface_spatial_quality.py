"""Frozen per-case spatial companion check; never replaces the primary target."""
import json
import numpy as np
from .surface_contact import box_surface_quadrature,tetrahedral_embedding
from .mesh_diagnostic import shape


def inspect(base,object_id,scenario):
    reasons=[];arrays=[];limits=dict(world_rms_m=.001,shape_rms_m=.0005,body_center_coordinate_rms_m=.001,penetration_m=.001,min_j=.02)
    output=dict(qualified=False,reasons=reasons,common27=None,dense2678=None,body_center_coordinate_rms_m=None,limits=limits)
    for directory,label in [('reference','primary'),('reference-spatial','companion')]:
        path=base/directory/object_id/scenario
        try:
            status=json.loads((path/'complete.json').read_text())
            if not status['success']:raise ValueError('generation failed')
            d=status['diagnostics']
            if not np.isfinite(d['min_j']) or d['min_j']<=limits['min_j']:reasons.append(label+' spatial invalid element volume')
            if not np.isfinite(d['max_penetration_m']) or d['max_penetration_m']>limits['penetration_m']:reasons.append(label+' spatial penetration exceeds limit')
            with np.load(path/'trajectory.npz') as a:r={key:a[key].copy() for key in ('frames','rest','tets','times','centers','velocities','reactions')}
            if not all(np.isfinite(v).all() for v in r.values()):raise ValueError('nonfinite arrays')
            if r['frames'].ndim!=3 or r['frames'].shape[1:]!=r['rest'].shape or len(r['frames'])!=len(r['times']) or len(r['times'])<2 or np.any(np.diff(r['times'])<=0):raise ValueError('invalid trajectory/time shape')
            if r['centers'].ndim!=3 or r['centers'].shape[0]!=len(r['times']) or r['centers'].shape[-1]!=3:raise ValueError('invalid body shape')
            if r['velocities'].shape!=r['frames'].shape or r['reactions'].shape!=r['centers'].shape:raise ValueError('invalid velocity/reaction shape')
            r['actual_dt']=float(d['dt'])
            if not np.isfinite(r['actual_dt']) or r['actual_dt']<=0:raise ValueError('invalid actual diagnostic dt')
            arrays.append(r)
        except (OSError,ValueError,KeyError) as exc:reasons.append(label+' spatial reference unavailable: '+str(exc))
    if len(arrays)!=2:
        reasons.append('Two spatial reference levels unavailable');return output
    a,b=arrays
    if a['rest'].shape==b['rest'].shape and np.allclose(a['rest'],b['rest'],atol=1e-12,rtol=0):
        reasons.append('Spatial companion mesh is identical to primary');return output
    if not np.isclose(a['actual_dt'],b['actual_dt'],atol=1e-15,rtol=0):
        reasons.append('Spatial primary and companion actual diagnostic dt differ');return output
    if a['times'].shape!=b['times'].shape or not np.allclose(a['times'],b['times'],atol=1e-10,rtol=0):
        reasons.append('Spatial reference time grids differ');return output
    if not np.allclose(a['rest'].min(0),b['rest'].min(0),atol=1e-12,rtol=0) or not np.allclose(a['rest'].max(0),b['rest'].max(0),atol=1e-12,rtol=0):
        reasons.append('Spatial rest domains differ');return output
    try:
        lo=a['rest'].min(0);hi=a['rest'].max(0)
        common=np.array(np.meshgrid(*[np.linspace(x,y,3) for x,y in zip(lo,hi)],indexing='ij')).reshape(3,-1).T
        dense,_=box_surface_quadrature(a['rest'],.003)
        if len(dense)!=2678:raise ValueError('Expected fixed2678 dense surface queries')
        for label,points in [('common27',common),('dense2678',dense)]:
            curves=[]
            for r in arrays:
                if label=='common27':
                    distance=np.linalg.norm(points[:,None]-r['rest'][None],axis=-1);ids=distance.argmin(1)
                    if np.any(distance[np.arange(27),ids]>1e-10):raise ValueError('Common27 nodes absent')
                    curves.append(r['frames'][:,ids])
                else:
                    ids,w=tetrahedral_embedding(r['rest'],r['tets'],points)
                    curves.append(np.einsum('qi,tqic->tqc',w,r['frames'][:,ids]))
            x,y=curves
            metric=dict(points=len(points),world_rms_m=float(np.sqrt(np.mean((x[1:]-y[1:])**2))),shape_rms_m=float(np.sqrt(np.mean([shape(i,j)**2 for i,j in zip(x[1:],y[1:])]))))
            output[label]=metric
            for key in ('world_rms_m','shape_rms_m'):
                if not np.isfinite(metric[key]) or metric[key]>limits[key]:reasons.append(label+' spatial '+key+' exceeds limit')
        if a['centers'].shape!=b['centers'].shape:raise ValueError('Spatial collider identities/shapes differ')
        rms=np.sqrt(np.mean((a['centers'][1:]-b['centers'][1:])**2,axis=(0,2)))
        output['body_center_coordinate_rms_m']=float(rms.max())
        if output['body_center_coordinate_rms_m']>limits['body_center_coordinate_rms_m']:reasons.append('Spatial body center difference exceeds limit')
    except (ValueError,IndexError,np.linalg.LinAlgError) as exc:reasons.append('Spatial comparison failed: '+str(exc))
    output['qualified']=not reasons
    return output


def augment(quality,base,object_id,scenario):
    spatial=inspect(base,object_id,scenario);result=dict(quality)
    result['qualified']=quality['qualified'] and spatial['qualified']
    result['reasons']=list(quality['reasons'])+spatial['reasons'];result['spatial_qualification']=spatial
    from .campaign import write
    write(base/'reference'/object_id/scenario/'spatial-quality.json',spatial)
    return result
