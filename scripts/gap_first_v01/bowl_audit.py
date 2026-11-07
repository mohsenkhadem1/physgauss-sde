"""Independent geometric observation/penetration checks on saved bowl states."""
import argparse,json
from pathlib import Path
import numpy as np
from .bowl_physics import build,command,triangle_quadrature
from .bowl_campaign import MARKERS,common_nodes,nearest_times

def overhead_visible(x,faces,marker,pusher_center,pusher_radius=.018,bowl_rotation=None):
    p=x[marker];tri=x[faces]
    a=tri[:,0,:2];v=tri[:,1,:2]-a;w=tri[:,2,:2]-a;r=p[:2]-a
    det=v[:,0]*w[:,1]-v[:,1]*w[:,0];valid=abs(det)>1e-15
    u=np.zeros(len(det));b=np.zeros(len(det))
    u[valid]=(r[valid,0]*w[valid,1]-r[valid,1]*w[valid,0])/det[valid]
    b[valid]=(v[valid,0]*r[valid,1]-v[valid,1]*r[valid,0])/det[valid]
    inside=valid&(u>=-1e-9)&(b>=-1e-9)&(u+b<=1+1e-9)
    height=tri[:,0,2]+u*(tri[:,1,2]-tri[:,0,2])+b*(tri[:,2,2]-tri[:,0,2])
    object_occludes=bool(np.any(inside&(height>p[2]+1e-8)))
    d2=np.sum((p[:2]-pusher_center[:2])**2)
    tool_occludes=d2<pusher_radius**2 and pusher_center[2]+np.sqrt(max(pusher_radius**2-d2,0))>p[2]+1e-8
    radius2=np.sum(p[:2]**2)
    bowl_occludes=radius2>.09**2
    if not bowl_occludes:
        hit=np.array([p[0],p[1],np.sqrt(.09**2-radius2)])
        local=hit@(np.eye(3) if bowl_rotation is None else bowl_rotation)
        bowl_occludes=bool(local[2]<=0 and hit[2]+.09>p[2]+1e-8)
    return not(object_occludes or tool_occludes or bowl_occludes),dict(object=object_occludes,tool=bool(tool_occludes),bowl=bool(bowl_occludes))

def audit_file(path):
    r=dict(np.load(path));n=round(len(r['rest'])**(1/3))-1;ids=common_nodes(n)
    system=build(n=n);qids,qw,qa=triangle_quadrature(r['rest'],r['faces'],6)
    interaction=str(r['interaction']);failures=[];checks=0;maximum=0.;min_tool_clearance=np.inf
    times=[.08,.12]+([.94,1.2] if interaction.startswith('calibration') else [])
    for k in range(len(r['frames'])):
        states=command(max(0.,float(r['times'][k])-float(r['dt'])),interaction)
        for ci,c in enumerate(system.colliders):
            for key,value in states.get(c.name,{}).items():setattr(c,key,value)
            c.center=np.asarray(r['centers'][k,ci])
        c=system.colliders[1]
        if c.center[2]-.018<=.09:
            # Sufficient full-sphere containment condition while near/below rim.
            min_tool_clearance=min(min_tool_clearance,.09-np.linalg.norm(c.center-system.colliders[0].center)-.018)
        points=np.einsum('qi,qic->qc',qw,r['frames'][k][qids])
        for c in system.colliders:
            phi,_=c.query(np.concatenate([points,r['frames'][k]]))
            maximum=max(maximum,float(np.max(np.maximum(-phi,0))))
        if k in nearest_times(r,times):
            for marker in ids[MARKERS]:
                checks+=1;ok,reason=overhead_visible(r['frames'][k],r['faces'],marker,system.colliders[1].center,bowl_rotation=system.colliders[0].rotation)
                if not ok:failures.append(dict(frame=k,time=float(r['times'][k]),marker=int(marker),reason=reason))
    return dict(path=str(path),visibility_checks=checks,visibility_failures=failures,
        dense_saved_surface_penetration_mm=maximum*1000,
        pusher_sufficient_clearance_mm=(min_tool_clearance*1000 if np.isfinite(min_tool_clearance) else None),
        integrator_min_j=float(r['min_j']),integrator_contact_penetration_mm=float(r['max_penetration_m']*1000))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('paths',nargs='+');p.add_argument('--output',default='reports/gap-first-v01/bowl/geometry-audit.json');a=p.parse_args()
    rows=[audit_file(Path(path)) for path in a.paths]
    Path(a.output).write_text(json.dumps(dict(files=rows,total_visibility_checks=sum(r['visibility_checks'] for r in rows),total_visibility_failures=sum(len(r['visibility_failures']) for r in rows)),indent=2))
    print(a.output)
