"""Development-only volume-consistent pose variant of frozen v12 rollout.

This is a reduced split model: inherited FEM predicts shape while a rigid pose
integrator handles free motion and static support. Contact impulses do NOT feed
back into the elastic shape solve. It is not a coupled deformable IPC method.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.external_benchmark.data import rigid_map
from scripts.contact_benchmark_v12.object_frame import estimate_attachment_state
from scipy.sparse import coo_matrix

def unit_volume_mass(rest,elements):
    rest=np.asarray(rest,float);elements=np.asarray(elements,int)
    D=np.transpose(rest[elements[:,1:]]-rest[elements[:,0,None]],(0,2,1))
    volume=np.abs(np.linalg.det(D))/6
    if not np.isfinite(volume).all() or np.any(volume<=0):raise ValueError("Invalid inherited rest tetrahedra")
    local=(volume/20)[:,None,None]*(np.eye(4)+np.ones((4,4)))
    row=np.broadcast_to(elements[:,:,None],local.shape);col=np.broadcast_to(elements[:,None,:],local.shape)
    matrix=coo_matrix((local.ravel(),(row.ravel(),col.ravel())),shape=(len(rest),len(rest))).tocsr()
    return matrix/float(matrix.sum())

def center_inertia(points,matrix):
    weights=np.asarray(matrix.sum(axis=1)).ravel();center=weights@points
    local=points-center;second=local.T@(matrix@local)
    return center,np.trace(second)*np.eye(3)-second


def assimilate_nodes(data,prediction,frame):
    """Rigidly align predicted nodes to SAME permitted sparse observed nodes."""
    valid=data['observed_mask'][frame]&np.isfinite(data['observed'][frame]).all(1)
    projected=data['observed_initial']+data['observation_matrix']@(prediction-data['nodes'])
    if valid.sum()<3:raise ValueError('Insufficient observed pose')
    R,b=rigid_map(projected[valid],data['observed'][frame,valid])
    return prediction@R+b

def integrate_pose(points,velocity,omega,dt,scene,matrix,friction=.5,substeps=10):
    """Unit-mass continuous P1 volume moments, gravity and sequential contact impulses.

    Inertia integrates the current P1 shape using rest-tetrahedral mass. Zero restitution, Coulomb impulse
    cap, static supports. Projection removes residual vertical penetration but
    introduces a disclosed position correction. No collision-free guarantee.
    """
    points=np.array(points,float,copy=True);v=np.array(velocity,float,copy=True);w=np.array(omega,float,copy=True)
    h=dt/substeps;events=0;max_pen=0.
    for _ in range(substeps):
        previous=points.copy();center,_=center_inertia(points,matrix);local=points-center
        v[2]-=9.81*h;rotation=Rotation.from_rotvec(w*h).as_matrix()
        points=local@rotation.T+center+v*h
        query=scene.query_surface(points,max_height=previous[:,2]+.002)
        gap=query['gap'];hit=np.flatnonzero(gap<0)
        if not len(hit):continue
        events+=1;max_pen=max(max_pen,float(-gap[hit].min()))
        center,inertia=center_inertia(points,matrix);r=points-center
        invI=np.linalg.pinv(inertia,rcond=1e-8)
        # A small contact manifold avoids applying a full impulse once per mesh
        # node and makes behavior less sensitive to sampling density.
        selected=hit[np.argsort(gap[hit])[:4]]
        for idx in selected:
            n=query['normal'][idx];arm=r[idx];at=v+np.cross(w,arm);vn=float(at@n)
            if vn>=0:continue
            cross=np.cross(arm,n);denom=1.+cross@invI@cross
            jn=-vn/max(denom,1e-12);impulse=jn*n
            v+=impulse;w+=invI@np.cross(arm,impulse)
            at=v+np.cross(w,arm);tangent=at-(at@n)*n;speed=np.linalg.norm(tangent)
            if speed>1e-10:
                direction=tangent/speed;cross_t=np.cross(arm,direction)
                jt=min(friction*jn,speed/max(1.+cross_t@invI@cross_t,1e-12));impulse=-jt*direction
                v+=impulse;w+=invI@np.cross(arm,impulse)
        points[:,2]+=float(-gap[hit].min())
    return points,v,w,dict(contact_substeps=events,max_preprojection_penetration_m=max_pen)

def rollout(data,base,scene,elements,fit_end=None,detect_detachment=True,friction=.5,substeps=80):
    """Use permitted prefix state, release commands and initial scene geometry."""
    if "evaluation" in data:raise ValueError("Future target payload forbidden")
    matrix=unit_volume_mass(data["nodes"],elements)
    center=lambda x: center_inertia(x,matrix)[0]
    end=int(data['train_end'] if fit_end is None else fit_end);out=np.asarray(base).copy()
    if end<2 or end>len(data['observed']):raise ValueError('Invalid prefix')
    state=estimate_attachment_state(data,end);detached=detect_detachment and state['state']=='detached_stationary'
    last=end-1;points=assimilate_nodes(data,base[last],last)
    prev=assimilate_nodes(data,base[last-1],last-1)
    dt=float(data['times'][last]-data['times'][last-1]);v=(center(points)-center(prev))/dt
    R,_=rigid_map(prev,points);omega=Rotation.from_matrix(R.T).as_rotvec()/dt
    if detached:v[:]=0.;omega[:]=0.
    free=detached or not bool(np.sum(data['attachment_active'][last])>0)
    diagnostics=[]
    for t in range(end,len(base)):
        dt=float(data['times'][t]-data['times'][t-1])
        release=bool(data['dataset']=='SoftVTBench' and t>data['grasp'] and data['gripper_command'][t]>0)
        if free or release:
            # Shape at separation remains in this reduced contact model. It is
            # not silently described as material relaxation/contact deformation.
            points,v,omega,diag=integrate_pose(points,v,omega,dt,scene,matrix,friction,substeps);free=True
            diagnostics.append(dict(frame=t,**diag));out[t]=points
        else:
            # Kinematic held frames use a consistent base-to-base derivative;
            # do not turn a prefix assimilation offset into spurious velocity.
            previous=base[t-1];points=base[t].copy();v=(center(points)-center(previous))/dt
            R,_=rigid_map(previous,points);omega=Rotation.from_matrix(R.T).as_rotvec()/dt
            out[t]=points
    return out,dict(model='volume-consistent inertia / inherited rigid contact pose; shape frozen at separation',fit_end=end,attachment_state=state,detect_detachment=detect_detachment,friction=friction,substeps=substeps,diagnostics=diagnostics,scene=scene.metadata())
