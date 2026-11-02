"""Independent dense-seeded continuous face-gap minimization (not a certificate)."""
import numpy as np
from scipy.optimize import minimize
from scripts.compositional_neural_mechanics.contact_mechanics import RigidPlane,RigidSphere


def map_jacobian(geometry,q):
    x,y,z=q;L,H,D=geometry.dimensions;t=geometry.taper;w=1+t*(2*x-1)
    return np.array([[L,0,0],[2*t*H*(y-.5),H*w,0],[2*t*D*(z-.5)+4*geometry.arch*(1-2*x),0,D*w]])


def search_gaps(model,u,grid=65,seeds_per_face=8):
    axis=np.linspace(0,1,grid);uv=np.stack(np.meshgrid(axis,axis,indexing='ij'),-1).reshape(-1,2);out=[]
    for c in model.colliders:
        best=float('inf');bestq=None;densebest=float('inf');attempts=[]
        for faceaxis in range(3):
            others=[k for k in range(3)if k!=faceaxis]
            for side in (0.,1.):
                q=np.empty((len(uv),3));q[:,faceaxis]=side;q[:,others]=uv;x=model.geometry.map(q)+model.sample_material(u,q)
                gaps=c.signed_distance(model.engine.tensor(x)).detach().cpu().numpy();indices=np.argsort(gaps);densebest=min(densebest,float(gaps[indices[0]]));seeds=[]
                for i in indices:
                    if all(np.linalg.norm(uv[i]-v)>=.1 for v in seeds):seeds.append(uv[i].copy())
                    if len(seeds)>=seeds_per_face:break
                def objective(v):
                    query=np.empty(3);query[faceaxis]=side;query[others]=v
                    point=model.geometry.map(query)+model.sample_material(u,query[None])[0]
                    if isinstance(c,RigidPlane):normal=np.asarray(c.normal,float);normal=normal/np.linalg.norm(normal);gap=float((point-np.asarray(c.point))@normal)
                    elif isinstance(c,RigidSphere):d=point-np.asarray(c.center);length=np.linalg.norm(d);normal=d/length;gap=float(length-c.radius)
                    else:raise ValueError('Known plane/sphere only')
                    tangent=(np.eye(3)+model.sample_material(u,query[None],gradient=True)[0])@map_jacobian(model.geometry,query)
                    return gap,normal@tangent[:,others]
                # Include dense sample itself, so unsuccessful optimizer never erases a detected intersection.
                candidates=[(float(gaps[indices[0]]),uv[indices[0]])]
                for seed in seeds:
                    result=minimize(objective,seed,jac=True,bounds=[(0,1)]*2,method='L-BFGS-B',options={'maxiter':80,'ftol':1e-15,'gtol':1e-10})
                    if np.isfinite(result.fun):candidates.append((float(result.fun),result.x))
                    attempts.append(dict(face_axis=faceaxis,side=side,success=bool(result.success),gap_m=float(result.fun),iterations=int(result.nit)))
                value,v=min(candidates,key=lambda t:t[0])
                if value<best:
                    best=value;query=np.empty(3);query[faceaxis]=side;query[others]=v;bestq=query.tolist()
        out.append(dict(name=c.name,minimum_gap_m=best,max_penetration_m=max(0.,-best),minimum_material_coordinate=bestq,dense_minimum_gap_m=densebest,
                        dense_underestimate_m=max(0.,-best)-max(0.,-densebest),grid=grid,seeds_per_face=seeds_per_face,search_attempts=attempts,
                        scope='Dense-seeded continuous local searches onall sixQ2faces; not a certifiedglobal minimum'))
    return out
