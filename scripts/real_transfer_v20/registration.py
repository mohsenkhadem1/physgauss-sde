"""Initial-geometry-only rigid registration; no future motion correspondence."""
import itertools,json,pickle,numpy as np
from scipy.spatial import cKDTree
from scripts.external_benchmark.data import fps,rigid_map
from .development_run import ROOT,write,sha

PAIRS=[('single_lift_sloth',t) for t in ('double_lift_sloth','double_stretch_sloth','single_push_sloth')]+[
       ('single_lift_zebra',t) for t in ('double_lift_zebra','double_stretch_zebra')]

def initial_bundle(case):
    path=ROOT/'runs/third_party/PhysTwin/data/different_types'/case/'final_data.pkl'
    # Only initial shape and prescribed tool positions leave this loader.
    with path.open('rb') as f:d=pickle.load(f)
    flip=np.diag([1.,-1.,-1.]);x=np.concatenate([d['object_points'][0],d['surface_points'],d['interior_points']])@flip
    return x,d['controller_points']@flip,sha(path)

def register(source,target):
    a=source[fps(source,min(600,len(source)))];b=target[fps(target,min(600,len(target)))]
    ac=a.mean(0);bc=b.mean(0)
    _,_,A=np.linalg.svd(a-ac,full_matrices=False);_,_,B=np.linalg.svd(b-bc,full_matrices=False)
    tree=cKDTree(b);candidates=[np.eye(3)]
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1,1),repeat=3):
            Q=np.eye(3)[list(permutation)]*np.array(signs)[:,None];R=A.T@Q@B
            if np.linalg.det(R)>.99:candidates.append(R)
    best=None
    for R in candidates:
        shift=bc-ac@R
        for _ in range(50):
            current=a@R+shift;distance,ids=tree.query(current)
            good=distance<=np.quantile(distance,.9)
            delta,offset=rigid_map(current[good],b[ids[good]])
            R=R@delta;shift=shift@delta+offset
            if np.linalg.norm(delta-np.eye(3))+np.linalg.norm(offset)<1e-8:break
        transformed=a@R+shift
        distances=np.r_[tree.query(transformed)[0],cKDTree(transformed).query(b)[0]]
        objective=np.quantile(distances,.9)
        if best is None or objective<best[0]:best=(objective,R,shift)
    _,R,shift=best;transformed=source@R+shift
    distances=np.r_[cKDTree(target).query(transformed)[0],cKDTree(transformed).query(target)[0]]
    extent=float(np.linalg.norm(np.ptp(source,axis=0)))
    return dict(rotation=R.tolist(),translation_m=shift.tolist(),median_distance_mm=float(np.median(distances)*1000),
                p95_distance_mm=float(np.quantile(distances,.95)*1000),extent_m=extent,
                initial_geometry_gate=bool(np.median(distances)<.03*extent and np.quantile(distances,.95)<.08*extent))

def main():
    rows=[]
    for source,target in PAIRS:
        a,_,s=initial_bundle(source);b,_,t=initial_bundle(target);row=register(a,b)
        row.update(source=source,target=target,source_sha256=s,target_sha256=t)
        rows.append(row);print(source,'->',target,row['median_distance_mm'],row['p95_distance_mm'],row['initial_geometry_gate'],flush=True)
    write(ROOT/'reports/real-transfer-v20/initial-registration.json',dict(pairs=rows,
        criterion='Initial shape only: symmetric median distance <3% and p95 <8% of source bounding-box diagonal. No nonrigid registration or target motion.',
        scope='Registration feasibility, not confirmed semantic correspondence. Same-object identity inferred from release labels; visual/semantic checks still required before strong transfer claims.'))

if __name__=='__main__':main()
