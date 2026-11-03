"""Physical orientation audit of the common Q2 nodal-to-surface representation.

For spring methods this audits their common interpolated volume, not a native
constitutive law. Positive sampled determinants do not prove global injectivity.
"""
import argparse,hashlib,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scripts.complex_scenes_v18.dense_jacobian import DenseJacobianAudit
from scripts.complex_scenes_v18.run_development import ROOT,write


def audit(folder,samples=9):
    folder=Path(folder);status=json.loads((folder/'complete.json').read_text())
    for name,key in [('geometry.npz','geometry_sha256'),('nodal-state.npz','nodal_state_sha256')]:
        if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=status[key]:raise ValueError('State integrity failure')
    with np.load(folder/'geometry.npz')as geometry,np.load(folder/'nodal-state.npz')as state:
        positions=state['positions_m'];times=state['times_s'];velocities=state['velocities_m_s']
        if positions.shape!=velocities.shape or len(times)!=len(positions)or not np.isfinite(velocities).all():raise ValueError('Invalid exported state')
        minima=[]
        for body in range(positions.shape[1]):
            rest=geometry[f'rest_{body}'];model=SimpleNamespace(rest=rest,elements=geometry[f'elements_{body}']);checker=DenseJacobianAudit(model,samples)
            minima.append([checker.minimum(x-rest)for x in positions[:,body]])
        minima=np.asarray(minima).T;bad=(minima<=0).any(axis=1)
        return dict(scope='Sampled orientation of common Q2 volume; not proof of global nonintersection or native material validity',forecast_completed=status['completed'],minimum_sampled_J=float(minima.min()),frames_with_nonpositive_J=int(bad.sum()),total_frames=len(times),first_nonpositive_time_s=float(times[np.flatnonzero(bad)[0]])if bad.any()else None,samples_per_axis=samples,minimum_J_per_frame_body=minima.tolist(),nodal_state_sha256=status['nodal_state_sha256'],geometry_sha256=status['geometry_sha256'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    if Path(args.output).name!=args.output:raise ValueError('Report basename required')
    result=audit(args.folder);write(ROOT/'reports/complex-scenes-v18'/args.output,result);print(json.dumps({k:v for k,v in result.items()if k!='minimum_J_per_frame_body'}))
