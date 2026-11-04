"""Separate frozen v2 baseline: split pose/shape plus archived rigid floor response.

Uses archived integrate_pose verbatim (80 substeps, friction .5, zero restitution,
unit-mass rigid cloud). Only supplied initial min-z plane, no asset downloads.
While held, a whole-cloud vertical projection removes plane penetration; after
prescribed release, archived impulse/projection continuation replaces ballistic
flight. These are kinematic/rigid corrections, NOT coupled FEM contact, inferred
forces, energy qualification or evidence of physical accuracy. No slip inference.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.gaussian_boundary_flow_v02.pose_shape import predict as pose_predict
from scripts.external_benchmark.data import rigid_map
from scripts.contact_benchmark_v12.contact_rollout import integrate_pose
from scripts.contact_benchmark_v12.scene import StaticScene

POLICY=dict(version=2, gravity_m_s2=9.81, substeps=80, friction=.5, restitution=0.,
            release='first prescribed inactive attachment; irreversible', inferred_slip=False,
            held_contact='whole-cloud vertical translation onto supplied initial min-z plane',
            released_contact='archived integrate_pose unchanged; unit mass and rigid node-cloud inertia',
            geometry='supplied initial min-z plane only; no inferred or future scene geometry',
            shape='raw FEM shape while held; rigid shape after separation',
            fallback='none; all cases and inherited sample failures retained',
            scope='kinematic/rigid contact baseline; not coupled FEM or physically validated contact')


def predict(data, raw_samples, *, prefix_frames, point_ids):
    out,meta=pose_predict(data,raw_samples,prefix_frames=prefix_frames,point_ids=point_ids)
    end=int(data['train_end']);last=end-1;times=np.asarray(data['times']);floor=float(data['floor'])
    if not np.isfinite(floor):raise ValueError('Finite initial floor required')
    scene=StaticScene(floor=floor);release=meta['prescribed_release_frame']
    diagnostics=[]
    for s in range(len(out)):
        max_projection=0.;contacts=0;max_prepenetration=0.
        unprojected_anchor=out[s,last].copy()
        unprojected_next=out[s,last+1].copy() if last+1<len(times) else None
        held_end=release if release is not None else len(times)
        # If already released, still admit the terminal observed pose onto plane.
        for t in range(last,max(last+1,held_end)):
            correction=max(0.,float(floor-out[s,t,:,2].min()))
            out[s,t,:,2]+=correction;max_projection=max(max_projection,correction)
        if release is not None:
            anchor=max(last,release-1);points=out[s,anchor].copy()
            if anchor>last:
                previous=out[s,anchor-1];dt=times[anchor]-times[anchor-1]
                velocity=(points.mean(0)-previous.mean(0))/dt
                r,_=rigid_map(previous,points);omega=Rotation.from_matrix(r.T).as_rotvec()/dt
            else:
                # Infer the already-frozen pose layer's prefix velocity from its
                # first ballistic step, removing known gravity displacement.
                dt=times[last+1]-times[last]
                velocity=(unprojected_next.mean(0)-unprojected_anchor.mean(0))/dt+np.array([0,0,.5*9.81*dt])
                r,_=rigid_map(unprojected_anchor,unprojected_next);omega=Rotation.from_matrix(r.T).as_rotvec()/dt
            for t in range(anchor+1,len(times)):
                points,velocity,omega,info=integrate_pose(points,velocity,omega,times[t]-times[t-1],
                                                        scene,friction=POLICY['friction'],substeps=POLICY['substeps'])
                out[s,t]=points;contacts+=info['contact_substeps']
                max_prepenetration=max(max_prepenetration,info['max_preprojection_penetration_m'])
        diagnostics.append(dict(maximum_held_vertical_projection_m=max_projection,
                                contact_substeps=contacts,maximum_preprojection_penetration_m=max_prepenetration,
                                maximum_returned_penetration_m=max(0.,float(floor-out[s,last:,:,2].min()))))
    meta.update(policy=POLICY.copy(),contact_diagnostics=diagnostics,
                inherited_pose_policy=meta['policy'],
                limitations='Projection/impulses correct rigid pose only; no elastic feedback, inferred forces, or energy accuracy claim')
    return out,meta
