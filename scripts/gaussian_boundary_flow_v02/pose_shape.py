"""Frozen split pose/shape diagnostic; not coupled mechanics or contact inference.

Policy v1: fit proper rigid object pose from the SAME declared prefix frames and
points; use last eight declared frames for velocity. Transport terminal pose in
prescribed controller frame while held. First prescribed inactive attachment
frame initiates irreversible free flight, with prefix-estimated velocities if
already free or controller finite differences at later separation. Gravity is
known SI 9.81; no impact/floor projection. Release shape is frozen. Inferred slip
is DISABLED: recorded command activation and unknown physical attachment differ.
No error-based fallback/selection. Rank-deficient observations fail explicitly.
Compare this fixed candidate with unchanged raw predictions on every case.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.external_benchmark.data import rigid_map

POLICY = dict(version=1, velocity_frames=8, gravity_m_s2=[0., 0., -9.81],
              inferred_slip=False, release='first prescribed inactive frame; irreversible',
              impacts='unsupported; no collision response',
              shape='raw FEM shape while held; freeze at separation',
              fallback='none; fail explicitly', scope='split pose/shape prediction, not coupled contact')


def _fit(source, target):
    if len(source) < 3 or np.linalg.matrix_rank(source-source.mean(0), tol=1e-10) < 2:
        raise ValueError('At least three non-collinear shared points required')
    return rigid_map(source, target)


def predict(data, raw_samples, *, prefix_frames, point_ids):
    """raw_samples [S,T,N,3]; returns transformed joint samples and audit metadata.

    Reads only declared prefix point/frame coordinates. Other prefix observations
    cannot influence the output. Future prescribed controller/activation is an
    input; future object observations are rejected by the prefix-length contract.
    """
    if any(k in data for k in ('evaluation', 'manual', 'forces', 'material', 'object_points')):
        raise ValueError('Evaluator/privileged fields forbidden')
    end = int(data['train_end']); obs = np.asarray(data['observed'])
    if len(obs) != end or len(data['observed_mask']) != end:
        raise ValueError('Only prefix observations permitted')
    frames = np.asarray(prefix_frames, dtype=int); ids = np.asarray(point_ids, dtype=int)
    if len(frames) < 2 or frames[-1] != end-1 or np.any(np.diff(frames) <= 0) or frames[0] < 0:
        raise ValueError('Ordered declared prefix frames ending at cutoff required')
    if ids.ndim != 1 or len(set(ids.tolist())) != len(ids) or np.any(ids < 0) or np.any(ids >= obs.shape[1]):
        raise ValueError('Distinct valid observation IDs required')
    nodes = np.asarray(data['nodes']); times = np.asarray(data['times'])
    raw = np.asarray(raw_samples, dtype=float)
    if raw.ndim != 4 or raw.shape[1:] != (len(times), len(nodes), 3) or not np.isfinite(raw).all():
        raise ValueError('Finite complete joint trajectories required; failures remain external')
    if np.any(np.diff(times) <= 0): raise ValueError('Increasing physical times required')
    initial = np.asarray(data['observed_initial'])[ids]
    selected = obs[frames][:, ids]
    valid = np.asarray(data['observed_mask'])[frames][:, ids].all(0)
    valid &= np.isfinite(selected).all(axis=(0,2))
    object_poses = [_fit(initial[valid], x[valid]) for x in selected]
    center0 = nodes.mean(0)
    centers = np.array([center0 @ r+b for r,b in object_poses])
    count = min(POLICY['velocity_frames'], len(frames))
    fit_times = times[frames[-count:]] - times[frames[-1]]
    design = np.c_[np.ones(count), fit_times]
    velocity = np.linalg.lstsq(design, centers[-count:], rcond=None)[0][1]
    terminal_R = object_poses[-1][0]
    # Row-vector orientation: relative rotation applied on the right.
    relative_rot = np.array([Rotation.from_matrix(r.T @ terminal_R).as_rotvec()
                             for r,b in object_poses[-count:]])
    omega_row = -np.linalg.lstsq(design, relative_rot, rcond=None)[0][1]
    active = np.asarray(data['attachment_active']).sum(1) > 0
    targets = np.asarray(data['controller_targets'])
    weights = np.asarray(data['attachment_active'])
    controller = []
    for t in range(len(times)):
        if active[t]:
            desired = np.average(targets[t], axis=0, weights=weights[t])
            controller.append(_fit(nodes, desired))
        else:
            controller.append(controller[-1] if controller else (np.eye(3), np.zeros(3)))
    last = end-1; C, c = controller[last]
    local_center = (centers[-1]-c) @ C.T
    relative_R = terminal_R @ C.T
    desired_centers = []; desired_rotations = []
    for t in range(last, len(times)):
        r,b = controller[t]
        desired_centers.append(local_center @ r+b)
        desired_rotations.append(relative_R @ r)
    release_candidates = np.flatnonzero(~active[last:])
    release = last+int(release_candidates[0]) if len(release_candidates) else None
    if release is not None:
        index = release-last
        if index > 0:
            start_center = desired_centers[index-1]; start_R = desired_rotations[index-1]
            if index > 1:
                dt = times[release-1]-times[release-2]
                velocity = (desired_centers[index-1]-desired_centers[index-2])/dt
                omega_row = Rotation.from_matrix(desired_rotations[index-2].T @ start_R).as_rotvec()/dt
            start_time = times[release-1]
        else:
            start_center = centers[-1]; start_R = terminal_R; start_time = times[last]
        for t in range(release, len(times)):
            dt = times[t]-start_time
            desired_centers[t-last] = start_center+velocity*dt+.5*np.array(POLICY['gravity_m_s2'])*dt**2
            desired_rotations[t-last] = start_R @ Rotation.from_rotvec(omega_row*dt).as_matrix()
    output = raw.copy()
    for s in range(len(raw)):
        for t in range(last, len(times)):
            shape_frame = min(t, max(last, release-1)) if release is not None else t
            shape = raw[s,shape_frame]
            r,b = _fit(nodes, shape)
            local = (shape-shape.mean(0)) @ r.T
            output[s,t] = local @ desired_rotations[t-last]+desired_centers[t-last]
    return output, dict(policy=POLICY.copy(), prefix_frames=frames.tolist(), point_ids=ids.tolist(),
                        shared_points=int(valid.sum()), prescribed_release_frame=release,
                        inferred_release_frame=None, calibration_frame=last,
                        node_shape_preserved_while_held=True,
                        uncertainty='Rigid pose is a point estimate; only raw FEM shape sample uncertainty retained',
                        limitations='No reaction forces, momentum coupling, collision response or calibrated pose uncertainty')
