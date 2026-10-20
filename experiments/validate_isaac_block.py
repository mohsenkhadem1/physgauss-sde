"""Independently replay geometry, fixture and load records from Isaac outputs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from boundary_block import forces_at


def validate(root, unloaded=False):
    root=Path(root)
    metadata=json.loads((root/'summary.json').read_text())
    spec=json.loads((root/'truth/boundaries.json').read_text())
    with np.load(root/'reference.npz') as ref:
        rest=ref['rest_points_m'];tet=ref['tetrahedra'];faces=ref['surface_faces']
        gaussian_ids=ref['face_vertex_ids'];gaussian_weights=ref['barycentric_weights']
    with np.load(root/'truth/trajectory.npz') as raw:
        time=raw['times_s'];position=raw['positions_m'];velocity=raw['velocities_m_s']
        simulation_time=raw['simulated_times_s']
        frame_forces=raw['commanded_handle_forces_N'];force_times=raw['saved_force_midpoints_s']
        step_times=raw['force_step_midpoints_s'];step_forces=raw['force_step_commands_N']
        handles=raw['handle_positions_m']
        gaussian_error=float(np.abs(raw['gaussian_means_m']-np.einsum('ni,fnij->fnj',gaussian_weights,position[:,gaussian_ids])).max())
    finite=all(np.isfinite(x).all() for x in [time,position,velocity,frame_forces,step_forces,handles])
    pin_error=float(np.linalg.norm(position[:,spec['fixture_nodes']]-rest[spec['fixture_nodes']],axis=-1).max())
    handle_error=float(np.linalg.norm(handles-position[:,spec['load_nodes']],axis=-1).max())
    rest_det=np.linalg.det(rest[tet[:,1:]]-rest[tet[:,:1]])
    det=np.linalg.det(position[:,tet[:,1:]]-position[:,tet[:,:1]])
    replay=np.asarray([forces_at(spec,float(t)) for t in step_times])
    frame_replay=np.asarray([forces_at(spec,float(t)) for t in force_times])
    force_error=float(max(np.max(np.abs(replay-step_forces)),np.max(np.abs(frame_replay-frame_forces))))
    centered=rest-rest.mean(0);elastic=[]
    for q in position:
        moving=q-q.mean(0);u,_,vt=np.linalg.svd(centered.T@moving)
        rotation=u@np.diag([1,1,np.linalg.det(u@vt)])@vt
        elastic.append(float(np.sqrt(np.mean((moving-centered@rotation)**2))))
    displacement=float(np.linalg.norm(position-rest,axis=-1).max())
    images=sorted((root/'observations').glob('rgb-*.png'))
    checks={'completed':metadata['completed'],'finite':finite,'positive_rest_tets':bool(np.all(rest_det>0)),
            'source_copies_match_hashes':all((root/'provenance'/name).exists() and
                hashlib.sha256((root/'provenance'/name).read_bytes()).hexdigest()==digest
                for name,digest in metadata['source_sha256'].items()),
            'gaussian_material_attachment':gaussian_error<1e-7,
            'no_inverted_saved_tets':bool(np.all(det>0)),'fixture_held':pin_error<2e-4,
            'handles_attached':handle_error<2e-3,'force_replay':force_error<1e-7,
            'correct_step_count':len(step_times)==round(metadata['seconds']*metadata['physics_hz']),
            'actual_physics_clock_matches':bool(np.allclose(simulation_time,time,atol=1e-7,rtol=1e-7)),
            'uniform_frame_clock':bool(np.allclose(np.diff(time),1/metadata['observation_fps'])),
            'fixture_on_surface':bool(set(spec['fixture_nodes'])<=set(np.unique(faces))),
            'loads_distinct_from_fixture':not bool(set(spec['load_nodes'])&set(spec['fixture_nodes'])),
            'rgb_count_if_present':not images or len(images)==len(time),
            'motion_check':displacement<1e-5 if unloaded else max(elastic)>1e-5}
    result={'checks':checks,'all_pass':all(checks.values()),'saved_frames':len(time),'rgb_frames':len(images),
            'maximum_pin_error_m':pin_error,'maximum_handle_attachment_error_m':handle_error,
            'minimum_saved_tet_volume_ratio':float((det/rest_det).min()),'force_replay_error_N':force_error,
            'maximum_displacement_m':displacement,'maximum_elastic_coordinate_rms_m':max(elastic),
            'unloaded_control':unloaded,'qualification':'Simulator integrity checks; no inverse-boundary recovery or calibrated uncertainty claim'}
    (root/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    if not result['all_pass']:raise ValueError(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path);parser.add_argument('--unloaded',action='store_true')
    args=parser.parse_args();print(json.dumps(validate(args.output,args.unloaded),indent=2))
