"""Export fixed gripper surface landmarks from pose/opening, without tactile.

These are prescribed tool positions, not inferred object contact points. A
mechanical model must still decide attachment/contact using prefix observations.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from deform360.robot import load_robot_state
from deform360.processing.control_points_stage import gripper_taxel_points

ROOT=Path(__file__).resolve().parents[2]

def build_controls(poses,openings):
    poses=np.asarray(poses);openings=np.asarray(openings)
    if poses.ndim==3:poses=poses[:,None];openings=openings[:,None]
    if poses.shape[:2]!=openings.shape or poses.shape[-2:]!=(4,4):
        raise ValueError('Inconsistent pose/opening shapes')
    if not np.isfinite(poses).all() or not np.isfinite(openings).all():
        raise ValueError('Nonfinite robot state')
    # Uniform grid on each rigid finger, identical across time and objects.
    indices=np.array([(row*32+col)*2+side for row in (0,5,11)
                      for col in (0,10,21,31) for side in (0,1)])
    result=np.array([[gripper_taxel_points(float(op),pose)[indices]
                      for pose,op in zip(frame,opening)]
                     for frame,opening in zip(poses,openings)])
    return result,indices

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--object',default='046-sponge')
    parser.add_argument('--episode',type=int,default=0);args=parser.parse_args()
    episode=ROOT/'runs/real-transfer-v19/Deform360/processed'/args.object/f'episode_{args.episode:04d}'
    source=episode/'robot/robot.npz';state=load_robot_state(source)
    positions,indices=build_controls(state.T_worlds,state.openings)
    target=episode/'position-controls.npz'
    np.savez_compressed(target,positions_m=positions,landmark_indices=indices)
    report=dict(object=args.object,episode=args.episode,shape=list(positions.shape),
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                output_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                input_contract='Only robot poses and openings; no tactile, object geometry, image or contact-window input.',
                meaning='Prescribed gripper surface landmarks; not known material attachments.')
    target.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
