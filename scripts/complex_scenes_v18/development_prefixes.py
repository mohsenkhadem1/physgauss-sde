"""Export only permitted development calibration tracks, never forecast targets.

Known-boundary diagnostic: scene geometry, gravity, mass and compliant boundary
parameters are supplied by the existing pilot adapter. Material is not supplied.
Visibility is a fixed surface subset, not an image-derived occlusion estimate.
"""
import hashlib,json
from pathlib import Path
import numpy as np
from scripts.complex_scenes_v18.run_development import ROOT,write


def export():
    out=ROOT/'runs/complex-scenes-v18/development-prefixes';out.mkdir(parents=True,exist_ok=False);rows=[]
    cuts=dict(clamp_release=.08,curved_push=.10,container_drop=.16,rigid_impact=.06,deformable_pair=.11,heterogeneous=.08)
    for p in sorted((ROOT/'runs/complex-scenes-v18/pilot-01').glob('*/complete.json')):
        status=json.loads(p.read_text());family=status['family'];variant=status['variant']
        if not status['completed']:raise ValueError('Explicit complete pilot required')
        raw=p.parent/'trajectory.npz'
        if hashlib.sha256(raw.read_bytes()).hexdigest()!=status['trajectory_sha256']:raise ValueError('Pilot hash mismatch')
        a=np.load(raw);q=a['material_queries'];indices=np.flatnonzero((q[:,1]==0)|(q[:,2]==1))
        cut=.14 if family=='deformable_pair'and variant else cuts[family]
        keep=a['times_s']<=cut+1e-10;name=f'{family}-v{variant}';folder=out/name;folder.mkdir()
        target=folder/'observations.npz';np.savez_compressed(target,positions_m=a['surface_positions_m'][keep][:,:,indices],times_s=a['times_s'][keep],surface_indices=indices,material_queries=q[indices])
        meta=dict(case=family,variant=variant,calibration_duration_s=float(a['times_s'][keep][-1]),development_only=True,reference_qualified=False,boundary_track='Known boundary parameters; material unknown; fixed two-face track subset',source_reference_sha256=status['trajectory_sha256'],observations_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),frame_count=int(keep.sum()),tracked_points_per_body=len(indices))
        write(folder/'metadata.json',meta);rows.append(meta)
    write(out/'manifest.json',dict(cases=rows,forecast_targets_exported=False,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print('Exported',len(rows),'development prefixes')


if __name__=='__main__':export()
