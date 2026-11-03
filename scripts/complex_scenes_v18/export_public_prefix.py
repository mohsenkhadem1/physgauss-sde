"""Export fixed visible calibration tracks and public actions; omit future states."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scripts.complex_scenes_v18.run_development import write
from scripts.complex_scenes_v18.world_queries import Query


def export(reference,out,cut,split='development'):
    reference=Path(reference);out=Path(out)
    if split not in ('development','held'):raise ValueError('Explicit evaluation split required')
    status=json.loads((reference/'complete.json').read_text());raw=reference/'trajectory.npz'
    if not status['completed']or hashlib.sha256(raw.read_bytes()).hexdigest()!=status['trajectory_sha256']:raise ValueError('Complete intact reference required')
    query=Query(**status['public_query'])
    with np.load(raw)as a:
        times=a['times_s'];q=a['material_queries'];indices=np.flatnonzero((q[:,1]==0)|(q[:,2]==1));keep=times<=cut+1e-10
        if keep.sum()<2 or not np.isclose(times[keep][-1],cut,atol=1e-9)or cut>=times[-1]-1e-10:raise ValueError('Strict calibration prefix on the sample clock required')
        positions=a['surface_positions_m'][keep][:,:,indices].copy();clock=times[keep].copy();queries=q[indices].copy()
    out.mkdir(parents=True,exist_ok=False);target=out/'observations.npz'
    np.savez_compressed(target,positions_m=positions,times_s=clock,surface_indices=indices,material_queries=queries)
    meta=dict(case=query.family,variant=query.variant,public_query=status['public_query'],world_id=status['world_id'],world_material_commitment=status['private_world_sha256'],split=split,development_only=split=='development',calibration_duration_s=float(clock[-1]),source_reference_sha256=status['trajectory_sha256'],observations_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),frame_count=len(clock),tracked_points_per_body=len(indices),boundary_track='Known public boundary; unknown material; fixed two-face tracks',forecast_targets_exported=False)
    write(out/'metadata.json',meta);return meta


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reference',required=True);p.add_argument('--output',required=True);p.add_argument('--cut',type=float,required=True);p.add_argument('--split',choices=['development','held'],default='development');args=p.parse_args();print(json.dumps(export(args.reference,args.output,args.cut,args.split)))
