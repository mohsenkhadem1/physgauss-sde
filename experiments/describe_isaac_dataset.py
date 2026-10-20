"""Truth-only visibility labels and dataset measurements; never model inputs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np


def visibility_labels(means,points,faces,face_ids,pose,k,depth):
    """0=hidden, 1=visible, 2=ambiguous. Conservative sensor-depth labelling.

    Bilinear inverse depth is exact on a planar perspective triangle. At raster
    edges, background or disagreement within the tolerance band, leave ambiguity.
    This is a raster visibility diagnostic, not exact analytic ray visibility.
    """
    xyz=np.c_[means,np.ones(len(means))]@np.linalg.inv(pose);z=-xyz[:,2]
    uv=np.c_[k[0,0]*xyz[:,0]/z+k[0,2],-k[1,1]*xyz[:,1]/z+k[1,2]]-.5
    xy=np.floor(uv).astype(int);inside=(z>.01)&(xy[:,0]>=0)&(xy[:,0]<639)&(xy[:,1]>=0)&(xy[:,1]<479)
    labels=np.full(len(means),2,dtype=np.uint8)
    tri=points[faces[face_ids]];normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    front=(normal*(pose[3,:3]-means)).sum(1)>0
    labels[inside&~front]=0
    ids=np.flatnonzero(inside&front);x,y=xy[ids].T;dx,dy=(uv[ids]-xy[ids]).T
    samples=np.stack([depth[y,x],depth[y,x+1],depth[y+1,x],depth[y+1,x+1]],1)
    weights=np.stack([(1-dx)*(1-dy),dx*(1-dy),(1-dx)*dy,dx*dy],1)
    valid=(samples>.01).all(1);sensor=np.zeros(len(ids))
    sensor[valid]=1/(weights[valid]/samples[valid]).sum(1)
    delta=z[ids]-sensor
    labels[ids[valid&(abs(delta)<.0005)]]=1
    labels[ids[valid&(delta>.0015)]]=0
    return labels


def episode_metrics(root):
    root=Path(root)
    with np.load(root/'truth/trajectory.npz') as raw:
        q=raw['positions_m'];means=raw['gaussian_means_m'];times=raw['times_s']
    with np.load(root/'reference.npz') as raw:
        p=raw['rest_points_m'];faces=raw['surface_faces'];face_vertices=raw['face_vertex_ids']
    with np.load(root/'truth/cameras.npz') as raw:poses=raw['camera_to_world_usd_row'];k=raw['K']
    lookup={tuple(face):i for i,face in enumerate(faces)}
    face_ids=np.array([lookup[tuple(face)] for face in face_vertices])
    labels=[];elastic=[];pc=p-p.mean(0)
    for i,state in enumerate(q):
        with np.load(root/'truth/depth'/f'{i:05d}.npz') as raw:depth=raw['depth_m']
        labels.append(visibility_labels(means[i],state,faces,face_ids,poses[i],k,depth))
        qc=state-state.mean(0);u,_,vt=np.linalg.svd(pc.T@qc)
        correction=np.eye(3);correction[-1,-1]=np.linalg.det(u@vt);rotation=u@correction@vt
        elastic.append(float(np.sqrt(np.mean((pc@rotation-qc)**2))))
    labels=np.asarray(labels)
    sparse=json.loads((root/'truth/recipe.json').read_text())['sparse_frames']
    np.savez_compressed(root/'truth/visibility.npz',labels=labels,times_s=times,
        label_names=np.array(['hidden','visible','ambiguous']),visible_tolerance_m=.0005,occluded_gap_m=.0015)
    stats={'episode_id':root.name,'peak_elastic_coordinate_rms_m':max(elastic),
           'mean_gaussian_visibility_fractions':{name:float((labels==i).mean()) for i,name in enumerate(['hidden','visible','ambiguous'])},
           'always_hidden_at_observed_times_fraction':float((labels[sparse]==0).all(0).mean()),
           'never_confidently_visible_in_sparse_inputs_fraction':float((labels[sparse]!=1).all(0).mean()),
           'visibility_method':'Front-facing triangle and bilinear inverse sensor depth; 0.5mm visible tolerance, >1.5mm occlusion gap; raster edges ambiguous',
           'elastic_measurement':'Coordinate RMS after best proper rigid alignment to rest; no displacement amplification'}
    (root/'truth/metrics.json').write_text(json.dumps(stats,indent=2)+'\n')
    return stats


def describe(root,output):
    root=Path(root);catalog=json.loads((root/'catalog.json').read_text())
    assert catalog['completed'] and catalog['episodes_verified']==48
    with ThreadPoolExecutor(max_workers=6) as pool:metrics=list(pool.map(episode_metrics,sorted((root/'episodes').iterdir())))
    summaries=catalog['summaries'];split_counts={split:sum(s['split']==split for s in summaries) for split in ['train','validation','test']}
    result={'dataset':catalog['dataset'],'plan_sha256':catalog['plan_sha256'],'episodes':len(summaries),
            'splits':split_counts,'frames_per_episode':61,'sparse_input_frames_per_episode':10,
            'dense_rgb_frames':len(summaries)*61,'sparse_rgb_frames':len(summaries)*10,
            'minimum_tet_volume_ratio':min(s['minimum_tet_volume_ratio'] for s in summaries),
            'maximum_constraint_error_m':max(s['maximum_constraint_error_m'] for s in summaries),
            'maximum_displacement_m':max(s['maximum_displacement_m'] for s in summaries),
            'all_independent_episode_audits_pass':all(json.loads((root/'episodes'/s['episode_id']/'validation.json').read_text())['passed'] for s in summaries),
            'numerical_convergence_established':False,'inverse_recovery_evaluated':False,
            'metrics':metrics,'summaries':summaries}
    Path(output).write_text(json.dumps(result,indent=2)+'\n')
    # Rebuild the complete manifest after truth postprocessing.
    from validate_isaac_dataset import sha
    files=[{'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(root.rglob('*')) if p.is_file() and p.name not in ['file-manifest.json','.generation.lock']]
    (root/'file-manifest.json').write_text(json.dumps({'files':files,'total_bytes':sum(f['bytes'] for f in files)},indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['metrics','summaries']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('dataset');parser.add_argument('--output',required=True)
    args=parser.parse_args();describe(args.dataset,args.output)
