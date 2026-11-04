"""Strict Isaac public RGB-D prefix manifest; no mechanics or action inference.

Reads only reference.npz, observations/cameras.npz and selected public sensor files.
Never reads recipe, summary, truth trajectories, future sensor files or force labels.
Cameras contain a sparse public schedule; only frames <= cutoff leave this exporter.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np

REFERENCE={'rest_points_m','tetrahedra','surface_faces','vertex_colors','means_m','colors','scales_m','opacities','face_vertex_ids','barycentric_weights'}
CAMERAS={'K','camera_to_world_usd_row','times_s','frame_ids'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def exact_npz(path,allowed):
    with np.load(path,allow_pickle=False)as z:
        if set(z.files)!=allowed:raise ValueError('Unexpected public array fields')
        result={k:z[k].copy()for k in sorted(allowed)}
    if any(v.dtype.hasobject or np.iscomplexobj(v) or not np.isfinite(v).all()for v in result.values()):raise ValueError('Finite real public arrays required')
    return result


def export_prefix(episode,destination,cutoff_frame=20):
    episode=Path(episode).resolve();destination=Path(destination)
    if isinstance(cutoff_frame,bool)or not isinstance(cutoff_frame,int)or cutoff_frame<0:raise ValueError('Nonnegative integer cutoff required')
    if destination.exists():raise FileExistsError('Prefix package is immutable; choose new destination')
    ref=exact_npz(episode/'reference.npz',REFERENCE)
    camera=exact_npz(episode/'observations/cameras.npz',CAMERAS)
    n=len(ref['rest_points_m']);g=len(ref['means_m'])
    for key,shape in [('rest_points_m',(n,3)),('vertex_colors',(n,3)),('means_m',(g,3)),('colors',(g,3)),('scales_m',(g,3)),('opacities',(g,)),('face_vertex_ids',(g,3)),('barycentric_weights',(g,3))]:
        if ref[key].shape!=shape:raise ValueError('Invalid reference shape: '+key)
    if n<4 or g<1:raise ValueError('Nonempty volume and Gaussian geometry required')
    for key,width in [('tetrahedra',4),('surface_faces',3),('face_vertex_ids',3)]:
        a=ref[key]
        if a.ndim!=2 or a.shape[1]!=width or a.dtype.kind not in 'iu' or not len(a) or a.min()<0 or a.max()>=n:raise ValueError('Invalid reference indices: '+key)
    if (ref['scales_m']<=0).any()or(ref['opacities']<0).any()or(ref['opacities']>1).any():raise ValueError('Invalid Gaussian scale/opacity')
    if not np.allclose(ref['barycentric_weights'].sum(1),1,atol=1e-6)or(ref['barycentric_weights']<0).any():raise ValueError('Invalid barycentric weights')
    ids=camera['frame_ids'];times=camera['times_s'];count=len(ids)
    if ids.ndim!=1 or ids.dtype.kind not in 'iu' or times.shape!=(count,) or count<1 or ids[0]<0 or np.any(np.diff(ids)<=0)or np.any(np.diff(times)<=0):raise ValueError('Invalid public camera clock')
    if camera['K'].shape!=(3,3) or camera['camera_to_world_usd_row'].shape!=(count,4,4):raise ValueError('Invalid camera shapes')
    if camera['K'][0,0]<=0 or camera['K'][1,1]<=0:raise ValueError('Positive focal lengths required')
    keep=ids<=cutoff_frame
    if keep.sum()<2:raise ValueError('At least two causal observed frames required')
    selected={'K':camera['K'],'frame_ids':ids[keep],'times_s':times[keep],'camera_to_world_usd_row':camera['camera_to_world_usd_row'][keep]}
    files=[]
    for frame,time in zip(selected['frame_ids'],selected['times_s']):
        record={'frame_id':int(frame),'time_s':float(time)}
        for sensor,suffix in [('rgb','.png'),('masks','.png'),('depth','.npz')]:
            path=episode/'observations'/sensor/f'{frame:05d}{suffix}'
            if not path.is_file():raise ValueError('Missing selected sensor file: '+str(path))
            record[sensor]={'path':str(path),'sha256':sha(path)}
        files.append(record)
    # Source sensor arrays are not decoded. Hash only selected sensor bytes.
    destination.mkdir(parents=True)
    np.savez_compressed(destination/'reference.npz',**ref)
    np.savez_compressed(destination/'cameras.npz',**selected)
    manifest={'schema':1,'source_episode':str(episode),'cutoff_frame':cutoff_frame,'observed_frames':selected['frame_ids'].tolist(),'observed_times_s':selected['times_s'].tolist(),'reference_sha256':sha(destination/'reference.npz'),'cameras_sha256':sha(destination/'cameras.npz'),'sensors':files,'known_position_commands':False,'future_object_access':False,'force_labels_exported':False,'physical_parameters_exported':False,'units':'metres and seconds; USD row camera poses, -Z forward +Y up','depth':'Positive camera-axis depth; background zero; supplied optional RGB-D observation','scope':'Initial geometry plus causal RGB-D prefix only; no tracked nodal observations, prescribed handle-position actions or physical-accuracy claim'}
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--episode',required=True);p.add_argument('--output',required=True);p.add_argument('--cutoff-frame',type=int,default=20);a=p.parse_args()
    m=export_prefix(a.episode,a.output,a.cutoff_frame);print(json.dumps({'frames':m['observed_frames'],'known_position_commands':False}))
