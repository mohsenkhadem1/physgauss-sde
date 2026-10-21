"""CPU audit of saved dataset geometry, clocks, loads, sensors and sparse inputs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from isaac_dataset import geometry, camera_pose, boundary_shift, forces_at, digest_json, load_model_inputs
from boundary_block import lumped_masses


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def ray_depth(points, faces, pose, k, xs, ys):
    """Independent Moller-Trumbore intersections at pixel centres, camera-axis depth."""
    ray=np.stack([(xs+.5-k[0,2])/k[0,0],-(ys+.5-k[1,2])/k[1,1],-np.ones(len(xs))],-1)@pose[:3,:3]
    tri=points[faces];e1=tri[:,1]-tri[:,0];e2=tri[:,2]-tri[:,0]
    h=np.cross(ray[:,None],e2[None]);det=(e1[None]*h).sum(-1)
    inv=np.divide(1,det,out=np.zeros_like(det),where=abs(det)>1e-10)
    s=pose[3,:3]-tri[:,0];u=inv*(s[None]*h).sum(-1);q=np.cross(s,e1)
    v=inv*(ray[:,None]*q[None]).sum(-1);t=inv*(e2*q).sum(-1)[None]
    return np.where((u>=0)&(v>=0)&(u+v<=1)&(t>0)&(abs(det)>1e-10),t,np.inf).min(1)


def validate(root, expected=None):
    root=Path(root);summary=json.loads((root/'summary.json').read_text())
    require(summary['completed'] and not (root/'failure.json').exists(),'Unsuccessful episode')
    case=json.loads((root/'truth/recipe.json').read_text())
    if expected is not None:require(case==expected,'Recipe differs from declared plan')
    for key in ['episode_id','split','group_id','material','boundary_type','force_type','fps']:
        require(summary[key]==case[key],f'Summary {key} differs from recipe')
    require(digest_json({k:v for k,v in case.items() if k!='recipe_sha256'})==case['recipe_sha256'],'Recipe hash')
    require(summary['recipe_sha256']==case['recipe_sha256'],'Summary recipe mismatch')
    for name,digest in summary['source_sha256'].items():
        require(sha(root/'provenance'/name)==digest,'Source provenance hash')
        require(sha(Path(__file__).with_name(name))==digest,'Validator helper/source differs from frozen generation source')
    points,tets,faces,spec=geometry(case)
    require(json.loads((root/'truth/boundaries.json').read_text())==spec,'Boundary geometry replay')
    with np.load(root/'reference.npz') as ref:reference={k:ref[k] for k in ref.files}
    np.testing.assert_array_equal(reference['rest_points_m'],points)
    np.testing.assert_array_equal(reference['tetrahedra'],tets)
    np.testing.assert_array_equal(reference['surface_faces'],faces)
    with np.load(root/'truth/trajectory.npz') as raw:a={k:raw[k] for k in raw.files}
    with np.load(root/'truth/cameras.npz') as raw:c={k:raw[k] for k in raw.files}
    for key,value in a.items():require(np.isfinite(value).all(),f'Nonfinite {key}')
    n=round(case['seconds']*case['fps'])+1;times=np.arange(n)/case['fps'];hz=summary['physics_hz']
    require(hz==case['physics_hz'],'Production physics rate differs from plan')
    np.testing.assert_allclose(a['times_s'],times,atol=1e-12)
    np.testing.assert_allclose(a['simulated_times_s'],times,atol=3e-7,rtol=0)
    q=a['positions_m'];require(q.shape==(n,len(points),3),'Trajectory shape')
    np.testing.assert_allclose(q[0],points,atol=1e-6)
    rest_det=np.linalg.det(points[tets[:,1:]]-points[tets[:,:1]])
    require(np.all(rest_det>0),'Rest tetrahedron orientation')
    ratios=np.linalg.det(q[:,tets[:,1:]]-q[:,tets[:,:1]])/rest_det
    require(ratios.min()>=.1 and summary['minimum_tet_volume_ratio']>=.1,'Tet collapse/inversion')
    fixed=spec['fixture_nodes'];targets=np.array([points[fixed]+boundary_shift(spec,t) for t in times])
    np.testing.assert_allclose(a['boundary_target_positions_m'],targets,atol=1e-12)
    bc_error=float(np.linalg.norm(q[:,fixed]-targets,axis=-1).max())
    require(bc_error<2e-4 and summary['maximum_constraint_error_m']<2e-4,'Boundary drift')
    step_times=(np.arange(round(hz*case['seconds']))+.5)/hz
    np.testing.assert_allclose(a['force_step_midpoints_s'],step_times,atol=1e-12)
    replay=np.array([forces_at(spec,t) for t in step_times])
    np.testing.assert_allclose(a['force_step_commands_N'],replay,atol=1e-9)
    moment=np.cross(a['force_step_application_positions_m']-spec['camera']['center_m'],replay).sum(1)
    np.testing.assert_allclose(a['force_step_net_moments_Nm'],moment,atol=1e-12)
    if spec['force_type']=='twist_couple':np.testing.assert_array_equal(replay.sum(1),0)
    frame_commands=np.concatenate([np.zeros((1,2,3)),replay[hz//case['fps']-1::hz//case['fps']]])
    np.testing.assert_allclose(a['commanded_handle_forces_N'],frame_commands,atol=1e-9)
    np.testing.assert_allclose(a['lumped_node_mass_kg'],lumped_masses(points,tets,case['material']['density_kg_m3']),atol=1e-12)
    means=np.einsum('ni,fnij->fnj',reference['barycentric_weights'],q[:,reference['face_vertex_ids']])
    np.testing.assert_allclose(a['gaussian_means_m'],means,atol=1e-12)
    # Load patches are attached to rigid 1g handles; compare pairwise distances.
    patch_error=0.
    for patch in spec['load_patches']:
        d=np.linalg.norm(q[:,patch,None]-q[:,None,patch],axis=-1)
        rest=np.linalg.norm(points[patch,None]-points[None,patch],axis=-1)
        patch_error=max(patch_error,float(abs(d-rest).max()))
    require(patch_error<5e-4,'Rigid load-patch attachment drift')
    poses=np.array([camera_pose(spec,t) for t in times])
    np.testing.assert_allclose(c['camera_to_world_usd_row'],poses,atol=1e-12)
    np.testing.assert_allclose(c['times_s'],times,atol=1e-12)
    np.testing.assert_array_equal(c['frame_ids'],np.arange(n))
    expected_k=np.array([[640*35/36,0,320],[0,480*35/27,240],[0,0,1.]])
    np.testing.assert_allclose(c['K'],expected_k,atol=1e-12)
    sparse=case['sparse_frames'];inputs=load_model_inputs(root)
    np.testing.assert_array_equal(inputs['cameras']['frame_ids'],sparse)
    np.testing.assert_allclose(inputs['cameras']['camera_to_world_usd_row'],poses[sparse],atol=1e-12)
    np.testing.assert_allclose(inputs['cameras']['times_s'],times[sparse],atol=1e-12)
    np.testing.assert_array_equal(inputs['cameras']['K'],c['K'])
    protocol=json.loads((root/'observations/input_protocol.json').read_text())
    require(protocol['observed_frame_ids']==sparse,'Sparse input protocol')
    max_depth_error=0.;minimum_pixels=640*480;edge_frames=0
    for folder,suffix in [('rgb','.png'),('masks','.png'),('depth','.npz')]:
        require(len(list((root/'truth'/folder).glob('*'+suffix)))==n,'Truth frame count')
        require(len(list((root/'observations'/folder).glob('*'+suffix)))==len(sparse),'Input frame count')
        for frame in sparse:require(sha(root/'truth'/folder/f'{frame:05d}{suffix}')==sha(root/'observations'/folder/f'{frame:05d}{suffix}'),'Sparse copy differs from truth')
    for frame in range(n):
        image=np.asarray(Image.open(root/'truth/rgb'/f'{frame:05d}.png'))
        mask=np.asarray(Image.open(root/'truth/masks'/f'{frame:05d}.png'))>0
        with np.load(root/'truth/depth'/f'{frame:05d}.npz') as raw:depth=raw['depth_m']
        require(image.shape==(480,640,3) and mask.shape==depth.shape==(480,640),'Sensor shape')
        require(np.isfinite(depth).all() and np.all(depth[~mask]==0) and np.all(depth[mask]>.01),'Depth/mask mismatch')
        minimum_pixels=min(minimum_pixels,int(mask.sum()))
        edge_frames+=bool(mask[[0,-1]].any() or mask[:,[0,-1]].any())
        if frame%15==0:
            ys,xs=np.where(mask);sample=np.linspace(0,len(xs)-1,48).astype(int);ys=ys[sample];xs=xs[sample]
            hit=ray_depth(q[frame],faces,poses[frame],c['K'],xs,ys)
            require(np.isfinite(hit).sum()>=46,'Depth silhouette / camera / mesh mismatch')
            valid=np.isfinite(hit);error=float(abs(hit[valid]-depth[ys,xs][valid]).max())
            max_depth_error=max(max_depth_error,error)
    require(minimum_pixels>=1000 and not edge_frames,'Object too small or clipped')
    require(max_depth_error<2e-4,'Rendered depth inconsistent with saved camera and nodal mesh')
    require(summary['frames']==n and summary['observed_frames']==len(sparse),'Summary frame counts')
    require(summary['minimum_mask_pixels']==minimum_pixels,'Summary mask pixel count')
    require(summary['minimum_tet_volume_ratio']<=float(ratios.min())+1e-6,'Runtime/saved tet minimum mismatch')
    require(summary['maximum_constraint_error_m']>=bc_error-1e-7,'Runtime/saved boundary error mismatch')
    return {'passed':True,'episode_id':case['episode_id'],'minimum_saved_tet_volume_ratio':float(ratios.min()),
            'boundary_error_m':bc_error,'load_patch_distance_error_m':patch_error,
            'maximum_ray_depth_error_m':max_depth_error,'depth_rays_tested':240,
            'minimum_mask_pixels':minimum_pixels,'clipped_frames':int(edge_frames)}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('episode',type=Path)
    args=parser.parse_args();print(json.dumps(validate(args.episode),indent=2))
