"""Export only calibration tracks and public geometry/environment to estimators."""
import json
from pathlib import Path
import numpy as np
from scripts.compositional_dynamics_v17.native_phystwin_inputs import export_prefix
from scripts.compositional_dynamics_v17.render_sequence import render_frame
from scripts.physical_refinement_v15.observations import Camera,box_mesh
from scripts.compositional_dynamics_v17.prototype import ROOT,REPORT,sha,write,face_grid

SOURCE=ROOT/'runs/compositional-dynamics-v17/calibration-development/calibration/trajectory.npz'
OUT=ROOT/'runs/compositional-dynamics-v17/calibration-public'
CAMERAS=[Camera((.23,-.30,.17),(.06,0,0),256,192,300.),Camera((-.15,.24,.13),(.06,0,0),256,192,300.),Camera((.28,.18,.16),(.06,0,0),256,192,300.)]
PUBLIC_KEYS={'object_points','object_colors','object_visibilities','object_motions_valid','controller_points','surface_points','interior_points'}


def queries():
    allq=np.stack(np.meshgrid(*([np.linspace(0,1,5)]*3),indexing='ij'),-1).reshape(-1,3)
    surface=np.unique(allq[np.any((allq==0)|(allq==1),axis=1)],axis=0)
    interior=np.stack(np.meshgrid(*([np.array([.25,.5,.75])]*3),indexing='ij'),-1).reshape(-1,3)
    return surface,interior


def material_map(q):return np.asarray(q)*[.12,.06,.06]+[0,-.03,-.03]


def observation_visibility(points,frame,camera):
    eye,R=camera.pose();q=(points-eye)@R.T
    uv=q[:,:2]/q[:,2,None]*camera.focal+[camera.width/2,camera.height/2]
    ij=np.floor(uv).astype(int);inside=(q[:,2]>0)&(ij[:,0]>=0)&(ij[:,0]<camera.width)&(ij[:,1]>=0)&(ij[:,1]<camera.height)
    visible=np.zeros(len(points),bool);indices=np.flatnonzero(inside)
    depth=frame['depth'][ij[indices,1],ij[indices,0]]
    visible[indices]=np.isfinite(depth)&(abs(depth-q[indices,2])<=.002)
    return visible


def main():
    if OUT.exists():raise ValueError('Preserve public observations')
    completed=json.loads(SOURCE.with_name('complete.json').read_text())
    if not completed['completed'] or completed['trajectory_sha256']!=sha(SOURCE):raise ValueError('Complete pinned calibration required')
    # This function reads no forecast file or private generator plan.
    surface_q,interior_q=queries();denseq=face_grid(17)
    indices=np.array([np.flatnonzero(np.all(denseq==q,axis=1))[0] for q in surface_q])
    with np.load(SOURCE) as raw:
        clock=raw['times_s'];selected=np.arange(0,201,10)
        if len(clock)!=201 or not np.allclose(clock[selected],np.arange(21)*.01,atol=1e-12,rtol=0):raise ValueError('Exact declared calibration clock required')
        faces=raw['surface_frames'][selected];tracks=faces.reshape(21,-1,3)[:,indices]
    floor_z=-.04;floor=box_mesh([0,0,floor_z-.001],[.45,.4,.002],[.75,.77,.79]);visibility=np.zeros((21,98),bool)
    per_camera=[]
    for camera in CAMERAS:
        masks=[]
        for i,face in enumerate(faces):
            frame=render_frame(face,camera,[floor],time_s=float(i*.01))
            masks.append(observation_visibility(tracks[i],frame,camera))
        masks=np.asarray(masks);visibility|=masks;per_camera.append(masks.sum(axis=1).tolist())
    masses=np.full(125,.27/125)
    data,base=export_prefix(initial_tracked_m=material_map(surface_q),observed_prefix_m=tracks,visible_prefix=visibility,controller_commands_m=np.empty((21,0,3)),other_surface_m=np.empty((0,3)),interior_m=material_map(interior_q),object_masses_kg=masses,frame_dt_s=.01)
    if set(data)!=PUBLIC_KEYS:raise ValueError('Unexpected estimator input fields')
    OUT.mkdir(parents=True);np.savez_compressed(OUT/'observations.npz',**data)
    metadata=dict(base,scope='Development noiseless oracle 3D surface tracks with raster-depth visibility, known canonical geometry and total mass. Not raw-video reconstruction. No material identity/parameters or forecast response supplied.',
                  public_floor_z_m=floor_z,gravity_vector_m_s2=[0,0,-9.8],initial_velocity_m_s=[0,0,0],
                  total_mass_kg_check=.27,native_mass_discretization='Equal known SI mass per125objectpoints; differs from continuum consistent-Q2 inertia',
                  material_queries=surface_q.tolist(),interior_queries=interior_q.tolist(),cameras=[vars(c) for c in CAMERAS],visibility_rule='Union of3fixed camera raster-depth tests; nearestpixel optical-depth tolerance2mm; no futurevisibility input',
                  visible_count_per_frame=visibility.sum(axis=1).tolist(),per_camera_visible_counts=per_camera,
                  observations_sha256=sha(OUT/'observations.npz'),forecast_public_setup=dict(duration_s=.25,frame_dt_s=.01,floor_z_m=-.07,initial_velocity_m_s=[0,0,0],initial_geometry_same=True,gravity_vector_m_s2=[0,0,-9.8]))
    write(OUT/'metadata.json',metadata)
    write(REPORT/'calibration-observation-export.json',dict(public_metadata_sha256=sha(OUT/'metadata.json'),public_npz_sha256=sha(OUT/'observations.npz'),input_calibration_sha256=sha(SOURCE),source_sha256=sha(__file__),fields=sorted(data),no_forecast_read=True,no_private_material_fields=True))
    print(json.dumps(dict(output=str(OUT),frames=21,points=98,min_visible=int(visibility.sum(1).min()))))


if __name__=='__main__':main()
