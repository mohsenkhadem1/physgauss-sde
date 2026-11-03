"""All-family variant-0 development videos with unamplified motion and orbit.

Fixed variant selection; comparison is temporal forecasting after a visible
calibration prefix, not the forthcoming held changed-interaction benchmark.
"""
import os
os.environ['MPLBACKEND']='Agg'
import argparse,hashlib,json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import imageio_ffmpeg
from scripts.complex_scenes_v18.run_development import ROOT,write
from scripts.complex_scenes_v18.scenarios import FAMILIES,build_case
from scripts.complex_scenes_v18.pair_contact import material_surface
from scripts.complex_scenes_v18.geometry import RigidBowl,RigidCylinder,RigidSphere


def surroundings(ax,scene,t,rigid):
    u=np.linspace(0,2*np.pi,25);v=np.linspace(0,np.pi,13)
    for collider in scene.models[0].colliders:
        center=np.asarray(collider.center)
        if collider.name in scene.rigid:center=rigid[list(scene.rigid).index(collider.name)]
        if isinstance(collider,RigidSphere):
            x=center[0]+collider.radius*np.outer(np.cos(u),np.sin(v));y=center[1]+collider.radius*np.outer(np.sin(u),np.sin(v));z=center[2]+collider.radius*np.outer(np.ones_like(u),np.cos(v))
        elif isinstance(collider,RigidCylinder):
            a,b=np.meshgrid(u,np.linspace(-.07,.07,7));x=center[0]+collider.radius*np.cos(a);y=center[1]+b;z=center[2]+collider.radius*np.sin(a)
        elif isinstance(collider,RigidBowl):
            a,r=np.meshgrid(u,np.linspace(0,.105,9));x=center[0]+r*np.cos(a);y=center[1]+r*np.sin(a);z=center[2]+r*r/(2*collider.curvature_radius)
        else:continue
        ax.plot_wireframe(x,y,z,color='#64748b',alpha=.5,linewidth=.5)
    if scene.models[0].support_mode=='bilateral':
        motion=scene.motions[0];poses=np.zeros((2,6));gates=np.ones(2)
        if motion is not None:poses,_,gates,_=motion(t)
        for side,spec in enumerate(scene.models[0].attachments):
            point=np.array([side*.12,spec.center[0]*.06-.03,spec.center[1]*.06-.03])+poses[side,:3]
            ax.scatter(*point,marker='x',color='#111827',s=70,alpha=float(.15+.85*gates[side]),linewidths=2)


def render(case,prediction_label='material-scene-fit-01'):
    base=ROOT/'runs/complex-scenes-v18';ref=list((base/'pilot-01').glob(f'{case}-v0-*/trajectory.npz'))
    if len(ref)!=1:raise ValueError('Unique development reference required')
    pred=base/prediction_label/f'{case}-v0'/'sealed-rollout'/f'{case}-v0'/'trajectory.npz';paths=[ref[0],pred]
    for path in paths:
        done=json.loads((path.parent/'complete.json').read_text())
        if not done['completed']or hashlib.sha256(path.read_bytes()).hexdigest()!=done['trajectory_sha256']:raise ValueError('Complete intact trajectories required')
    arrays=[np.load(p)for p in paths];times=arrays[0]['times_s'];x=[np.real_if_close(a['surface_positions_m'])for a in arrays]
    if x[0].shape!=x[1].shape or not np.allclose(times,arrays[1]['times_s'],atol=1e-9):raise ValueError('Matched complete sample clocks required')
    scene,_,_=build_case(case,variant=0);q,faces=material_surface(9);mid=q[faces].mean(1);checker=(np.floor(mid[:,0]*8)+np.floor(mid[:,1]*4)+np.floor(mid[:,2]*4))%2
    cut=json.loads((base/'development-prefixes'/f'{case}-v0'/'metadata.json').read_text())['calibration_duration_s']
    out=ROOT/'output/complex-scenes-v18';out.mkdir(exist_ok=True,parents=True);movie=out/f'development-{case}-v0-actual-predicted.mp4'
    if movie.exists():raise ValueError('Preserve existing video')
    fig=plt.figure(figsize=(12.8,6),dpi=100);axes=[fig.add_subplot(1,2,i+1,projection='3d')for i in range(2)];fig.subplots_adjust(left=.01,right=.99,top=.84,bottom=.12,wspace=.02)
    titles=dict(clamp_release='Moving clamps and release',curved_push='Pushing around a curved obstacle',container_drop='Gravity drop into a curved container',rigid_impact='Moving rigid striker',deformable_pair='Two deformable objects colliding',heterogeneous='Spatially varying material stiffness')
    fig.suptitle(titles[case],fontsize=18,y=.97);clock=fig.text(.5,.90,'',ha='center',fontsize=11)
    fig.text(.5,.035,'Development • coarse simulation reference • 20.8× slow motion • true deformation scale • orbiting camera',ha='center',fontsize=10)
    writer=imageio_ffmpeg.write_frames(str(movie),(1280,600),fps=24,codec='libx264',quality=8,macro_block_size=1);writer.send(None)
    try:
        for frame,t in enumerate(times):
            clock.set_text(f'{t:.3f} s | '+('Observed calibration prefix'if t<=cut+1e-10 else'Future forecast — no further observations'))
            for j,ax in enumerate(axes):
                ax.clear()
                for body in range(x[j].shape[1]):
                    color=np.array([.15,.48,.75]if j==0 else[.92,.43,.17]);shade=(.75+.25*checker)*(1-.18*body)
                    ax.add_collection3d(Poly3DCollection(x[j][frame,body][faces],facecolors=np.clip(shade[:,None]*color,0,1),edgecolors=(0,0,0,.06),linewidths=.12))
                rigid=arrays[j]['rigid_centers_m'][frame]if 'rigid_centers_m'in arrays[j]else np.empty((0,3))
                if scene.rigid and len(rigid)!=len(scene.rigid):raise ValueError('Moving striker state required in both panels')
                surroundings(ax,scene,t,rigid)
                bounds=((-0.03,.33),(-.12,.12),(-.12,.12))if case=='deformable_pair'else((-.05,.17),(-.11,.11),(-.13,.13))
                ax.set(xlim=bounds[0],ylim=bounds[1],zlim=bounds[2]);ax.set_box_aspect([b-a for a,b in bounds]);ax.view_init(elev=23,azim=-70+50*frame/max(len(times)-1,1));ax.set_axis_off();ax.set_title('Simulation reference'if j==0 else'Our prediction',fontsize=15)
            fig.canvas.draw();writer.send(np.asarray(fig.canvas.buffer_rgba())[...,:3].copy())
            if frame==round(.8*(len(times)-1)):fig.savefig(out/f'development-{case}-preview.png',dpi=100)
    finally:writer.close();plt.close(fig)
    result=dict(case=case,variant=0,scope='Development temporal forecast, not held interaction transfer or image quality comparison',movie=str(movie.relative_to(ROOT)),movie_sha256=hashlib.sha256(movie.read_bytes()).hexdigest(),calibration_duration_s=cut,physical_duration_s=float(times[-1]),frames=len(times),fps=24,deformation_scale=1,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths})
    write(ROOT/'reports/complex-scenes-v18'/f'video-{case}-development.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',choices=FAMILIES,required=True);p.add_argument('--prediction-label',default='material-scene-fit-01');a=p.parse_args();print(json.dumps(render(a.case,a.prediction_label)))
