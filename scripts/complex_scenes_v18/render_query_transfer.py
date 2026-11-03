"""Actual/reference versus prediction development clip, with an orbiting camera."""
import os
os.environ['MPLBACKEND']='Agg'
import hashlib,json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import imageio_ffmpeg
from scripts.complex_scenes_v18.pair_contact import material_surface
from scripts.complex_scenes_v18.run_development import ROOT,write


def main():
    base=ROOT/'runs/complex-scenes-v18/query-transfer-development';out=ROOT/'output/complex-scenes-v18';out.mkdir(parents=True,exist_ok=True)
    movie=out/'changed-interaction-actual-versus-predicted.mp4'
    if movie.exists():raise ValueError('Preserve existing video')
    paths=[base/'reference/clamp_release-v1-n2-dt0.00002500-s5/trajectory.npz',base/'ours/clamp_release-v1/trajectory.npz']
    arrays=[np.load(p)for p in paths];times=arrays[0]['times_s'];positions=[a['surface_positions_m'][:,0]for a in arrays]
    if positions[0].shape!=positions[1].shape or not np.allclose(times,arrays[1]['times_s'],atol=1e-9):raise ValueError('Clock/shape mismatch')
    q,faces=material_surface(9);mid=q[faces].mean(1);checker=(np.floor(mid[:,0]*8)+np.floor(mid[:,1]*4)+np.floor(mid[:,2]*4))%2
    fig=plt.figure(figsize=(12.8,6),dpi=100);axes=[fig.add_subplot(1,2,i+1,projection='3d')for i in range(2)]
    fig.subplots_adjust(left=.01,right=.99,top=.86,bottom=.13,wspace=.02)
    fig.suptitle('Changed loading and clamp locations — no material refit',fontsize=17,y=.97)
    fig.text(.5,.04,'Development example • coarse simulation reference • ~20× slow motion • deformation is not amplified',ha='center',fontsize=10)
    text=fig.text(.5,.90,'',ha='center',fontsize=11)
    writer=imageio_ffmpeg.write_frames(str(movie),(1280,600),fps=24,codec='libx264',quality=8,macro_block_size=1);writer.send(None)
    try:
        for frame,t in enumerate(times):
            text.set_text(f'Simulated time: {t:.3f} s')
            for index,(ax,x,title,color)in enumerate(zip(axes,positions,['Simulation reference','Our prediction'],[np.array([.16,.48,.73]),np.array([.90,.43,.16])])):
                ax.clear();shade=.8+.2*checker;collection=Poly3DCollection(x[frame][faces],facecolors=np.clip(shade[:,None]*color,0,1),edgecolors=(0,0,0,.08),linewidths=.15)
                ax.add_collection3d(collection);ax.set(xlim=(-.01,.13),ylim=(-.065,.065),zlim=(-.065,.065));ax.set_box_aspect((1.4,1.3,1.3))
                ax.view_init(elev=22,azim=-70+45*frame/max(len(times)-1,1));ax.set_axis_off();ax.set_title(title,fontsize=15,pad=0)
            fig.canvas.draw();rgb=np.asarray(fig.canvas.buffer_rgba())[...,:3].copy();writer.send(rgb)
            if frame==round(.75*(len(times)-1)):fig.savefig(out/'changed-interaction-preview.png',dpi=100)
    finally:writer.close();plt.close(fig)
    record=dict(scope='Development visualization; not held benchmark or rendering-quality score',movie_path=str(movie.relative_to(ROOT)),movie_sha256=hashlib.sha256(movie.read_bytes()).hexdigest(),frames=len(times),fps=24,physical_duration_s=float(times[-1]),deformation_scale=1,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths})
    write(ROOT/'reports/complex-scenes-v18/query-transfer-video.json',record);print(json.dumps(record))


if __name__=='__main__':main()
