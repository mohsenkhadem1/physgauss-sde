"""Render all qualification cases and a labelled finite-model local video."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio_ffmpeg
plt.rcParams['animation.ffmpeg_path']=imageio_ffmpeg.get_ffmpeg_exe()
from matplotlib.animation import FuncAnimation,FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from experiments.campaign_clamped_deformation_v02 import geometry

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/gaussian-campaign-v01/clamped-deformation-v03'
OUT=ROOT/'runs/operator-dev/clamped-deformation-media-v03'


def run():
    OUT.mkdir(parents=True,exist_ok=True);d=json.loads((RUN/'summary.json').read_text());rows=d['rows']
    fig,ax=plt.subplots(1,3,figsize=(18,5.5),layout='constrained');labels=[f"{r['object_index']} {r['load'][:5]}"for r in rows];x=np.arange(len(rows))
    ax[0].bar(x,[1e6*r['volume_nonrigid_coordinate_RMS_m']for r in rows],color=['#178579'if r['deformation_passed']else'#bc5944'for r in rows]);ax[0].axhline(100,color='black',ls='--',label='100 µm signal gate');ax[0].axhline(20,color='gray',label='20 µm noise');ax[0].set_ylabel('Nonrigid coordinate RMS (µm)');ax[0].legend(fontsize=8)
    ax[1].bar(x,[r['nonrigid_to_total_RMS_ratio']for r in rows],color='#287ab1');ax[1].axhline(.3,color='black',ls='--');ax[1].set_ylabel('Nonrigid / total motion RMS')
    ax[2].bar(x,[1e6*r['independent_max_position_error_m']for r in rows],color=['#178579'if r['temporal_algebra_passed']else'#bc5944'for r in rows]);ax[2].axhline(2,color='black',ls='--');ax[2].set_ylabel('Independent max position error (µm)')
    for a in ax:a.set_xticks(x,labels,rotation=65,ha='right',fontsize=7);a.spines[['top','right']].set_visible(False)
    fig.suptitle('All 26 clamped cases: first 10 development, last 16 new settings — richer linear finite model\nSignal strength and timestep qualification are separate checks');fig.savefig(OUT/'qualification.png',dpi=160);plt.close(fig)
    X,tets,faces,fixed,_=geometry();u=np.load(RUN/'exact-state-LOCAL.npz')['displacement'];center=4
    fig=plt.figure(figsize=(12,6));axes=[fig.add_subplot(1,2,j+1,projection='3d')for j in range(2)];artists=[];scale=5
    edges=sorted(set(tuple(sorted((int(a),int(b))))for t in faces for a in t for b in t if a<b))
    for j,a in enumerate(axes):
        for aa,bb in edges:a.plot(*X[[aa,bb]].T,color='.65',lw=.45,alpha=.35)
        poly=Poly3DCollection(X[faces],facecolors=['#48a2bc'if j==0 else'#ebac57'],edgecolor='#244859',linewidth=.7,alpha=.5);a.add_collection3d(poly);artists.append(poly)
        a.scatter(*X[fixed].T,c='#9c3232',s=35,label='Fixed faces');a.set(xlim=(-.02,.40),ylim=(-.03,.12),zlim=(-.04,.16),xlabel='x (m)',ylabel='y (m)',zlabel='z (m)');a.set_box_aspect((.42,.15,.2));a.set_title(['Bending: 0.128 N','Stretching: 0.576 N'][j]);a.legend(loc='upper left',fontsize=8)
    caption=fig.text(.5,.035,'',ha='center');fig.suptitle('Both ends clamped — simulation — displacement shown at 5×\nFinite linear model; no prediction or physical validation claimed',fontsize=13)
    frames=121
    def update(k):
        idx=min(k//3,u.shape[2]-1)
        for j,(a,p)in enumerate(zip(axes,artists)):
            p.set_verts((X+scale*u[center,j,idx])[faces]);a.view_init(elev=20+8*np.sin(k/frames*2*np.pi),azim=-65+55*k/(frames-1))
        caption.set_text(f'Geometric-center material/support setting | physical time {idx*.05:.2f} s | gray: undeformed | red: clamp')
        return artists+[caption]
    anim=FuncAnimation(fig,update,frames=frames,interval=1000/30,blit=False);anim.save(OUT/'clamped-bending-stretching.mp4',writer=FFMpegWriter(fps=30,codec='libx264',extra_args=['-pix_fmt','yuv420p']),dpi=100);plt.close(fig)
    print(json.dumps(dict(plot=str(OUT/'qualification.png'),video=str(OUT/'clamped-bending-stretching.mp4'),all_cases_shown=True,video_displacement_scale=scale)))

if __name__=='__main__':run()
