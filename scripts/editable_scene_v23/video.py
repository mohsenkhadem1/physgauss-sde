"""Geometry-only visual inspection of the integration pilot (not real footage)."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'runs/editable-scene-v23'

def triangles(rest):
    axes=[np.unique(np.round(rest[:,i],10))for i in range(3)]
    lookup={tuple(np.round(p,10)):i for i,p in enumerate(rest)};faces=[]
    for normal in range(3):
        other=[i for i in range(3)if i!=normal]
        for side in (axes[normal][0],axes[normal][-1]):
            for i in range(len(axes[other[0]])-1):
                for j in range(len(axes[other[1]])-1):
                    ids=[]
                    for di,dj in ((0,0),(1,0),(1,1),(0,1)):
                        p=np.zeros(3);p[normal]=side;p[other[0]]=axes[other[0]][i+di];p[other[1]]=axes[other[1]][j+dj];ids.append(lookup[tuple(p)])
                    faces.extend([[ids[0],ids[1],ids[2]],[ids[0],ids[2],ids[3]]])
    return np.asarray(faces)

def main():
    data=[]
    for kind in ('lift_release','drop','force_pulse'):
        with np.load(BASE/f'{kind}-12000-dt5e-05/states.npz')as f:data.append({k:f[k].copy()for k in f.files})
    fig=plt.figure(figsize=(12,4),dpi=100);axes=[fig.add_subplot(1,3,i+1,projection='3d')for i in range(3)];surfaces=[]
    colors=['#279b98','#477ac3','#e69b32']
    for ax,d,color,title in zip(axes,data,colors,['Lift and release','Drop onto floor','Localized force pulse']):
        faces=triangles(d['rest']);poly=Poly3DCollection(d['positions'][0][faces],facecolors=color,edgecolors='#263b47',linewidths=.3,alpha=.96);ax.add_collection3d(poly);surfaces.append((poly,faces))
        ax.set(xlim=(-.02,.18),ylim=(-.075,.075),zlim=(-.095,.10));ax.set_box_aspect((1,.75,1));ax.set_title(title)
        ax.set_xticks([0,.1]);ax.set_yticks([-.05,0,.05]);ax.set_zticks([-.05,0,.05]);ax.view_init(elev=22,azim=-65);ax.set_xlabel('x (m)');ax.set_ylabel('y (m)');ax.set_zlabel('z (m)')
        if title!='Localized force pulse':ax.plot_surface(*np.meshgrid([-.02,.18],[-.075,.075]),np.full((2,2),-.08),color='#b7bdc2',alpha=.3)
    heading=fig.suptitle('Editable-scene integration pilot | specified 12 kPa material | t = 0.000 s',fontsize=11)
    fig.text(.5,.015,'Simulation geometry only. Known material and boundaries; no learned scene inference.',ha='center',fontsize=9)
    fig.subplots_adjust(left=.02,right=.98,bottom=.12,top=.85,wspace=.08)
    out=BASE/'editable-scene-pilot.mp4';ffmpeg=ROOT/'runs/media-tooling/ffmpeg-7.0.2-amd64-static/ffmpeg'
    with (BASE/'video-encode.log').open('w')as log:
        proc=subprocess.Popen([str(ffmpeg),'-y','-f','rawvideo','-pix_fmt','rgb24','-s','1200x400','-r','20','-i','-','-an','-c:v','libx264','-crf','20','-pix_fmt','yuv420p',str(out)],stdin=subprocess.PIPE,stderr=log)
        for t in range(len(data[0]['times'])):
            for (surface,faces),d in zip(surfaces,data):surface.set_verts(d['positions'][t][faces])
            heading.set_text(f'Editable-scene integration pilot | specified 12 kPa material | t = {data[0]["times"][t]:.3f} s')
            fig.canvas.draw();frame=np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy();proc.stdin.write(frame.tobytes())
            if t==80:fig.savefig(BASE/'pilot-preview.png')
        proc.stdin.close();assert proc.wait()==0
    print(out)

if __name__=='__main__':main()
