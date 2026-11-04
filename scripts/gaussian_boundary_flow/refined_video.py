"""Actual versus predicted corner-surface geometry, fixed median/worst selection."""
from pathlib import Path
from collections import Counter
import json,subprocess
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scripts.gaussian_boundary_flow.model import Block


def main():
    root=Path('runs/gaussian-boundary-flow-refined-v03');out=Path('reports/gaussian-boundary-flow-v01')
    d=torch.load(root/'evaluation-reverse.pt',weights_only=True);truth=d['target'].numpy();pred=d['prediction']['flow'].numpy();b=Block();rest=b.mesh.rest.numpy()
    errors=np.sqrt(np.mean((truth[:,17:]-pred[:,17:])**2,axis=(1,2,3)))*1000;order=errors.argsort();ids=[int(order[len(order)//2]),int(order[-1])]
    faces=Counter(tuple(sorted(f)) for c in b.mesh.cells.tolist() for f in [(c[0],c[1],c[2]),(c[0],c[1],c[3]),(c[0],c[2],c[3]),(c[1],c[2],c[3])]);faces=np.array([f for f,n in faces.items()if n==1])
    ffmpeg=Path('runs/media-tooling/ffmpeg-7.0.2-amd64-static/ffmpeg')
    proc=subprocess.Popen([str(ffmpeg),'-y','-f','rawvideo','-pix_fmt','rgb24','-s','1200x700','-r','20','-i','-','-an','-c:v','libx264','-pix_fmt','yuv420p',str(out/'refined-actual-vs-predicted.mp4')],stdin=subprocess.PIPE,stderr=subprocess.DEVNULL)
    for t in range(41):
        fig=plt.figure(figsize=(12,7),dpi=100,facecolor='#f8fafc')
        fig.suptitle('Hidden boundary inference: a new force reversal',fontsize=17,fontweight='bold',y=.98)
        for row,i in enumerate(ids):
            for col,(values,label,color) in enumerate([(truth,'Actual simulation','#3b82f6'),(pred,'Predicted motion','#10b981')]):
                ax=fig.add_subplot(2,3,row*3+col+1,projection='3d');p=values[i,t]+rest
                # Plot physical x,z,y to keep gravity vertical on the page.
                p=p[:,[0,2,1]];surface=Poly3DCollection(p[faces],facecolors=color,edgecolors='#334155',linewidths=.65,alpha=.85);ax.add_collection3d(surface)
                ax.set(xlim=(-.07,.07),ylim=(-.07,.07),zlim=(.17,.32));ax.set_box_aspect((1,1,1));ax.view_init(elev=22,azim=35+45*t/40);ax.set_axis_off();ax.set_title(label,fontsize=12)
            ax=fig.add_subplot(2,3,row*3+3);e=np.sqrt(np.mean((truth[i]-pred[i])**2,axis=(1,2)))*1000;times=np.arange(41)*.025
            ax.plot(times,e,color='#7c3aed',lw=2);ax.axvspan(0,.4,color='#e2e8f0');ax.axvline(t*.025,color='#f59e0b');ax.set(xlim=(0,1),ylim=(0,max(e.max()*1.1,.1)),xlabel='Physical time (s)',ylabel='Coordinate RMSE (mm)');ax.spines[['top','right']].set_visible(False);ax.set_title(('Median' if row==0 else 'Worst')+f' case | future RMSE {errors[i]:.3f} mm',fontsize=11)
        fig.text(.5,.025,f't = {t*.025:.3f} s | '+('Observed prefix' if t<=16 else 'Forecast: force reversed')+' | true deformation scale; 0.5x playback',ha='center',fontsize=11)
        fig.subplots_adjust(top=.90,bottom=.12,wspace=.22,hspace=.28,left=.02,right=.97);fig.canvas.draw();frame=np.asarray(fig.canvas.buffer_rgba())[...,:3].copy();proc.stdin.write(frame.tobytes())
        if t==30:fig.savefig(out/'refined-preview.png',dpi=100)
        plt.close(fig)
    proc.stdin.close()
    if proc.wait()!=0:raise RuntimeError('ffmpeg failed')
    (out/'refined-video-provenance.json').write_text(json.dumps({'selection':'median and worst over all 128 held-out force-reversal cases','indices':ids,'future_rmse_mm':errors[ids].tolist(),'deformation_magnification':1,'surface':'Visualization interpolates the eight output corner nodes; not a photorealistic render or full fine-mesh surface','physics_nodes':64,'playback_speed':.5},indent=2))

if __name__=='__main__':main()
