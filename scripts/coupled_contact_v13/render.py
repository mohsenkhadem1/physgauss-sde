"""Fixed-case reference/prediction geometry movies; real metres, quarter-speed playback."""
from pathlib import Path
import sys,json,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scripts.coupled_contact_v13.campaign import BASE,REPORT,ROOT,write
from scripts.coupled_contact_v13.scenarios import build,command

def boundary(tets):
    faces=np.concatenate([tets[:,list(ids)] for ids in [(0,1,2),(0,1,3),(0,2,3),(1,2,3)]])
    keys=np.sort(faces,axis=1);_,idx,counts=np.unique(keys,axis=0,return_index=True,return_counts=True)
    return faces[idx[counts==1]]

def main():
    import imageio_ffmpeg
    matplotlib.rcParams['animation.ffmpeg_path']=imageio_ffmpeg.get_ffmpeg_exe()
    p=json.loads((REPORT/'protocol.json').read_text());out=[]
    for object_id,scenario in p['video_cases']:
        refpath=BASE/'reference'/object_id/scenario/'trajectory.npz';predpath=BASE/'evaluation'/object_id/scenario/'predictions.npz'
        if not refpath.exists() or not predpath.exists():continue
        with np.load(refpath) as d:reference=d['frames'].copy();times=d['times'].copy();faces=boundary(d['tets']);reference_centers=d['centers'].copy()
        with np.load(predpath) as d:predictions={k:d[k].copy() for k in d.files}
        coarse=build(scenario,cells=tuple(p['coarse_cells']));coarse_faces=boundary(coarse.tets)
        methods=[('Reference FEM',reference,faces,reference_centers)]
        for name,label in [('fem','Inferred FEM'),('native','Adapted spring core'),('rigid','Rigid + same contact')]:
            centers=None;q=BASE/'prediction'/object_id/scenario/name/'trajectory.npz'
            if q.exists():
                with np.load(q) as d:centers=d['centers'].copy()
            methods.append((label,predictions.get(name),coarse_faces,centers))
        path=REPORT/f'{object_id}-{scenario}.mp4';fig=plt.figure(figsize=(16,5.4));axes=[fig.add_subplot(1,4,i+1,projection='3d') for i in range(4)]
        allxyz=np.concatenate([x.reshape(-1,3) for _,x,_,_ in methods if x is not None]);low=np.min(allxyz,axis=0)-.02;high=np.max(allxyz,axis=0)+.02
        # Common world-coordinate view across all methods; no individual alignment.
        span=max(high-low);center=(high+low)/2;low=center-span/2;high=center+span/2
        writer=FFMpegWriter(fps=25,bitrate=2200,extra_args=['-threads','2'])
        with writer.saving(fig,str(path),100):
            for frame,t in enumerate(times):
                for ax,(label,x,tri,centers) in zip(axes,methods):
                    ax.clear()
                    if x is not None:
                        ax.add_collection3d(Poly3DCollection(x[frame][tri],facecolor='#49a7ac',edgecolor='#23646a',linewidth=.35,alpha=.88))
                    else:ax.text2D(.1,.5,'Forecast failed',transform=ax.transAxes,color='red')
                    for j,c in enumerate(coarse.colliders):
                        ccenter=centers[frame,j] if centers is not None else c.center
                        if c.name in command(scenario,t):ccenter=command(scenario,t)[c.name]['center']
                        if c.kind=='sphere':
                            u,v=np.meshgrid(np.linspace(0,2*np.pi,16),np.linspace(0,np.pi,10));ball=c.radius*np.array([np.cos(u)*np.sin(v),np.sin(u)*np.sin(v),np.cos(v)])
                            ax.plot_surface(ball[0]+ccenter[0],ball[1]+ccenter[1],ball[2]+ccenter[2],color='#cf7756' if c.name=='payload' else '#7b8290',alpha=.35,linewidth=0)
                        elif c.kind=='plane' and c.name=='floor':
                            xx,yy=np.meshgrid([low[0],high[0]],[low[1],high[1]]);ax.plot_surface(xx,yy,np.zeros_like(xx),color='#aeb2ba',alpha=.12)
                        elif c.kind=='box':
                            corners=np.array([[i,j,k] for i in [-1,1] for j in [-1,1] for k in [-1,1]])*c.half_extents+ccenter
                            quads=[[0,1,3,2],[4,5,7,6],[0,1,5,4],[2,3,7,6],[0,2,6,4],[1,3,7,5]]
                            ax.add_collection3d(Poly3DCollection(corners[quads],facecolor='#858a94',alpha=.3))
                    ax.set(xlim=(low[0],high[0]),ylim=(low[1],high[1]),zlim=(low[2],high[2]),title=label,xlabel='x (m)',ylabel='y (m)',zlabel='z (m)');ax.set_box_aspect((1,1,1));ax.view_init(23,-60+25*frame/max(len(times)-1,1))
                fig.suptitle(f'{object_id}: {scenario} | physical time {t:.2f} s | playback 4x slower | same world scale',fontsize=13)
                writer.grab_frame()
        plt.close(fig)
        out.append(dict(path=str(path.relative_to(ROOT)),frames=len(times),playback_fps=25,simulation_fps=p['fps'],reference_sha256=hashlib.sha256(refpath.read_bytes()).hexdigest(),prediction_sha256=hashlib.sha256(predpath.read_bytes()).hexdigest(),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    write(REPORT/'video-provenance.json',out)
if __name__=='__main__':main()
