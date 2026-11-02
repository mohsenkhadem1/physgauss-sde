"""Camera-orbit video of a static simulated equilibrium and its predictions.

No interpolated trajectory is passed off as a simulated rollout. All panels use
the same material queries and camera; numerical errors are in world coordinates.
"""
import os
os.environ.setdefault('MPLBACKEND','Agg')
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib.pyplot as plt
import imageio_ffmpeg
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scripts.compositional_neural_mechanics.mechanics import GeometrySpec

def boundary_quads(n):
    grid=np.arange(n**3).reshape(n,n,n);faces=[]
    for axis in range(3):
        for side in (0,n-1):
            face=np.take(grid,side,axis=axis)
            for i in range(n-1):
                for j in range(n-1):faces.append([face[i,j],face[i+1,j],face[i+1,j+1],face[i,j+1]])
    return np.asarray(faces)

def render(geometry,paths,labels,output,frames=144):
    plt.rcParams['animation.ffmpeg_path']=imageio_ffmpeg.get_ffmpeg_exe()
    output=Path(output)
    if output.exists():raise ValueError('Preserve existing video')
    fields=[]
    for path in paths:
        with np.load(path,allow_pickle=False) as data:fields.append(data['field'].copy())
    n=round(len(fields[0])**(1/3))
    if n**3!=len(fields[0]) or any(x.shape!=fields[0].shape for x in fields):raise ValueError('Matching uniform material cube queries required')
    q=np.stack(np.meshgrid(*([np.linspace(0,1,n)]*3),indexing='ij'),-1).reshape(-1,3)
    rest=geometry.map(q);positions=[(rest+u)*1000 for u in fields];faces=boundary_quads(n)
    errors=[np.linalg.norm(u-fields[0],axis=-1)*1000 for u in fields]
    maximum=max(max(e) for e in errors[1:]) if len(errors)>1 else 1.
    maximum=max(maximum,.01);normal=plt.Normalize(0,maximum)
    cmap=LinearSegmentedColormap.from_list('position_error',['#68b6c8','#f0bd62','#cc4b47'])
    fig=plt.figure(figsize=(5*len(paths),5),facecolor='#f8fafc');axes=[]
    allp=np.concatenate(positions);center=(allp.min(0)+allp.max(0))/2;radius=(allp.max(0)-allp.min(0)).max()*.58
    for i,(p,label,error) in enumerate(zip(positions,labels,errors)):
        ax=fig.add_subplot(1,len(paths),i+1,projection='3d');axes.append(ax)
        colors='#58a6b8' if i==0 else cmap(normal(error[faces].mean(-1)))
        ax.add_collection3d(Poly3DCollection(p[faces],facecolors=colors,edgecolor='#243746',linewidth=.35,alpha=1))
        for dimension,setter in enumerate((ax.set_xlim,ax.set_ylim,ax.set_zlim)):setter(center[dimension]-radius,center[dimension]+radius)
        ax.set_box_aspect((1,1,1),zoom=1.35);ax.set_axis_off()
        rms=1000*np.sqrt(np.mean((fields[i]-fields[0])**2))
        ax.set_title(label+('\nNumerical reference' if i==0 else f'\nField RMS error: {rms:.3f} mm'),fontsize=12,pad=0)
    fig.suptitle('Static equilibrium • camera orbit • simulation and prediction',fontsize=14)
    fig.text(.5,.02,'Colour on predictions: pointwise position error (mm); identical camera and scale',ha='center',fontsize=10)
    if len(paths)>1:
        bar=fig.add_axes([.36,.105,.28,.025]);fig.colorbar(plt.cm.ScalarMappable(norm=normal,cmap=cmap),cax=bar,orientation='horizontal')
    output.parent.mkdir(parents=True,exist_ok=True)
    writer=FFMpegWriter(fps=24,bitrate=2400)
    with writer.saving(fig,str(output),100):
        for frame in range(frames):
            for ax in axes:ax.view_init(elev=22+8*np.sin(2*np.pi*frame/frames),azim=30+360*frame/frames)
            if frame==0:fig.savefig(output.with_suffix('.png'),dpi=120)
            writer.grab_frame()
    plt.close(fig)
    metadata=dict(scope=__doc__,geometry=vars(geometry),frames=frames,fps=24,
                  inputs=[dict(path=str(path),label=label,sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest()) for path,label in zip(paths,labels)],
                  field_RMS_errors_mm=[float(1000*np.sqrt(np.mean((u-fields[0])**2))) for u in fields],
                  video_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    output.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--scene',required=True);parser.add_argument('--paths',nargs='+',required=True);parser.add_argument('--labels',nargs='+',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();scene=json.loads(Path(args.scene).read_text())
    if len(args.paths)!=len(args.labels):raise ValueError('One label per input')
    render(GeometrySpec(**scene['geometry']),args.paths,args.labels,args.output)
