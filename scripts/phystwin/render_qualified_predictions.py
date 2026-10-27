"""Honest static held-force comparison: camera motion, not simulated dynamics."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import imageio_ffmpeg
from scripts.phystwin.qualified_reference import BASE,ROOT,protocol,observation_points,write_json
from scripts.phystwin.qualified_policy import verify_seal


def triangles():
    grid=np.arange(81).reshape(9,3,3);faces=[]
    for axis in range(3):
        other=[i for i in range(3) if i!=axis]
        for side in [0,grid.shape[axis]-1]:
            for i in range(grid.shape[other[0]]-1):
                for j in range(grid.shape[other[1]]-1):
                    vertices=[]
                    for a,b in [(0,0),(1,0),(1,1),(0,1)]:
                        index=[0,0,0];index[axis]=side;index[other[0]]=i+a;index[other[1]]=j+b;vertices.append(grid[tuple(index)])
                    faces.extend([[vertices[k] for k in [0,1,2]],[vertices[k] for k in [0,2,3]]])
    return np.array(faces)


def main():
    spec=protocol();analysis=BASE/'analysis';verify_seal(json.loads((analysis/'seal.json').read_text()))
    out=ROOT/'reports/phystwin-qualified-support-v05';out.mkdir(parents=True,exist_ok=True)
    object_errors=json.loads((analysis/'test-object-mse.json').read_text())['adaptive'];order=np.argsort(object_errors)
    selection=[('median',int(order[(len(order)-1)//2])),('worst',int(order[-1]))]
    with np.load(analysis/'test-predictions.npz') as z:
        cases=[]
        for label,i in selection:
            row=i*spec['noise_repeats'];choice=int(z['adaptive'][row]);truth=z['truth'][row].reshape(81,3)
            fields=[truth,z['0_mean'][row].reshape(81,3),z[f'{choice}_mean'][row].reshape(81,3)]
            cases.append(dict(label=label,object=i,action=choice,fields=fields))
    rest=observation_points();faces=triangles()
    vmax=max(float(np.linalg.norm(c['fields'][k]-c['fields'][0],axis=1).max()*1000) for c in cases for k in [1,2]);vmax=max(vmax,.001)
    norm=Normalize(0,vmax);cmap=plt.get_cmap('viridis')
    fig=plt.figure(figsize=(15.36,7.68));axes=[fig.add_subplot(1,3,i+1,projection='3d') for i in range(3)]
    fig.subplots_adjust(left=.02,right=.98,bottom=.2,top=.82,wspace=.04)
    title=fig.suptitle('',fontsize=18);subtitle=fig.text(.5,.86,'',ha='center',fontsize=11)
    bar=fig.add_axes([.35,.105,.3,.025]);fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=bar,orientation='horizontal',label='Prediction position error (mm; same scale in both cases)')
    fig.text(.5,.035,'Static held-force response • camera orbit only • deformation ×1 • first noise repeat • fixture hidden from policies',ha='center',fontsize=11)
    matplotlib.rcParams['animation.ffmpeg_path']=imageio_ffmpeg.get_ffmpeg_exe()
    writer=FFMpegWriter(fps=12,codec='libx264',extra_args=['-threads','2','-crf','20','-pix_fmt','yuv420p']);records=[]
    with writer.saving(fig,str(out/'actual-vs-predicted.mp4'),dpi=100):
        for c in cases:
            fields=c['fields'];positions=[1000*(rest+u) for u in fields];all_points=np.concatenate([1000*rest,*positions]);lo=all_points.min(0)-5;hi=all_points.max(0)+5
            errors=[0.]+[float(1000*np.sqrt(np.mean((u-fields[0])**2))) for u in fields[1:]]
            title.set_text(f"Held-force prediction: {c['label']}-error test object")
            subtitle.set_text(f"Object {c['object']} • {spec['test'][c['object']]['layout'].replace('_',' ')} support • chosen probe: {'left' if c['action']==1 else 'right'}")
            labels=['Reference FEM','Prediction before probe','Prediction after probe']
            for k,ax in enumerate(axes):
                ax.clear();color='#5d89aa' if k==0 else cmap(norm(1000*np.linalg.norm(fields[k]-fields[0],axis=1)[faces].mean(1)))
                ax.add_collection3d(Poly3DCollection(positions[k][faces],facecolors=color,edgecolors='#4d5560',linewidths=.15))
                ax.add_collection3d(Poly3DCollection((1000*rest)[faces],facecolors=(0,0,0,0),edgecolors=(.5,.5,.5,.15),linewidths=.2))
                ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(lo[2],hi[2]),xlabel='x (mm)',ylabel='y (mm)',zlabel='z (mm)')
                ax.set_box_aspect(hi-lo);ax.tick_params(labelsize=8);ax.set_title(labels[k]+('' if k==0 else f'\nCoordinate RMSE {errors[k]:.3f} mm'),fontsize=13)
            for frame in range(120):
                phase=frame/119;az=-70+45*np.sin(2*np.pi*phase);elev=22+5*np.sin(2*np.pi*phase)
                for ax in axes:ax.view_init(elev=elev,azim=az)
                if frame==0:fig.savefig(out/f'prediction-{c["label"]}.png',dpi=100)
                writer.grab_frame()
            records.append(dict(selection=c['label'],object=c['object'],noise_repeat=0,chosen_action=c['action'],before_rmse_mm=errors[1],after_rmse_mm=errors[2]))
    plt.close(fig)
    write_json(out/'video-selection.json',dict(cases=records,selection_rule='Median and worst adaptive object-mean test MSE across four noise repeats; show the first repeat for each. Selected only after sealed evaluation.',scope='Camera orbit of static sampled FEM and posterior-mean shapes under the held force. Not temporal forecasting or a dynamic simulation. Deformation is not magnified.',error_color_max_mm=vmax))
    report=out/'REPORT.md';text=report.read_text();marker='## Actual versus predicted shapes'
    if marker not in text:report.write_text(text+'\n'+marker+'\n\n[Watch the comparison](actual-vs-predicted.mp4). It shows the median and worst test objects by adaptive prediction error, before and after the chosen probe. The camera moves around static held-force shapes; this is not a dynamics forecast. Deformation is shown at its actual scale. [Selection details](video-selection.json).\n')
    print(json.dumps(records),flush=True)

if __name__=='__main__':main()
