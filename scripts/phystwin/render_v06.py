"""Transparent v06 visualizations: static posterior shapes and PhysTwin holds."""
from pathlib import Path
import json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import imageio_ffmpeg
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.phystwin.sequential_reference import BASE,OUT,protocol,write
from scripts.phystwin.sequential_fixture import points,ACTIONS
from scripts.phystwin.render_qualified_predictions import triangles
from scripts.phystwin.sequential_policy import verify
matplotlib.rcParams['animation.ffmpeg_path']=imageio_ffmpeg.get_ffmpeg_exe()
REST=points();FACES=triangles()

def figure(vmax,footer):
 fig=plt.figure(figsize=(15.36,7.68));axes=[fig.add_subplot(1,3,i+1,projection='3d') for i in range(3)];fig.subplots_adjust(left=.075,right=.94,bottom=.22,top=.74,wspace=.22)
 title=fig.suptitle('',fontsize=18);subtitle=fig.text(.5,.89,'',ha='center',fontsize=11);bar=fig.add_axes([.35,.105,.3,.025]);norm=Normalize(0,max(vmax,.001));cmap=plt.get_cmap('viridis')
 fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=bar,orientation='horizontal',label='Prediction position error (real mm; common scale)');fig.text(.5,.035,footer,ha='center',fontsize=10)
 return fig,axes,title,subtitle,norm,cmap

def draw(axes,positions,labels,bounds,norm,cmap,az):
 lo,hi=bounds
 for k,ax in enumerate(axes):
  ax.clear();errors=1000*np.linalg.norm(positions[k]-positions[0],axis=1);colors='#5d89aa' if k==0 else cmap(norm(errors[FACES].mean(1)))
  ax.add_collection3d(Poly3DCollection((1000*positions[k])[FACES],facecolors=colors,edgecolors='#4d5560',linewidths=.18))
  ax.add_collection3d(Poly3DCollection((1000*REST)[FACES],facecolors=(0,0,0,0),edgecolors=(.5,.5,.5,.2),linewidths=.2))
  ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(lo[2],hi[2]),xlabel='x (mm)',ylabel='y (mm)',zlabel='z (mm)');ax.set_box_aspect(hi-lo);ax.tick_params(labelsize=8);ax.set_title(labels[k],fontsize=12);ax.view_init(22,az)
  for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:axis.set_major_locator(MaxNLocator(4))
def bounds(positions):
 x=1000*np.concatenate([REST,*positions]);return x.min(0)-5,x.max(0)+5

def main():
 spec=protocol();OUT.mkdir(exist_ok=True,parents=True);directory=BASE/'analysis-rgbd';verify(json.loads((directory/'seal.json').read_text()));s=json.loads((directory/'test-summary.json').read_text());mse=np.array(s['policies']['adaptive']['object_mse'])[:,2];order=np.argsort(mse);cases=[]
 with np.load(directory/'test-adaptive.npz') as z:
  for label,obj in [('median',int(order[(len(order)-1)//2])),('worst',int(order[-1]))]:
   i=obj*spec['noise_repeats'];pos=[REST+z['truth'][i].reshape(81,3),REST+z['mean'][i,0].reshape(81,3),REST+z['mean'][i,2].reshape(81,3)];error=[0,*[float(1000*np.sqrt(np.mean((p-pos[0])**2))) for p in pos[1:]]];cases.append(dict(label=label,object=obj,positions=pos,errors=error,choices=z['choices'][i].tolist()))
 vmax=max(float(1000*np.linalg.norm(p-c['positions'][0],axis=1).max()) for c in cases for p in c['positions'][1:]);fig,axes,title,subtitle,norm,cmap=figure(vmax,'Static held-load prediction • camera motion only • deformation ×1 • RGB-D marker observations • first noise repeat')
 writer=FFMpegWriter(fps=12,codec='libx264',extra_args=['-threads','2','-crf','20','-pix_fmt','yuv420p'])
 with writer.saving(fig,str(OUT/'actual-vs-predicted.mp4'),dpi=100):
  for c in cases:
   title.set_text(f"RGB-D sequential probing: {c['label']}-error object")
   subtitle.set_text(f"Object {c['object']} • {spec['test'][c['object']]['layout'].replace('_',' ')} • probes: "+' then '.join(ACTIONS[a+1].replace('_',' ') for a in c['choices']))
   labels=['Reference FEM',f"Before probes\nRMSE {c['errors'][1]:.3f} mm",f"After two probes\nRMSE {c['errors'][2]:.3f} mm"]
   for frame in range(96):
    draw(axes,c['positions'],labels,bounds(c['positions']),norm,cmap,-70+40*np.sin(2*np.pi*frame/95))
    if frame==0:fig.savefig(OUT/f"prediction-{c['label']}.png",dpi=100)
    writer.grab_frame()
 plt.close(fig)
 bridge=[]
 for obj in range(3):
  arrays=[];scores=[]
  for budget in ['initial','two_probes']:
   folder=BASE/'phystwin'/f'object-{obj}-{budget}';arrays.append(np.load(folder/'held_oblique-dt1.npz'));scores.append(json.loads((folder/'summary-dt1.json').read_text())['results']['held_oblique']['hold_rmse_mm'])
  bridge.append(dict(object=obj,truth=arrays[0]['reference'],initial=arrays[0]['predicted'],after=arrays[1]['predicted'],scores=scores,times=arrays[0]['times']))
 vmax=max(float(1000*np.linalg.norm(c[k]-c['truth'],axis=2).max()) for c in bridge for k in ['initial','after']);fig,axes,title,subtitle,norm,cmap=figure(vmax,'Deformation ×1 • interpolated quasistatic reference ramp, not dynamic ground truth • quantitative scores use the hold')
 writer=FFMpegWriter(fps=5,codec='libx264',extra_args=['-threads','2','-crf','20','-pix_fmt','yuv420p'])
 with writer.saving(fig,str(OUT/'phystwin-held-force.mp4'),dpi=100):
  for c in bridge:
   title.set_text(f"PhysTwin visual bridge: object {c['object']}, held oblique load")
   limits=bounds([v for k in ['truth','initial','after'] for v in c[k]])
   for frame,t in enumerate(c['times']):
    phase='HOLD' if 1.4<=t<=3.4 else ('RAMP' if t<1.4 else 'RELEASE / RESET');subtitle.set_text(f"t={t:.1f} s • {phase} • both fits use visible RGB-D markers; held load excluded from fitting")
    labels=['Quasistatic reference',f"PhysTwin: initial interaction only\nHold RMSE {c['scores'][0]:.3f} mm",f"PhysTwin: plus two probes\nHold RMSE {c['scores'][1]:.3f} mm"]
    draw(axes,[c[k][frame] for k in ['truth','initial','after']],labels,limits,norm,cmap,-70+25*np.sin(2*np.pi*frame/(len(c['times'])-1)))
    if frame==15:fig.savefig(OUT/f"phystwin-object-{c['object']}.png",dpi=100)
    writer.grab_frame()
 plt.close(fig)
 write(OUT/'video-selection.json',dict(bayesian=[{k:v for k,v in c.items() if k!='positions'} for c in cases],selection='Median and worst object-mean RGBD adaptive two-probe MSE over three noise repeats; display first repeat. Not selected for visual success.',phystwin_objects=[0,1,2],scope='Bayesian video shows static held-load shapes with camera orbit. PhysTwin video compares dynamic spring rollouts with interpolated static reference ramps; hold errors are the quantitative metric. All deformation shown at scale1.'))
 print('VIDEOS_COMPLETE',flush=True)
if __name__=='__main__':main()
