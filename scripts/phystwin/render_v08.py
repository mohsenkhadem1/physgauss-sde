"""Median and worst strong-load cost-aware static held-load predictions, at true scale."""
from pathlib import Path
import json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.phystwin.render_v06 import figure,draw,bounds,REST,plt,FFMpegWriter
from scripts.phystwin.v08_study import BASE,OUT,protocol,write

def bayesian():
 s=protocol();z=np.load(BASE/'test-cost_aware-65.npz');mse=((z['mean'][:,-1]-z['truth'])**2).mean(1).reshape(24,3).mean(1);order=np.argsort(mse);plans=json.loads((BASE/'plans/test-cost_aware.json').read_text());cases=[]
 for label,obj in [('median',int(order[11])),('worst',int(order[-1]))]:
  row=3*obj;positions=[REST+z['truth'][row].reshape(81,3),REST+z['mean'][row,0].reshape(81,3),REST+z['mean'][row,-1].reshape(81,3)];errors=[float(1000*np.sqrt(np.mean((p-positions[0])**2))) for p in positions];cases.append(dict(label=label,object=obj,positions=positions,errors=errors,history=plans[row]['history']))
 vmax=max(float(1000*np.linalg.norm(p-c['positions'][0],axis=1).max()) for c in cases for p in c['positions'][1:]);fig,axes,title,subtitle,norm,cmap=figure(vmax,'Static held-load shapes • camera orbit only • deformation ×1 • synthetic RGB-D markers • first noise repeat')
 writer=FFMpegWriter(fps=12,codec='libx264',extra_args=['-threads','2','-crf','20','-pix_fmt','yuv420p'])
 with writer.saving(fig,str(OUT/'actual-vs-predicted.mp4'),dpi=100):
  for c in cases:
   title.set_text(f"Strong-load probing: {c['label']}-error object {c['object']}");names=[s['action_names'][a].replace('_',' ') for a in c['history']];subtitle.set_text('Actions: '+(' → '.join(names) if names else 'stop immediately')+' | max. two acquisitions')
   labels=['Reference FEM',f"Initial prediction\nRMSE {c['errors'][1]:.3f} mm",f"Final prediction\nRMSE {c['errors'][2]:.3f} mm"]
   for frame in range(72):
    draw(axes,c['positions'],labels,bounds(c['positions']),norm,cmap,-70+40*np.sin(2*np.pi*frame/71))
    if frame==0:fig.savefig(OUT/f"prediction-{c['label']}.png",dpi=100)
    writer.grab_frame()
 plt.close(fig);write(OUT/'video-selection.json',dict(selection='Median and worst object-average final test MSE, first noise repeat; not selected for success.',cases=[{k:v for k,v in c.items() if k!='positions'} for c in cases]))
def phystwin():
 cases=[]
 for obj in range(6):
  arrays=[];scores=[]
  for policy in ['fixed','ensemble']:
   folder=BASE/'phystwin'/f'test-{obj}-{policy}';arrays.append(np.load(folder/'held_oblique-dt1.npz'));scores.append(json.loads((folder/'summary-dt1.json').read_text())['results']['held_oblique']['hold_rmse_mm'])
  cases.append(dict(object=obj,truth=arrays[0]['reference'],fixed=arrays[0]['predicted'],adaptive=arrays[1]['predicted'],scores=scores,times=arrays[0]['times']))
 vmax=max(float(1000*np.linalg.norm(c[k]-c['truth'],axis=2).max()) for c in cases for k in ['fixed','adaptive']);fig,axes,title,subtitle,norm,cmap=figure(vmax,'Deformation ×1 • interpolated quasistatic reference ramp, not dynamic ground truth • error scores use settled holds')
 writer=FFMpegWriter(fps=5,codec='libx264',extra_args=['-threads','2','-crf','20','-pix_fmt','yuv420p'])
 with writer.saving(fig,str(OUT/'phystwin-fixed-vs-ensemble.mp4'),dpi=100):
  for c in cases:
   title.set_text(f"Matched PhysTwin action policies: object {c['object']}");limits=bounds([v for k in ['truth','fixed','adaptive'] for v in c[k]])
   for frame,t in enumerate(c['times']):
    subtitle.set_text(f"t={t:.1f} s • 27 markers, 91 training frames, 2700 updates in each fit • held load never trains the model")
    labels=['Quasistatic reference',f"Fixed pokes\nHold RMSE {c['scores'][0]:.3f} mm",f"PhysTwin ensemble pokes\nHold RMSE {c['scores'][1]:.3f} mm"]
    draw(axes,[c[k][frame] for k in ['truth','fixed','adaptive']],labels,limits,norm,cmap,-70+25*np.sin(2*np.pi*frame/(len(c['times'])-1)))
    if frame==15:fig.savefig(OUT/f"phystwin-object-{c['object']}.png",dpi=100)
    writer.grab_frame()
 plt.close(fig);write(OUT/'phystwin-video-selection.json',dict(objects=list(range(6)),selection='All six bridge objects in protocol order; no success selection.',scope='Spring rollouts versus interpolated static reference ramps. Only settled hold scores support quantitative comparison; transient reference is not validated dynamic truth.'))

if __name__=='__main__':
 if not all((OUT/name).exists() for name in ['actual-vs-predicted.mp4','video-selection.json','prediction-median.png','prediction-worst.png']):bayesian()
 if not all((OUT/name).exists() for name in ['phystwin-fixed-vs-ensemble.mp4','phystwin-video-selection.json',*[f'phystwin-object-{i}.png' for i in range(6)]]):phystwin()
