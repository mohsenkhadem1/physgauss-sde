"""Median and worst cost-aware static held-load predictions, at true scale."""
from pathlib import Path
import json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.phystwin.render_v06 import figure,draw,bounds,REST,plt,FFMpegWriter
from scripts.phystwin.v07_study import BASE,OUT,protocol,write

def main():
 s=protocol();z=np.load(BASE/'test-cost_aware-65.npz');mse=((z['mean'][:,-1]-z['truth'])**2).mean(1).reshape(24,3).mean(1);order=np.argsort(mse);plans=json.loads((BASE/'plans/test-cost_aware.json').read_text());cases=[]
 for label,obj in [('median',int(order[11])),('worst',int(order[-1]))]:
  row=3*obj;positions=[REST+z['truth'][row].reshape(81,3),REST+z['mean'][row,0].reshape(81,3),REST+z['mean'][row,-1].reshape(81,3)];errors=[float(1000*np.sqrt(np.mean((p-positions[0])**2))) for p in positions];cases.append(dict(label=label,object=obj,positions=positions,errors=errors,history=plans[row]['history']))
 vmax=max(float(1000*np.linalg.norm(p-c['positions'][0],axis=1).max()) for c in cases for p in c['positions'][1:]);fig,axes,title,subtitle,norm,cmap=figure(vmax,'Static held-load shapes • camera orbit only • deformation ×1 • synthetic RGB-D markers • first noise repeat')
 writer=FFMpegWriter(fps=12,codec='libx264',extra_args=['-threads','2','-crf','20','-pix_fmt','yuv420p'])
 with writer.saving(fig,str(OUT/'actual-vs-predicted.mp4'),dpi=100):
  for c in cases:
   title.set_text(f"Cost-aware probing: {c['label']}-error object {c['object']}");names=[s['action_names'][a].replace('_',' ') for a in c['history']];subtitle.set_text('Actions: '+(' → '.join(names) if names else 'stop immediately')+' | max. two acquisitions')
   labels=['Reference FEM',f"Initial prediction\nRMSE {c['errors'][1]:.3f} mm",f"Final prediction\nRMSE {c['errors'][2]:.3f} mm"]
   for frame in range(72):
    draw(axes,c['positions'],labels,bounds(c['positions']),norm,cmap,-70+40*np.sin(2*np.pi*frame/71))
    if frame==0:fig.savefig(OUT/f"prediction-{c['label']}.png",dpi=100)
    writer.grab_frame()
 plt.close(fig);write(OUT/'video-selection.json',dict(selection='Median and worst object-average final test MSE, first noise repeat; not selected for success.',cases=[{k:v for k,v in c.items() if k!='positions'} for c in cases]))
if __name__=='__main__':main()
