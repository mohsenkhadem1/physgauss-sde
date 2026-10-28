"""All six matched-budget PhysTwin cases, including failures, at true scale."""
from pathlib import Path
import json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.phystwin.render_v06 import figure,draw,bounds,plt,FFMpegWriter
from scripts.phystwin.v07_study import BASE,OUT,write

def main():
 cases=[]
 for obj in range(6):
  arrays=[];scores=[]
  for policy in ['fixed','adaptive_pokes']:
   folder=BASE/'phystwin'/f'object-{obj}-{policy}';arrays.append(np.load(folder/'held_oblique-dt1.npz'));scores.append(json.loads((folder/'summary-dt1.json').read_text())['results']['held_oblique']['hold_rmse_mm'])
  cases.append(dict(object=obj,truth=arrays[0]['reference'],fixed=arrays[0]['predicted'],adaptive=arrays[1]['predicted'],scores=scores,times=arrays[0]['times']))
 vmax=max(float(1000*np.linalg.norm(c[k]-c['truth'],axis=2).max()) for c in cases for k in ['fixed','adaptive']);fig,axes,title,subtitle,norm,cmap=figure(vmax,'Deformation ×1 • interpolated quasistatic reference ramp, not dynamic ground truth • error scores use settled holds')
 writer=FFMpegWriter(fps=5,codec='libx264',extra_args=['-threads','2','-crf','20','-pix_fmt','yuv420p'])
 with writer.saving(fig,str(OUT/'phystwin-fixed-vs-adaptive.mp4'),dpi=100):
  for c in cases:
   title.set_text(f"Matched PhysTwin action policies: object {c['object']}");limits=bounds([v for k in ['truth','fixed','adaptive'] for v in c[k]])
   for frame,t in enumerate(c['times']):
    subtitle.set_text(f"t={t:.1f} s • 27 markers, 91 training frames, 2700 updates in each fit • held load never trains the model")
    labels=['Quasistatic reference',f"Fixed pokes\nHold RMSE {c['scores'][0]:.3f} mm",f"Adaptive pokes\nHold RMSE {c['scores'][1]:.3f} mm"]
    draw(axes,[c[k][frame] for k in ['truth','fixed','adaptive']],labels,limits,norm,cmap,-70+25*np.sin(2*np.pi*frame/(len(c['times'])-1)))
    if frame==15:fig.savefig(OUT/f"phystwin-object-{c['object']}.png",dpi=100)
    writer.grab_frame()
 plt.close(fig);write(OUT/'phystwin-video-selection.json',dict(objects=list(range(6)),selection='All six bridge objects in protocol order; no success selection.',scope='Spring rollouts versus interpolated static reference ramps. Only settled hold scores support quantitative comparison; transient reference is not validated dynamic truth.'))
if __name__=='__main__':main()
