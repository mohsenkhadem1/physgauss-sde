"""Matched RGB and error crops at predeclared middle test times for all scenes."""
from pathlib import Path
import sys,json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import font_manager
ROOT=Path(__file__).resolve().parents[2];REPORT=ROOT/'reports/pidg-dense-mechanics-v01';OUT=REPORT/'figures';OUT.mkdir(exist_ok=True)
for p in (Path.home()/'.local/share/fonts/arial').glob('*.TTF'):font_manager.fontManager.addfont(str(p))
plt.rcParams.update({'font.family':'Arial','font.size':9,'pdf.fonttype':42,'text.color':'black','axes.labelcolor':'black','xtick.color':'black','ytick.color':'black'})
fig,axs=plt.subplots(3,5,figsize=(7.15,4.7),gridspec_kw={'height_ratios':[.8,.65,1.1]});provenance=[]
for i,scene in enumerate(['balls_reaction','rubber_duck','motion_kuro']):
 r=json.loads((REPORT/scene/'results.json').read_text());selected=r['selection']['selected'];native=r['metrics']['native-refined']['640'];ours=r['metrics'][selected]['640'];frame=native['rows'][len(native['rows'])//2]['frame'];folder=ROOT/'runs/pidg-dense-mechanics-v01'/scene
 a=np.load(folder/f'native-refined/test-render-640/{frame}.npz');b=np.load(folder/f'{selected}/test-render-640/{frame}.npz');assert np.array_equal(a['truth'],b['truth']);y,x=np.nonzero(a['mask']);pad=10;box=[max(0,y.min()-pad),min(len(a['mask']),y.max()+pad+1),max(0,x.min()-pad),min(a['mask'].shape[1],x.max()+pad+1)];y0,y1,x0,x1=box;ims=[a['truth'],a['prediction'],b['prediction']];errs=[np.mean(np.abs(im-a['truth']),axis=-1) for im in ims[1:]]
 for j,im in enumerate(ims+errs):
  ax=axs[i,j];crop=im[y0:y1,x0:x1];ax.imshow(crop,**({'cmap':'magma','vmin':0,'vmax':.15} if j>=3 else {}));ax.set_xticks([]);ax.set_yticks([])
  if i==0:ax.set_title(['Reference','PIDG','Ours (hybrid)','PIDG error','Our error'][j],fontsize=9,weight='bold')
  if j==0:ax.set_ylabel(scene.replace('_',' ').title(),fontsize=9,weight='bold')
  if j in [1,2]:
   row=next(x for x in (native if j==1 else ours)['rows'] if x['frame']==frame);ax.set_xlabel(f"{row['psnr_db']:.2f} dB",fontsize=9)
 provenance.append({'scene':scene,'frame':frame,'crop_yxyx':list(map(int,box)),'selection':'Middle declared test frame; identical reference-foreground crop plus 10 pixels; all scenes shown','error_scale':[0,.15]})
fig.subplots_adjust(left=.045,right=.995,top=.94,bottom=.14,wspace=.035,hspace=.34);cax=fig.add_axes([.66,.035,.28,.018]);bar=fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0,.15),cmap='magma'),cax=cax,orientation='horizontal',ticks=[0,.075,.15]);bar.ax.tick_params(labelsize=8);bar.set_label('Mean absolute RGB error',fontsize=8);fig.savefig(OUT/'three-scene-comparison.pdf',bbox_inches='tight');fig.savefig(OUT/'three-scene-comparison.png',bbox_inches='tight',dpi=200);plt.close(fig);(OUT/'provenance.json').write_text(json.dumps(provenance,indent=2))
