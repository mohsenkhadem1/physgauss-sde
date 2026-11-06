"""Measured PIDG appearance-refinement figures; no prediction editing or warping."""
from pathlib import Path
import json,hashlib,sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import figure_style
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
ROOT=Path(__file__).resolve().parents[3];PAPER=Path(__file__).resolve().parents[1];OUT=PAPER/'figures'
for p in (Path.home()/'.local/share/fonts/arial').glob('*'):
 if p.suffix.lower()=='.ttf':font_manager.fontManager.addfont(str(p))
plt.rcParams.update({'font.family':'Arial','font.size':9,'text.color':'black','axes.labelcolor':'black','xtick.color':'black','ytick.color':'black','pdf.fonttype':42})
selection=json.loads((ROOT/'reports/pidg-appearance-transfer-v01/selection.json').read_text());run=ROOT/selection['run'];r=json.loads((run/'evaluation/results.json').read_text());old=json.loads((ROOT/'runs/pidg-mechanics-v04/evaluation/results.json').read_text());native=json.loads((ROOT/'runs/pidg-comparison-v02/refined-pidg/test-render-640/metrics.json').read_text());frame=r['novel_views'][10]['frame'];name=f'{frame:04d}';src=ROOT/f'runs/pidg-comparison-v02/refined-pidg/test-render-640/{name}.npz';a=np.load(src);images=[a['truth'],a['prediction'],np.asarray(Image.open(ROOT/f'runs/pidg-mechanics-v04/evaluation/{name}.png'))/255,np.asarray(Image.open(run/f'evaluation/{name}.png'))/255]
y,x=np.nonzero(a['mask']);pad=14;box=(max(0,y.min()-pad),min(len(a['mask']),y.max()+pad+1),max(0,x.min()-pad),min(a['mask'].shape[1],x.max()+pad+1))
fig=plt.figure(figsize=(7.15,3.55));titles=['Reference','PIDG','Ours: coarse appearance','Ours: transferred appearance']
for j,(title,im) in enumerate(zip(titles,images)):
 ax=fig.add_axes([j*.25,.53,.245,.39]);ax.imshow(im[box[0]:box[1],box[2]:box[3]]);ax.axis('off');ax.set_title(title,fontsize=8.5,fontweight='bold',pad=3)
 if j:
  source=[native['rows'],old['rows'],r['rows']][j-1];row=next(row for row in source if row['frame']==name and row.get('width',640)==640);ax.text(.5,-.07,f"{row['psnr_db']:.2f} dB",ha='center',transform=ax.transAxes,fontsize=9)
fig.text(.005,.975,'Held-out image: same camera, same reference-derived crop',fontsize=10,fontweight='bold')
for j,angle in enumerate([-60,0,60]):
 for i,(label,folder) in enumerate([('PIDG',ROOT/'runs/pidg-view-audit-v03/refined-pidg'),('Ours: transferred',run/'evaluation')]):
  ax=fig.add_axes([.105+j*.29,.025+(1-i)*.205,.285,.195]);ax.imshow(Image.open(folder/f'{name}-yaw{angle:+04d}.png'));ax.axis('off')
  if i==0:ax.set_title(f'{angle:+d} degrees',fontsize=9,pad=0)
for i,label in enumerate(['PIDG','Ours']):fig.text(.005,.305-i*.205,label,fontsize=9,fontweight='bold')
fig.text(.005,.465,'Camera sweep: no reference RGB at nonzero angles',fontsize=10,fontweight='bold')
fig.savefig(OUT/'pidg-rendering-comparison.pdf',bbox_inches='tight',pad_inches=.03,dpi=240);fig.savefig(OUT/'pidg-rendering-comparison.png',bbox_inches='tight',pad_inches=.03,dpi=240);plt.close(fig)
# Quantitative comparison keeps full-frame and foreground errors separate.
fig,axs=plt.subplots(1,2,figsize=(7.15,1.75));labels=['PIDG','Coarse appearance','Transferred appearance'];values=[[native['metrics']['psnr_db'],old['means']['640']['psnr_db'],r['means']['640']['psnr_db']],[native['metrics']['foreground_psnr_db'],old['means']['640']['foreground_psnr_db'],r['means']['640']['foreground_psnr_db']]]
for ax,v,title in zip(axs,values,['Full-image PSNR (dB)','Foreground PSNR (dB)']):
 bars=ax.barh(labels[::-1],v[::-1],color=['#008c88','#aab8c2','#df9b50']);ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=9,color='black');ax.set_xlim(0,max(v)*1.15);ax.set_title(title,fontsize=10,fontweight='bold');ax.tick_params(labelsize=8);ax.spines[['top','right']].set_visible(False)
fig.tight_layout(pad=.4,w_pad=2);fig.savefig(OUT/'pidg-rendering-metrics.pdf',bbox_inches='tight',pad_inches=.03);plt.close(fig)
# An explicit solver/render feedback diagram, with a real fitting image thumbnail.
fig=plt.figure(figsize=(7.15,1.65));ax=fig.add_axes([0,0,1,1]);ax.axis('off');labels=['Observed RGB\n+ camera poses','Fit physical roles\n+ solve equilibrium','Bind fitted Gaussians\n+ transport covariance','Render observed /\nheld-out cameras']
for i,label in enumerate(labels):
 x=.01+i*.252;ax.add_patch(FancyBboxPatch((x,.43),.225,.45,boxstyle='round,pad=.008',fc='#edf3f5',ec='#78909c',lw=.8));ax.text(x+.112,.655,label,ha='center',va='center',fontsize=9,color='black')
 if i<3:ax.add_patch(FancyArrowPatch((x+.23,.65),(x+.249,.65),arrowstyle='-|>',mutation_scale=10,color='black'))
ax.add_patch(FancyArrowPatch((.88,.41),(.38,.41),connectionstyle='arc3,rad=-.18',arrowstyle='-|>',mutation_scale=10,color='black'));ax.text(.63,.07,'Training image loss -> implicit mechanics adjoint',ha='center',fontsize=9,color='black');fig.savefig(OUT/'pidg-rendering-loop.pdf',bbox_inches='tight',pad_inches=.03);plt.close(fig)
(PAPER/'evidence/pidg-figure-provenance.json').write_text(json.dumps({'selection':selection,'frame':frame,'camera_yaw_degrees':[-60,0,60],'selection_rule':'Middle declared test time, fixed angles; not selected for gain','held_crop_yxyx':[int(x) for x in box],'crop_rule':'Reference foreground extent +14 pixels; identical for all methods','report_sha256':hashlib.sha256((run/'evaluation/results.json').read_bytes()).hexdigest(),'fonts':'Arial; all text black','limitations':'New camera images have no pixel reference; no numerical accuracy claim for camera sweep.'},indent=2)+'\n')
