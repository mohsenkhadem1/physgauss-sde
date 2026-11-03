"""Reproducible vector diagrams and figures from frozen, audited results."""
import json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent/'figures';OUT.mkdir(exist_ok=True)
R=ROOT/'reports/compositional-dynamics-v17';V=ROOT/'reports/compositional-neural-mechanics-v16';E={}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
BLUE='#1675a9';ORANGE='#d46b19';GREEN='#237e66';INK='#163447';GRAY='#677482'
def read(p):
 E[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text())
def save(fig,name):
 fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight',pad_inches=.03);fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight',pad_inches=.03);plt.close(fig)
def box(ax,x,y,w,h,title,body,color=BLUE):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.025,rounding_size=.07',lw=1.3,edgecolor=color,facecolor=color+'10'))
 ax.text(x+w/2,y+h-.20,title,ha='center',va='center',fontsize=11,weight='bold',color=color)
 ax.text(x+w/2,y+.30,body,ha='center',va='center',fontsize=10,color=INK,linespacing=1.4)
def arrow(ax,a,b,color=GRAY,dashed=False):
 ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,lw=1.4,color=color,linestyle='--'if dashed else'-',connectionstyle='arc3,rad=0'))
fig,ax=plt.subplots(figsize=(10.8,3.6));ax.set(xlim=(0,10.8),ylim=(0,3.6));ax.axis('off')
box(ax,.10,2.3,1.85,1.05,'Observe','3D positions\nGeometry / mass')
box(ax,2.30,2.3,2.0,1.05,'Fit descriptors','Material + interaction\nFrozen shared weights')
box(ax,4.85,2.3,2.0,1.05,'Shared local laws','Bulk energy $W_\\theta$\nAttachment energy $\\psi_\\phi$',GREEN)
box(ax,7.45,2.3,3.05,1.05,'Separate appearance','Initial Gaussian reconstruction\nNo future-image fitting',ORANGE)
arrow(ax,(1.98,2.82),(2.27,2.82));arrow(ax,(4.82,2.82),(4.33,2.82),GREEN)
box(ax,.10,.55,1.85,1.05,'Query','Tool pose / gravity\nInitial state / contact')
box(ax,2.30,.55,2.0,1.05,'Compose laws','Bulk + attachments\nContact + body loads',GREEN)
box(ax,4.85,.55,2.0,1.05,'Q2 mechanics','Equilibrium\nor dynamics',GREEN)
box(ax,7.45,.55,1.37,1.05,'Physical','Shape\nReactions',BLUE)
box(ax,9.13,.55,1.37,1.05,'Visual','Gaussian\nimages',ORANGE)
for a,b in [((1.98,1.08),(2.27,1.08)),((4.33,1.08),(4.82,1.08)),((6.88,1.08),(7.42,1.08)),((8.85,1.08),(9.10,1.08))]:arrow(ax,a,b)
arrow(ax,(3.30,2.27),(3.30,1.63));arrow(ax,(5.85,2.27),(4.15,1.63),GREEN);arrow(ax,(9.82,2.27),(9.82,1.63),ORANGE)
ax.text(.10,1.95,'observed responses only',fontsize=9,color=GRAY)
ax.text(5.4,.13,'Evaluate physical error and held-view image quality separately; retain failed forecasts.',ha='center',fontsize=10,color=INK)
save(fig,'method_overview')
fig,ax=plt.subplots(figsize=(7.0,2.55));ax.set(xlim=(0,7),ylim=(0,2.55));ax.axis('off')
box(ax,.08,1.35,1.65,.95,'Material state','$F,\\; z=(s,h,f)$',BLUE)
box(ax,2.05,1.35,2.25,.95,'Objective energy','$W=e^s E_0\\sum_j c_j(h,f)b_j(F)$',GREEN)
box(ax,4.62,1.35,2.25,.95,'Derived response','$P=\\partial W/\\partial F$\n$M\\ddot u=-\\nabla_u\\Pi$',GREEN)
arrow(ax,(1.76,1.82),(2.02,1.82));arrow(ax,(4.33,1.82),(4.59,1.82))
ax.text(.13,.78,'Invariant basis',fontsize=10,weight='bold',color=INK);ax.text(.13,.43,'Isochoric shape\n+ volume',fontsize=9,color=GRAY)
ax.text(2.10,.78,'Tiny shared decoder',fontsize=10,weight='bold',color=INK);ax.text(2.10,.43,'2 → 16 SiLU → 7\nPositive coefficients',fontsize=9,color=GRAY)
ax.text(4.66,.78,'Scene adaptation',fontsize=10,weight='bold',color=INK);ax.text(4.66,.43,'Fit descriptors\nFreeze network weights',fontsize=9,color=GRAY)
save(fig,'energy_block')
g=read(R/'fresh-cohort-geometry-recovery-results.json');im=read(R/'fresh-cohort-appearance-paired-summary.json');full=read(R/'fresh-cohort-appearance-results.json')
fig,axes=plt.subplots(2,2,figsize=(8,5.1),layout='constrained')
for ax,key,label in [(axes[0,0],'world_point_rms_m','World vector RMS (mm) ↓'),(axes[0,1],'shape_point_rms_m','Shape vector RMS (mm) ↓')]:
 for meth,off,col,lab in [('ours',-.18,BLUE,'Ours'),('native_phystwin',.18,ORANGE,'Native PhysTwin')]:
  for i,r in enumerate(g['rows']):
   v=r['methods'][meth].get('geometry',{}).get(key)
   if v is None:ax.text(i+off,.03,'×',transform=ax.get_xaxis_transform(),ha='center',color=ORANGE,fontsize=14)
   else:ax.bar(i+off,v*1000,width=.34,color=col,label=lab if i==0 else None)
 ax.set_xticks(range(8),[r['world_id'].replace('world-','W')+'\n'+r['role'].replace('forecast-','').replace('mm','')for r in g['rows']]);ax.set_ylabel(label);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
axes[0,0].legend(frameon=False,fontsize=8);axes[0,0].set_xlabel('World / drop height (mm)');axes[0,1].set_xlabel('World / drop height (mm)')
for ax,key,label in [(axes[1,0],'psnr_foreground_union_db','Held-view foreground PSNR (dB) ↑'),(axes[1,1],'ssim_rgb_foreground_union','Held-view foreground SSIM ↑')]:
 for method,off,col in [('shared_q2',-.18,BLUE),('native',.18,ORANGE)]:
  for i in range(4):
   row=next(x for x in full['per_world_summary']if x['world_id']==f'world-{i}'and x['method']==method and x['camera_role']=='held');v=row['metrics'][key]['full_coverage_mean']
   if v is None:ax.text(i+off,.03,'×',transform=ax.get_xaxis_transform(),ha='center',color=ORANGE,fontsize=14)
   else:ax.bar(i+off,v,width=.34,color=col)
 ax.set_xticks(range(4),['W0','W1','W2','W3']);ax.set_ylabel(label);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
axes[1,1].set_ylim(0,1);save(fig,'dynamic_results')
# Exact scored RGB arrays, common crop across all times/methods in the first declared world.
base=ROOT/'runs/compositional-dynamics-v17/fresh-cohort-appearance-inference';world='world-0';role='forecast-80mm';cam='held-0';frames=[0,13,18,30];rows={(r['method'],r['frame']):r for r in full['rows']if r['world_id']==world and r['query']==role and r['camera']==cam};images={};masks=[]
for i in range(31):
 for method in ['reference','shared_q2','native']:
  p=base/'references'/world/role/f'{i:03d}-{cam}.npz'if method=='reference'else ROOT/rows[method,i]['prediction']['path']
  with np.load(p)as f:
   masks.append(f['body_visible'].astype(bool));
   if i in frames:images[method,i]=f['rgb'].copy();E[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
y,x=np.where(np.logical_or.reduce(masks));y0,y1=max(0,int(y.min())-14),min(480,int(y.max())+15);x0,x1=max(0,int(x.min())-14),min(848,int(x.max())+15)
fig,axes=plt.subplots(3,4,figsize=(7.2,4.3),gridspec_kw={'wspace':.02,'hspace':.08})
for a,method in enumerate(['reference','shared_q2','native']):
 for b,i in enumerate(frames):
  ax=axes[a,b];ax.imshow(np.clip(images[method,i][y0:y1,x0:x1],0,1));ax.set_xticks([]);ax.set_yticks([]);[s.set_visible(False)for s in ax.spines.values()]
  if a==0:ax.set_title(f'$t={i/100:.2f}$ s',fontsize=9)
  if b==0:ax.set_ylabel({'reference':'Reference','shared_q2':'Ours','native':'Native PhysTwin'}[method],fontsize=9)
save(fig,'qualitative')
forward=read(V/'oracle-recovered-plus-stable-io-comparison.json');adapt=read(V/'unknown-action-reaction-paper-summary.json')
fig,axes=plt.subplots(1,3,figsize=(8,2.6),layout='constrained')
methods=[('matched_analytic','Analytic'),('neo_hookean','Neo-Hookean'),('shared_laws','Shared laws'),('custom_direct_stable','Direct 6k'),('custom_direct_long_stable','Direct 30k')]
for j,cohort in enumerate(['reserved_composition','reserved_geometry']):
 vals=[1000*next(x for x in forward['rows']if x['cohort']==cohort and x['method']==m)['object_mean_scores']['displacement_coordinate_rmse_m']for m,_ in methods]
 axes[j].barh(range(5),vals,color=[GRAY,GRAY,BLUE,ORANGE,ORANGE]);axes[j].set_yticks(range(5),[l for _,l in methods]);axes[j].invert_yaxis();axes[j].set_xscale('log');axes[j].set_xlabel('Coordinate RMSE (mm), log scale');axes[j].set_title('New compositions'if j==0 else'New geometries',fontsize=10);axes[j].grid(axis='x',alpha=.2)
policies=[('initial','Initial'),('fixed_stretch','+ stretch'),('uniform_random','+ random'),('target_task_variance','+ adaptive')]
vals=[next(r for r in adapt['rows']if r['model']=='shared_laws_fitted'and r['policy']==p)['position_rmse_mm']for p,_ in policies]
axes[2].barh(range(4),vals,color=[GRAY,BLUE,ORANGE,GREEN]);axes[2].set_yticks(range(4),[l for _,l in policies]);axes[2].invert_yaxis();axes[2].set_xlabel('Coordinate RMSE (mm)');axes[2].set_title('Unknown-parameter refinement',fontsize=10);axes[2].grid(axis='x',alpha=.2)
save(fig,'transfer_refinement')
amb=read(V/'appearance-physical-ambiguity.json');fig,ax=plt.subplots(figsize=(7,1.65));ax.set(xlim=(0,7),ylim=(0,1.65));ax.axis('off')
box(ax,.08,.45,2.0,1.03,'Explanation A','$W,\\;\\psi$\n$u^*,\\; f$',BLUE);box(ax,2.52,.45,2.0,1.03,'Explanation B','$cW,\\;c\\psi$\n$u^*,\\;cf$',GREEN);box(ax,4.96,.45,1.95,1.03,'Any camera','Same geometry\nSame images',ORANGE)
arrow(ax,(2.11,.96),(2.49,.96));arrow(ax,(4.55,.96),(4.93,.96));ax.text(3.5,.14,'Zero body load; all stiffness-bearing terms co-scaled. Here c = 2: reactions double.',ha='center',fontsize=9,color=INK);save(fig,'observability')
(OUT.parent/'evidence/figure-sources.json').write_text(json.dumps({'inputs_sha256':E,'qualitative_crop_yxyx':[y0,y1,x0,x1],'qualitative_selection':'First declared world,80mmload,held-0,frames0/13/18/30; commoncrop fromall31frames/allmethods, unchangedRGB.','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
print('Generated six vector figure sets')
