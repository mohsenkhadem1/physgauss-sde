"""Generate paper figures from saved benchmark results; never fit or score-select models."""
from pathlib import Path
import sys,json,hashlib,pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
ROOT=Path(__file__).resolve().parents[2]; HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
OUT=HERE/'figures';OUT.mkdir(exist_ok=True)
TEAL='#008c88'; ORANGE='#dd8546'; BLUE='#5478a9'; INK='#25364b'; GRAY='#8795a5'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelcolor':INK,'text.color':INK,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.facecolor':'white'})
sources={}
def read(path):
 p=ROOT/path;sources[path]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text())
def save(fig,name):
 fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight',pad_inches=.03);fig.savefig(OUT/(name+'.png'),bbox_inches='tight',pad_inches=.03,dpi=180);plt.close(fig)
soft=read('reports/benchmark-improvement-v14/softvt-summary-b.json')['groups']['prospective_softvt_phase_b']
real=read('reports/real-transfer-v22/summary.json')
transfer=read('reports/real-transfer-v22/transfer-scores.json')
# Paired episode view, all cases, common zero-based axes.
fig,axs=plt.subplots(1,2,figsize=(7,2.5),gridspec_kw={'width_ratios':[1,1.35]})
r=soft['contact_scene_state'];x=np.array([a['baseline_mm']for a in r['per_case']]);y=np.array([a['proposal_mm']for a in r['per_case']]);lim=max(x.max(),y.max())*1.08
axs[0].plot([0,lim],[0,lim],c=GRAY,lw=1,ls='--');axs[0].scatter(x,y,c=np.where(y<x,TEAL,ORANGE),s=22,zorder=3);axs[0].set(xlim=(0,lim),ylim=(0,lim),xlabel='Without slip correction (mm)',ylabel='With slip correction (mm)',title='40 fresh episodes');axs[0].set_aspect('equal');axs[0].text(.04,.94,'36 improve / 4 worsen',transform=axs[0].transAxes,va='top',fontsize=8)
names=['Prefix rigid','Rigid + contact','Mechanics + contact','Ours + slip'];vals=[soft[k]['baseline_mean_mm']for k in ['prefix_calibrated_rigid','rigid_scene_state','contact_scene_state']]+[r['proposal_mean_mm']]
b=axs[1].barh(names[::-1],vals[::-1],color=[TEAL,GRAY,BLUE,'#b7c1ce']);axs[1].bar_label(b,fmt='%.2f',padding=3,fontsize=8);axs[1].set(xlim=(0,24),xlabel='Coordinate RMSE (mm), lower is better',title='Same 1,998 forecast frames');axs[1].grid(axis='x',alpha=.15);axs[1].set_axisbelow(True);fig.tight_layout(w_pad=1.3);save(fig,'soft_results')
# Both passing transfers plus complete eight-sequence comparison.
fig,axs=plt.subplots(1,2,figsize=(7,2.5),gridspec_kw={'width_ratios':[1,1.25]})
methods=['neural','analytic','position_ridge','official_phystwin'];vals=[real['geometry'][m]['tracking_mm']for m in methods]
b=axs[0].barh(['Ours','Analytic','Position ridge','PhysTwin'][::-1],vals[::-1],color=[ORANGE,GRAY,BLUE,TEAL]);axs[0].bar_label(b,fmt='%.1f',padding=3);axs[0].set(xlim=(0,49),xlabel='Tracking error (mm)',title='Within-sequence prediction: 8 cases')
passed=[r for r in transfer['rows']if r['registration']['initial_geometry_gate']];xx=np.arange(2);w=.24
for i,(m,name,col)in enumerate([('neural','Ours',TEAL),('analytic','Analytic',BLUE),('native_frozen','Frozen PhysTwin',ORANGE)]):
 vals=[r['results'][m]['whole_interaction']['tracking_mm']for r in passed];b=axs[1].bar(xx+(i-1)*w,vals,w,label=name,color=col);axs[1].bar_label(b,fmt='%.1f',fontsize=7,padding=2)
axs[1].set(xticks=xx,xticklabels=['Lift → push\nSloth','One → two hands\nZebra'],ylabel='Tracking error (mm)',ylim=(0,75),title='Frozen transfer: 2 qualifying pairs');axs[1].legend(ncol=3,fontsize=6.6,loc='upper center',frameon=False)
fig.tight_layout(w_pad=1.2);save(fig,'real_results')
# Diagram: explicit common abstraction, adapters visibly named.
fig=plt.figure(figsize=(7.1,3.05));ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
def box(x,y,w,h,title,body,color):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.009,rounding_size=.018',fc=color,ec='none'))
 ax.text(x+.014,y+h-.06,title,fontweight='bold',fontsize=9,va='top');ax.text(x+.014,y+h-.155,body,fontsize=7.6,va='top',linespacing=1.5)
def arrow(a,b,color=INK):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=11,lw=1.2,color=color,connectionstyle='arc3'))
box(.015,.57,.23,.36,'1  Observe','Initial geometry + sparse tracks\nKnown tool poses and calibration\nFit only the observed prefix','#eaf0f7')
box(.305,.57,.36,.36,'2  Separate the physical roles','Material coordinates: bulk response\nTool-relative coordinates: attachments\nWorld coordinates: gravity + support','#e5f3f0')
box(.725,.57,.26,.36,'3  Query new positions','Freeze scene material\nSupply future tool commands\nPredict positions and deformation','#e5f3f0')
arrow((.25,.75),(.30,.75));arrow((.67,.75),(.72,.75))
box(.015,.05,.43,.36,'SoftVTBench adapter','Analytic FEM + fitted position correction\nController-relative slip → release\nReduced rigid contact rollout','#edf1f6')
box(.525,.05,.46,.36,'PhysTwin adapter','Frozen neural bulk energy + fitted scene descriptors\nEquilibrium under changed tool positions\nShared initial Gaussian appearance for rendering','#edf1f6')
arrow((.40,.56),(.23,.42));arrow((.58,.56),(.75,.42));ax.text(.5,.475,'ONE REPRESENTATION PRINCIPLE · TWO DECLARED IMPLEMENTATIONS',ha='center',fontsize=6.5,color=GRAY)
save(fig,'method')
# Metric decomposition, no conflating render quality with mechanics.
fig,axs=plt.subplots(1,3,figsize=(7,2.1))
for ax,(key,title) in zip(axs,[('tracking_mm','World positions'),('displacement_mm','Motion from initial state'),('pair_distance_change_mm','Internal distance changes')]):
 s=transfer['summary']['initial_geometry_gate_passed'];vals=[s[m][key]for m in ['neural','analytic','native_frozen']];b=ax.bar(['Ours','Analytic','Frozen\nPhysTwin'],vals,color=[TEAL,BLUE,ORANGE]);ax.bar_label(b,fmt='%.1f',padding=3,fontsize=8);ax.set(title=title,ylabel='Error (mm)',ylim=(0,max(vals)*1.27));ax.tick_params(axis='x',labelsize=7);ax.grid(axis='y',alpha=.14);ax.set_axisbelow(True)
fig.tight_layout(w_pad=1.1);save(fig,'transfer_metrics')
# Teaser from actual saved SoftVT clouds, with exact common world-space limits.
from scripts.contact_benchmark_v12.data import load
from scripts.external_benchmark.evaluate import expand_prediction
case='spatial-soft-task2-demo41';data=load(dict(subset='spatial-soft',task=2,episode=41));start=data['train_end'];t=start+int(.75*(len(data['times'])-start-1))
p=ROOT/f'runs/benchmark-improvement-v14/robust/evaluation/prospective_softvt_phase_b/{case}/predictions.npz';sources[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
with np.load(p)as z:seq=[data['evaluation']['points'],expand_prediction(data,z['contact_scene_state']),expand_prediction(data,z['benchmark_v14'])]
fig=plt.figure(figsize=(7,2.18));allp=np.concatenate([a[t]for a in seq]);center=(allp.min(0)+allp.max(0))/2;radius=np.ptp(allp,axis=0).max()*.55
for i,(a,name,col)in enumerate(zip(seq,['Simulator reference','Without slip correction','Ours: slip-aware prediction'],[GRAY,ORANGE,TEAL])):
 ax=fig.add_subplot(1,3,i+1,projection='3d');pts=a[t];err=np.linalg.norm(pts-seq[0][t],axis=1)*1000
 ax.scatter(*seq[0][t].T,c='#aab4bf',s=2,alpha=.22);ax.scatter(*pts.T,c=col,s=7,alpha=.85,edgecolors='none')
 for k in range(3):getattr(ax,'set_'+['x','y','z'][k]+'lim')(center[k]-radius,center[k]+radius)
 ax.view_init(24,-60);ax.set_box_aspect((1,1,1));ax.set_axis_off();ax.set_title(name,fontsize=9,pad=-4,color=col,fontweight='bold')
 if i:ax.text2D(.5,.02,f'Frame coordinate RMSE: {np.sqrt(np.mean((pts-seq[0][t])**2))*1000:.1f} mm',ha='center',transform=ax.transAxes,fontsize=7.5)
fig.subplots_adjust(left=0,right=1,top=.88,bottom=.04,wspace=-.05);save(fig,'teaser')
# Real image comparison: deterministic first sequence, final frame, same crop.
from PIL import Image
case='single_lift_sloth';frame=84;camera=2
raw=ROOT/f'runs/editable-scene-v23/phystwin-full-rgb/data/different_types/{case}/color/{camera}/{frame}.png'
images=[np.asarray(Image.open(raw).convert('RGB'))]
folder=ROOT/f'runs/real-transfer-v20/render-audit/v22-corrected-mask240/{case}'
for m in ['neural','official_phystwin']:images.append(np.asarray(Image.open(folder/f'{m}-camera{camera}-frame{frame}.png').convert('RGB')))
# Saved comparison renders are output images; use full images to avoid hidden shifts.
fig,axs=plt.subplots(1,3,figsize=(7,2.1))
for ax,img,title in zip(axs,images,['Recorded RGB','Ours: shared appearance','PhysTwin: shared appearance']):ax.imshow(img);ax.set_title(title,fontsize=8);ax.axis('off')
fig.tight_layout(pad=.1,w_pad=.3);save(fig,'render_comparison')
selection=dict(soft_teaser=dict(case='spatial-soft-task2-demo41',selection='Largest episode-level improvement among the 40 fresh episodes; illustrative best case, not representative sampling.',frame=t,frame_rule='75% of forecast suffix; common world axes; no geometric exaggeration'),real_panel=dict(case=case,frame=frame,camera=camera,selection='First declared standard sequence, final recorded frame; full images.'),scope='Scientific figures from saved results; schematic method diagram contains no claimed experimental outputs.')
(HERE/'evidence/figure-sources.json').write_text(json.dumps(dict(sources=sources,selection=selection),indent=2)+'\n')
(HERE/'evidence/numbers.json').write_text(json.dumps(dict(soft={k:{a:v for a,v in soft[k].items()if a in ['proposal_mean_mm','baseline_mean_mm','wins','losses','paired_episode_bootstrap_95_interval_mm']}for k in ['contact_scene_state','rigid_scene_state','prefix_calibrated_rigid','shape_vs_original']},real_geometry=real['geometry'],real_render=real['render'],transfer=transfer['summary']),indent=2)+'\n')
print('Generated six figures and evidence manifest.')
