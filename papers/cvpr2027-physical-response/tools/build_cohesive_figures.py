"""Evidence-preserving figures for the cohesive physical-response manuscript.

Run from repository root with .venv/bin/python. No fitting, simulation or target
selection is performed. Mesh examples are fixed world00/query0/final frame.
"""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'papers/cvpr2027-compositional-mechanics/tools'))
import design_editorial_figures as e
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyBboxPatch,FancyArrowPatch,Ellipse
from matplotlib.collections import PolyCollection
D=e.d;OUT=Path(__file__).resolve().parents[1]/'figures';OUT.mkdir(exist_ok=True)
INK='#183044';MUTED='#65798a';TEAL='#008c88';GOLD='#dc8032';BLUE='#397bc1';PURPLE='#8766ac'
plt.rcParams.update({'font.size':8,'font.family':'DejaVu Sans','pdf.fonttype':42,'axes.titlesize':9,'axes.labelsize':8})
SOURCES={};RECORDS={}
def read(p):
 p=ROOT/p;SOURCES[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text())
def save(f,name):
 for ext in ['pdf','png']:f.savefig(OUT/(name+'.'+ext),bbox_inches='tight',pad_inches=.025,dpi=240)
 plt.close(f)
def txt(f,x,y,t,size=8,c=INK,w='normal',**kw):f.text(x,y,t,fontsize=size,color=c,weight=w,**kw)
def arr(ax,a,b,c=INK):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=9,lw=1.1,color=c))
def card(ax,x,y,w,h,c='#f3f7f9',ec='none'):ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.004,rounding_size=.012',fc=c,ec=ec,lw=.7))
def teaser():
 from matplotlib import font_manager
 for path in (Path.home()/'.local/share/fonts/arial').glob('*'):
  if path.suffix.lower()=='.ttf':font_manager.fontManager.addfont(str(path))
 assert font_manager.findfont('Arial',fallback_to_default=False)
 with plt.rc_context({'font.family':'Arial','text.color':'black','axes.labelcolor':'black','xtick.color':'black','ytick.color':'black'}):
  f=plt.figure(figsize=(7.15,2.78));methods=['reference','ours','phystwin','spring-gaus'];rec=[]
  for j,m in enumerate(methods):
   txt(f,.235+j*.215,.958,{'reference':'Reference','ours':'Ours','phystwin':'PhysTwin*','spring-gaus':'Spring-Gaus*'}[m],10,'black','bold',ha='center')
  for row,family in enumerate(['curved_push','container_drop','deformable_pair']):
   data,t=D.selected(family);y=.69-row*.285;lim=e.extent([data[m][-1] for m in methods]);errors={}
   txt(f,.002,y+.11,['Curved\ncontact','Drop +\nobstacle','Two soft\nobjects'][row],9,'black','bold',linespacing=1.05)
   txt(f,.002,y+.045,f'{t[-1]:.3f} s',8.5,'black')
   for j,m in enumerate(methods):
    left=.13+j*.215;x=data[m][-1];error=np.linalg.norm(x-data['reference'][-1],axis=-1)*1000
    e.obj(f,[left,y,.21,.255],x,lim,error=None if m=='reference' else error,base=BLUE)
    if m!='reference':
     rms=float(np.sqrt(np.mean(error**2)));errors[m]=rms
     txt(f,left+.105,y-.012,f'{rms:.2f} mm',9.5,'black','bold' if m=='ours' else 'normal',ha='center')
   rec.append(dict(family=family,world=family+'-world00',query=0,frame='final',time_s=float(t[-1]),frame_vector_rms_mm=errors))
  cb=f.colorbar(plt.cm.ScalarMappable(norm=D.NORM,cmap=D.CMAP),cax=f.add_axes([.60,.045,.37,.020]),orientation='horizontal',extend='max')
  cb.ax.tick_params(labelsize=8,length=2,pad=1);cb.set_ticks([0,5,10,15,20])
  txt(f,.002,.040,'Point error (mm); lower is better. >20 saturates.',8.5,'black')
  save(f,'cohesive-teaser');RECORDS['teaser']=rec

def method():
 f=plt.figure(figsize=(7.15,3.6));c=f.add_axes([0,0,1,1]);c.axis('off');c.set(xlim=(0,1),ylim=(0,1));x=e.block();lim=e.extent([x])
 txt(f,.012,.955,'OBSERVE & FIT',9,BLUE,'bold');txt(f,.29,.955,'COMPOSE PHYSICAL ROLES',9,TEAL,'bold');txt(f,.815,.955,'FREEZE & QUERY',9,GOLD,'bold')
 for i in range(3):
  xx=x.copy();xx[:,2]+=.06*i*np.sin(np.pi*D.Q[:,0]);a=e.obj(f,[.012+i*.031,.69-i*.055,.14,.19],xx[None],lim,base=['#bed3e2','#75a8cd',BLUE][i]);p=e.project(xx[::80]);a.scatter(p[:,0],p[:,1],s=4,color=GOLD,zorder=15)
 txt(f,.015,.49,'Position-track prefix',7.8,BLUE,'bold');txt(f,.015,.405,'Geometry, mass, camera\nand tool poses supplied',7,MUTED,linespacing=1.3)
 card(c,.012,.285,.218,.08,'#edf3f8',BLUE);txt(f,.121,.32,'Fit scene descriptors',8,BLUE,'bold',ha='center');arr(c,(.115,.395),(.115,.37),BLUE)
 roles=[(.29,'Material coordinates',TEAL,r'$W(F,z_m)$','Bulk response'),(.46,'Tool-relative',GOLD,r'$\psi(\delta,z_b)$','Attachments'),(.63,'World coordinates',PURPLE,r'$\Pi_c + \Pi_g$','Contact + gravity')]
 for left,title,color,eq,desc in roles:
  card(c,left,.46,.155,.425,'#f3f7f8');txt(f,left+.0775,.855,title,6.8,color,'bold',ha='center')
  a=e.obj(f,[left+.01,.66,.135,.15],x[None],lim,base=color)
  if left>.60:
   a.plot([lim[0][0],lim[1][0]],[lim[0][1],lim[0][1]],c=INK,lw=1.5);arr(c,(left+.14,.77),(left+.14,.69),PURPLE)
  if .4<left<.6:
   c.plot([left+.025,left+.025],[.685,.755],color=INK,lw=3);c.plot([left+.025,left+.055],[.72,.72],color=GOLD,ls=':',lw=1)
  txt(f,left+.0775,.60,eq,9,color,ha='center');txt(f,left+.0775,.515,desc,7.2,INK,ha='center')
  arr(c,(left+.0775,.45),(left+.0775,.405),color)
 card(c,.29,.315,.495,.08,'#e8f3f1',TEAL);txt(f,.5375,.344,'Mechanics: compose energies and solve for motion',8,TEAL,'bold',ha='center')
 arr(c,(.235,.325),(.285,.355),BLUE)
 xx=x.copy();xx[:,2]+=.22*D.Q[:,0]**2;e.obj(f,[.815,.63,.175,.21],xx[None],e.extent([xx]),base=TEAL);txt(f,.902,.58,'Positions / deformation',7.2,TEAL,'bold',ha='center')
 card(c,.815,.875,.175,.045,'#fff3e7');txt(f,.902,.888,'New tool positions',6.7,GOLD,'bold',ha='center');c.plot([.815,.797,.797],[.897,.897,.405],color=GOLD,lw=1.0);arr(c,(.797,.405),(.775,.365),GOLD)
 arr(c,(.79,.365),(.82,.65),TEAL);arr(c,(.902,.55),(.902,.485),GOLD)
 a=f.add_axes([.815,.345,.175,.17]);a.axis('off');projected=e.project(xx);tri=projected[D.FACES];centers=tri.mean(1);material_centers=D.Q[D.FACES].mean(1)
 for k in np.argsort(centers[:,2]):
  tangent=tri[k,1,:2]-tri[k,0,:2];angle=float(np.degrees(np.arctan2(tangent[1],tangent[0])));rgb=plt.cm.viridis(.15+.7*material_centers[k,0]);a.add_patch(Ellipse(centers[k,:2],.10,.065,angle=angle,fc=rgb,ec='none',alpha=.78))
 lo,hi=e.extent([xx]);a.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),aspect='equal');txt(f,.902,.31,'Gaussian surface',7.2,GOLD,'bold',ha='center');txt(f,.902,.275,'Illustrative schematic',6.2,MUTED,ha='center')
 # Explicit adapter inventory, avoiding a universal-network claim.
 c.plot([.012,.988],[.255,.255],color='#cfdae2',lw=.8)
 adapters=[(.012,'Controlled simulations','Frozen bulk energy\nQ2 mechanics + dynamic contact'),(.345,'Real PhysTwin captures','Neural / analytic equilibrium\nShared Gaussian transport'),(.678,'SoftVTBench manipulation','Analytic shape + correction\nSlip / release + reduced contact')]
 for left,title,body in adapters:
  card(c,left,.05,.307,.17,'#f1f5f8');txt(f,left+.012,.187,title,7.6,INK,'bold');txt(f,left+.012,.105,body,6.9,MUTED,linespacing=1.3)
 txt(f,.5,.007,'Shared physical roles; distinct benchmark adapters. Appearance is not refitted to future images.',6.8,MUTED,ha='center');save(f,'cohesive-method')

def outcomes():
 physical=read('reports/complex-scenes-v18/held-physical-results.json');appearance=read('reports/complex-scenes-v18/held-appearance-summary.json');real=read('reports/real-transfer-v22/summary.json');transfer=read('reports/real-transfer-v22/transfer-scores.json')
 f,axs=plt.subplots(1,3,figsize=(7.15,2.15),gridspec_kw={'width_ratios':[1.05,1,1]})
 methods=['ours','analytic-control','phystwin','spring-gaus'];names=['Ours','Analytic','PhysTwin','Spring-Gaus'];colors=[TEAL,BLUE,GOLD,PURPLE]
 ax=axs[0]
 for m,name,col in zip(methods,names,colors):
  rows=[r for r in physical['rows'] if r['method']==m and r['state']=='complete'];err=float(np.mean([r['metrics']['vector_rms_mm'] for r in rows]));psnr=appearance['method_summary'][m]['completed_query_means']['psnr_foreground_union_db'];ax.scatter(err,psnr,s=36,c=col,zorder=3);ax.annotate(name,(err,psnr),xytext=(4,4 if m not in ['ours','phystwin'] else -10),textcoords='offset points',fontsize=6.4,color=col)
 ax.set(xlabel='Motion RMS (mm) ↓',ylabel='Foreground PSNR (dB) ↑',xlim=(0,16),ylim=(22,40),title='Controlled: 48-query means');ax.grid(alpha=.15)
 vals=[real['geometry'][m]['tracking_mm'] for m in ['neural','analytic','official_phystwin']];b=axs[1].barh(['Ours','Analytic','PhysTwin'][::-1],vals[::-1],color=[GOLD,BLUE,TEAL]);axs[1].bar_label(b,fmt='%.1f',padding=2,fontsize=7);axs[1].set(xlim=(0,49),xlabel='Track error (mm) ↓',title='Real: 8 sequences');axs[1].tick_params(axis='y',labelsize=7)
 vals=[transfer['summary']['initial_geometry_gate_passed'][m]['tracking_mm'] for m in ['neural','analytic','native_frozen']];b=axs[2].barh(['Ours','Analytic','Frozen\nPhysTwin'][::-1],vals[::-1],color=[GOLD,BLUE,TEAL]);axs[2].bar_label(b,fmt='%.1f',padding=2,fontsize=7);axs[2].set(xlim=(0,54),xlabel='Track error (mm) ↓',title='Real transfer: 2 pairs');axs[2].tick_params(axis='y',labelsize=7)
 for a in axs[1:]:a.grid(axis='x',alpha=.15);a.set_axisbelow(True)
 f.tight_layout(pad=.4,w_pad=1.2);save(f,'cohesive-outcomes')

def families():
 f=plt.figure(figsize=(3.45,2.5));records=[]
 for i,(family,name) in enumerate(zip(D.FAMILIES,['Clamp / release','Curved push','Container drop','Rigid impact','Two soft bodies','Heterogeneous'])):
  data,t=D.selected(family);lim=e.extent([data['reference'][-1],data['ours'][-1]]);col=i%3;row=i//3;left=col/3;bottom=.50-row*.46;x=data['reference'][-1]
  e.obj(f,[left+.008,bottom+.015,.315,.33],x,lim,base=BLUE);txt(f,left+.1667,bottom+.39,name,7.5,INK,'bold',ha='center');records.append(dict(family=family,world=family+'-world00',query=0,frame='final',time_s=float(t[-1])))
 txt(f,.5,.005,'Reference geometry; one fixed example per family',6.7,MUTED,ha='center');save(f,'cohesive-families');RECORDS['families']=records

def observability():
 ambiguity=read('reports/compositional-neural-mechanics-v16/appearance-physical-ambiguity.json');summary=read('reports/compositional-neural-mechanics-v16/unknown-action-reaction-paper-summary.json');pos=read('reports/compositional-neural-mechanics-v16/unknown-action-final-independent-audit.json');force=read('reports/compositional-neural-mechanics-v16/unknown-reaction-final-independent-audit.json')
 assert ambiguity['images_bit_identical'] and ambiguity['stiffness_multiplier']==2
 f=plt.figure(figsize=(7.15,2.55));c=f.add_axes([0,0,1,1]);c.axis('off');c.set(xlim=(0,1),ylim=(0,1));c.plot([.48,.48],[.10,.9],color='#d6e0e6',lw=.8)
 txt(f,.012,.94,'A  Appearance cannot identify every parameter',8.2,INK,'bold');x=e.block();lim=e.extent([x])
 for left,title in [(.005,'Material + supports'),(.19,'Both scaled by 2')]:
  e.obj(f,[left,.40,.17,.34],x[None],lim,base=TEAL);txt(f,left+.085,.80,title,6.8,INK,ha='center')
 txt(f,.183,.565,'=',14,MUTED,ha='center');txt(f,.19,.34,'Identical shape and images',7,TEAL,'bold',ha='center');txt(f,.19,.23,'Zero body load; prescribed positions',6.5,MUTED,ha='center');txt(f,.19,.14,'Shape illustration; exact recorded scaling test',6.1,MUTED,ha='center')
 a=f.add_axes([.39,.39,.07,.36]);a.bar([0,1],ambiguity['reaction_norms_N'],color=[BLUE,GOLD],width=.65);a.set(ylim=(0,1.6),xticks=[0,1],xticklabels=['1x','2x']);a.tick_params(labelsize=6,pad=1);a.set_yticks([0,1]);txt(f,.425,.80,'Reaction',6.8,INK,ha='center');txt(f,.425,.27,'Norm (N)',6.2,MUTED,ha='center')
 txt(f,.515,.94,'B  An additional observation improves prediction',8.2,INK,'bold')
 rec={}
 for axpos,data,key,title,unit in [([.55,.29,.17,.48],pos,'position_coordinate_rmse_m','Held positions','RMSE (mm)'),([.80,.29,.17,.48],force,'force_coordinate_rmse_N','Unseen reactions','RMSE (N)')]:
  a=f.add_axes(axpos);pairs=[]
  for world in range(6):
   pair=[]
   for policy in ['initial','fixed_stretch']:
    rows=[r for r in data['rows'] if r['world']==world and r['model']=='shared_laws_fitted' and r['policy']==policy];assert len(rows)==2 and all(r['state']=='complete' for r in rows);pair.append(float(np.mean([r[key] for r in rows]))*(1000 if key.endswith('_m') else 1))
   pairs.append(pair);a.plot([0,1],pair,c='#a7bac8',lw=.8,marker='o',ms=2.8,alpha=.8)
  means=np.mean(pairs,axis=0);a.plot([0,1],means,c=TEAL,lw=2,marker='o',ms=5,zorder=4);a.set(xticks=[0,1],xticklabels=['Initial','+ stretch'],ylabel=unit,xlim=(-.18,1.18),ylim=(0,max(np.max(pairs)*1.10,means.max()*1.3)));a.tick_params(labelsize=6.5);a.grid(axis='y',alpha=.15);a.set_title(title,fontsize=7.5);a.set_ylabel(unit,fontsize=6.8);rec[key]={'six_object_pairs':pairs,'mean':means.tolist()}
  for j,v in enumerate(means):a.annotate(f'{v:.3f}',(j,v),xytext=(-7 if j==0 else 5,6),textcoords='offset points',fontsize=6.5,color=TEAL,weight='bold')
 txt(f,.75,.14,'Six objects; two held commands each',6.7,MUTED,ha='center');txt(f,.75,.065,'Adaptive selection ties fixed stretch; calibration remains unresolved.',6.3,MUTED,ha='center');save(f,'cohesive-observability');RECORDS['observability']=dict(exact_scale=ambiguity,additional_observation=rec,meaning='Different experiments. Zero-load scale gauge illustrated at left; known gravity/density provide independent scale in six-object study at right.')

def real_capture():
 from PIL import Image
 base=ROOT/'runs/real-transfer-v20/render-assets/data/different_types/single_lift_sloth';camera=2;frame=84;labels_path=base/'mask/mask_info_2.json';labels=json.loads(labels_path.read_text());object_id=next(k for k,v in labels.items() if v!='hand');mask_path=base/f'mask/2/{object_id}/84.png';mask=np.asarray(Image.open(mask_path))>0
 rgb_path=ROOT/'runs/editable-scene-v23/phystwin-full-rgb/data/different_types/single_lift_sloth/color/2/84.png';raw=np.asarray(Image.open(rgb_path).convert('RGB'));h,w=mask.shape;ys,xs=np.nonzero(mask);pad=round(.10*max(np.ptp(xs),np.ptp(ys)));box=[max(0,int(ys.min())-pad),min(h,int(ys.max())+pad+1),max(0,int(xs.min())-pad),min(w,int(xs.max())+pad+1)]
 paths=[rgb_path,ROOT/'runs/real-transfer-v20/render-audit/v22-corrected-mask240/single_lift_sloth/official_phystwin-camera2-frame84.png',ROOT/'runs/real-transfer-v20/render-audit/v22-corrected-mask240/single_lift_sloth/neural-camera2-frame84.png'];images=[np.asarray(Image.open(p).convert('RGB')) for p in paths];assert all(im.shape[:2]==mask.shape for im in images)
 f,axs=plt.subplots(1,3,figsize=(3.45,1.55));titles=['Recorded RGB','PhysTwin','Ours']
 for a,im,title in zip(axs,images,titles):a.imshow(im[box[0]:box[1],box[2]:box[3]]);a.set_title(title,fontsize=7.8,color=INK if title=='Recorded RGB' else GOLD if title=='PhysTwin' else TEAL);a.axis('off')
 scores=read('runs/real-transfer-v20/render-audit/v22-corrected-mask240/single_lift_sloth/scores.json');selected={r['method']:r for r in scores['rows'] if r['camera']==camera and r['frame']==frame}
 for a,m in zip(axs[1:],['official_phystwin','neural']):a.text(.5,-.065,f"PSNR {selected[m]['psnr_db']:.2f} dB",ha='center',va='top',fontsize=6.8,transform=a.transAxes)
 f.tight_layout(pad=.15,w_pad=.30);save(f,'cohesive-real-capture')
 for p in paths+[mask_path,labels_path]:SOURCES[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
 RECORDS['real_capture']=dict(case='single_lift_sloth',camera=camera,frame=frame,selection='First declared sequence, final declared frame; not chosen for improvement',crop_yxyx=box,crop_rule='Reference object mask extent plus10percent maximum span, identical pixel bounds for all images',presentation='Recorded RGB retains original background; comparison renders unchanged. No alignment, warping, masking or prediction-dependent crop.',image_metrics={m:selected[m] for m in ['official_phystwin','neural']},interpretation='Continuation failure: native method is stronger on this fixed frame and on aggregate within-sequence reconstruction. Display crop differs from evaluator crop; metrics are unchanged saved scores.')

if __name__=='__main__':
 teaser();method();outcomes();families();observability();real_capture();SOURCES.update(D.SOURCES)
 payload=dict(scope='Presentation of existing saved predictions, no new fitting or experiment; source geometry unaltered',selection=RECORDS,sources=SOURCES,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),limitations=['Simulated references and family-specific adaptations are not universal physics certificates.','Real within-sequence fitting favors official PhysTwin; real transfer contains only two qualifying pairs.','Analytic control is retained and stronger in the controlled aggregate.','Method diagram is schematic; distinct benchmark adapters are explicitly labeled.'])
 (OUT/'cohesive-figure-provenance.json').write_text(json.dumps(payload,indent=2)+'\n');print('Wrote six figures and source provenance')
