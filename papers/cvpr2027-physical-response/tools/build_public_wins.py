"""Selected public-dataset successes with common crops and unchanged predictions."""
from pathlib import Path
import sys,json,pickle,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PolyCollection
from scipy.spatial import cKDTree,ConvexHull
from PIL import Image
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT));OUT=Path(__file__).resolve().parents[1]/'figures'
from scripts.external_benchmark.data import phystwin
from scripts.external_benchmark.evaluate import expand_prediction
from scripts.contact_benchmark_v12.data import load
TEAL='#008c88';ORANGE='#d9822b';INK='#18354b';GOLD='#e5b323';sources={}
def record(p):
 p=Path(p);sources[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest();return p
def read(p):return json.loads(record(ROOT/p).read_text())
def arr(p):
 with np.load(record(p)) as f:return f['predicted'].copy()
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'savefig.facecolor':'white'})
fig=plt.figure(figsize=(7.05,5.15));outer=fig.add_gridspec(2,1,left=.025,right=.985,top=.88,bottom=.10,hspace=.68)
manifest={'selection':{},'scope':'Illustrative selected wins; unchanged saved forecasts, no new fitting. Real panels are manual-landmark overlays on the same recorded RGB, not predicted RGB or rendering-quality evidence.'}
# Real: strongest qualifying transfer gain, fixed final frame and recorded camera0.
source='single_lift_sloth';case='single_push_sloth';key=source+'--'+case
data=phystwin(source,240);flip=data['evaluation']['transform'];initial=data['support_geometry']@flip
reg=next(r for r in read('reports/real-transfer-v20/initial-registration.json')['pairs'] if r['source']==source and r['target']==case)
initial=initial@(flip@np.array(reg['rotation'])@flip)+np.array(reg['translation_m'])@flip
base=ROOT/'runs/real-transfer-v20';ours=arr(base/'frozen-transfer/v22-corrected-frozen'/key/'neural/prediction.npz')@flip;native=arr(base/'native-transfer'/key/'prediction.npz')@flip
raw=ROOT/'runs/third_party/PhysTwin/data/different_types'/case
with record(raw/'gt_track_3d.pkl').open('rb') as f:truth=np.asarray(pickle.load(f))
with record(raw/'calibrate.pkl').open('rb') as f:C=np.asarray(pickle.load(f))[0]
K=np.array(json.loads(record(raw/'metadata.json').read_text())['intrinsics'])[0]
t=min(len(ours),len(native),len(truth))-1;valid=np.isfinite(truth[0]).all(1)&np.isfinite(truth[t]).all(1);ids=cKDTree(initial).query(truth[0,valid])[1];gt=truth[t,valid];pred=[gt,native[t,ids],ours[t,ids]]
err=[np.zeros(len(gt))]+[np.linalg.norm(x-gt,axis=1)*1000 for x in pred[1:]];point=int(np.argmax(err[1]-err[2]))
def project(p):q=(p-C[:3,3])@C[:3,:3];q=q@K.T;return q[:,:2]/q[:,2,None]
uv=[project(p) for p in pred];rgb=np.asarray(Image.open(record(ROOT/f'runs/editable-scene-v23/phystwin-full-rgb/data/different_types/{case}/color/0/{t}.png')).convert('RGB'))
alluv=np.concatenate(uv);lo=np.maximum(alluv.min(0)-45,0);hi=np.minimum(alluv.max(0)+45,[rgb.shape[1],rgb.shape[0]])
roi=np.stack([u[point] for u in uv]);mid=(roi.min(0)+roi.max(0))/2;extent=np.maximum(roi.max(0)-roi.min(0)+[45,70],[150,110]);zl=mid-extent/2;zh=mid+extent/2
sub=outer[0].subgridspec(2,3,height_ratios=[1.6,1],hspace=.12,wspace=.12)
for j,(name,color) in enumerate([('Recorded + manual tracks',INK),('Frozen PhysTwin adaptation',ORANGE),('Ours: frozen neural adapter',TEAL)]):
 for k in range(2):
  ax=fig.add_subplot(sub[k,j]);ax.imshow(rgb);ax.set(xlim=(lo[0],hi[0]) if k==0 else (zl[0],zh[0]),ylim=(hi[1],lo[1]) if k==0 else (zh[1],zl[1]));ax.set_xticks([]);ax.set_yticks([])
  if j:
   selected=range(len(gt)) if k==0 else [point]
   for n in selected:
    ax.plot([uv[0][n,0],uv[j][n,0]],[uv[0][n,1],uv[j][n,1]],color=color,lw=.9 if k==0 else 1.8)
   ax.scatter(uv[j][list(selected),0],uv[j][list(selected),1],s=15 if k==0 else 45,c=color,edgecolors='white',linewidths=.4,zorder=5)
  chosen=range(len(gt)) if k==0 else [point]
  ax.scatter(uv[0][list(chosen),0],uv[0][list(chosen),1],s=15 if k==0 else 45,facecolors='none',edgecolors='white',linewidths=1,zorder=6)
  if k==0:
   ax.set_title(name,color=color,fontsize=7.3,pad=3);ax.add_patch(Rectangle(zl,*(zh-zl),fill=False,ec=GOLD,lw=1.1))
  else:
   for spine in ax.spines.values():spine.set_color(GOLD);spine.set_linewidth(1.3)
   label='Same highlighted landmark' if j==0 else f'3D error: {err[j][point]:.1f} mm'
   ax.text(.5,-.15,label,transform=ax.transAxes,ha='center',fontsize=7,color=color)
fig.text(.025,.98,'A  PhysTwin real transfer: lift → push',color=INK,weight='bold',fontsize=9,va='top')
fig.text(.025,.947,'Selected successful pair · full-interaction error: 58.23 → 42.78 mm · camera 0, final frame',fontsize=7,color=INK,va='top')
manifest['selection']['phystwin']={'source':source,'target':case,'frame':t,'camera':0,'valid_landmarks':int(valid.sum()),'highlighted_valid_landmark_index':point,'selection_rule':'Only qualifying pair with lower neural than native transfer error; final shared frame; largest 3D per-landmark improvement determines common zoom. All valid landmarks visible in overview.','highlight_error_mm':{'native':float(err[1][point]),'ours':float(err[2][point])},'frame_mean_3d_mm':{'native':float(err[1].mean()),'ours':float(err[2].mean())},'zoom_xyxy':[*zl.tolist(),*zh.tolist()]}
# SoftVT: established strongest episode, fixed75percent forecast frame.
case='spatial-soft-task2-demo41';d=load(dict(subset='spatial-soft',task=2,episode=41));t=d['train_end']+int(.75*(len(d['times'])-d['train_end']-1));p=record(ROOT/f'runs/benchmark-improvement-v14/robust/evaluation/prospective_softvt_phase_b/{case}/predictions.npz')
with np.load(p) as z:seq=[d['evaluation']['points'],expand_prediction(d,z['contact_scene_state']),expand_prediction(d,z['benchmark_v14'])]
pts=[a[t] for a in seq];tri=ConvexHull(seq[0][0]).simplices
az=np.deg2rad(-60);el=np.deg2rad(24);basis=np.array([[-np.sin(az),np.cos(az),0],[-np.sin(el)*np.cos(az),-np.sin(el)*np.sin(az),np.cos(el)],[np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)]])
p2=[a@basis.T for a in pts];coords=np.concatenate(p2);lo=coords[:,:2].min(0);hi=coords[:,:2].max(0);pad=(hi-lo)*.12;lo-=pad;hi+=pad
errors=[np.zeros(len(pts[0]))]+[np.linalg.norm(a-pts[0],axis=1)*1000 for a in pts[1:]];projected_gain=np.linalg.norm(p2[1][:,:2]-p2[0][:,:2],axis=1)-np.linalg.norm(p2[2][:,:2]-p2[0][:,:2],axis=1);projected_gain=np.where(errors[1]>errors[2],projected_gain,-np.inf);idx=int(np.argmax(projected_gain));region=np.stack([a[idx,:2] for a in p2]);c=(region.min(0)+region.max(0))/2;span=np.maximum((hi-lo)*.42,(region.max(0)-region.min(0))*1.4);zl=c-span/2;zh=c+span/2
sub=outer[1].subgridspec(2,3,height_ratios=[1.6,1],hspace=.10,wspace=.12)
from matplotlib import colors
norm=colors.Normalize(0,30);cm=plt.get_cmap('viridis')
for j,(name,col) in enumerate([('Simulator reference',INK),('Mechanics + contact',ORANGE),('Ours: slip-aware prediction',TEAL)]):
 for k in range(2):
  ax=fig.add_subplot(sub[k,j]);ax.set_aspect('equal');ax.set(xlim=(lo[0],hi[0]) if k==0 else (zl[0],zh[0]),ylim=(lo[1],hi[1]) if k==0 else (zl[1],zh[1]));ax.set_xticks([]);ax.set_yticks([])
  order=np.argsort(p2[j][tri,2].mean(1));ts=tri[order];fc=np.tile([.55,.65,.71,1.],(len(ts),1)) if j==0 else cm(norm(errors[j][ts].mean(1)))
  ax.add_collection(PolyCollection(p2[j][ts,:2],facecolors=fc,edgecolors=(1,1,1,.3),linewidths=.18))
  contour=ConvexHull(p2[0][:,:2]).vertices;ax.plot(*np.vstack([p2[0][contour,:2],p2[0][contour[0],:2]]).T,color=INK,lw=.75,ls='--')
  if k==0:ax.set_title(name,color=col,fontsize=7.3,pad=3);ax.add_patch(Rectangle(zl,*(zh-zl),fill=False,ec=GOLD,lw=1.1));[sp.set_visible(False) for sp in ax.spines.values()]
  else:
   [sp.set_color(GOLD) for sp in ax.spines.values()];[sp.set_linewidth(1.3) for sp in ax.spines.values()]
   ax.scatter(*p2[0][idx,:2],s=28,facecolors='white',edgecolors=INK,linewidths=.7,zorder=8)
   if j:
    ax.plot([p2[0][idx,0],p2[j][idx,0]],[p2[0][idx,1],p2[j][idx,1]],color=col,lw=1.5,zorder=7);ax.scatter(*p2[j][idx,:2],s=25,c=col,edgecolors='white',linewidths=.5,zorder=9)
   label='Same material point' if j==0 else f'Point error: {errors[j][idx]:.1f} mm'
   ax.text(.5,-.18,label,transform=ax.transAxes,ha='center',fontsize=7,color=col)
fig.text(.025,.489,'B  SoftVTBench: improved placement after slip',color=INK,weight='bold',fontsize=9,va='top')
fig.text(.025,.456,'Selected best episode-level gain · 22.63 → 17.40 mm · unchanged shape, better position',fontsize=7,color=INK,va='top')
cax=fig.add_axes([.73,.025,.24,.008]);cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cm),cax=cax,orientation='horizontal');cb.set_ticks([0,15,30]);cb.ax.tick_params(labelsize=5.5,pad=1);fig.text(.49,.025,'Point error (mm), common scale →',fontsize=6,color=INK)
manifest['selection']['softvt']={'case':case,'frame':int(t),'selection_rule':'Largest episode-level gain among40fresh episodes, fixed75percent forecast time; zoom uses largest projected-point improvement among nodes improving in3D, identical world-space crop for allmethods.','highlighted_point_index':idx,'highlight_error_mm':{'baseline':float(errors[1][idx]),'ours':float(errors[2][idx])},'frame_coordinate_rmse_mm':{'baseline':float(np.sqrt(np.mean((pts[1]-pts[0])**2))*1000),'ours':float(np.sqrt(np.mean((pts[2]-pts[0])**2))*1000)},'zoom_xyxy':[*zl.tolist(),*zh.tolist()],'surface':'Initial convex hull connectivity, visualization only; unchanged saved nodepositions.','color_scale_mm':[0,30]}
for ext in ['pdf','png']:fig.savefig(OUT/f'public-dataset-wins.{ext}',dpi=220,bbox_inches='tight',pad_inches=.035)
plt.close(fig);manifest['sources_sha256']=sources;manifest['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();(OUT/'public-dataset-wins-provenance.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest['selection'],indent=2))
