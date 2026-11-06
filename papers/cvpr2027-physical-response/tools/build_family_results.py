"""All six families,24 worlds and four complete methods; no outcome selection.

Each point is one world's mean across its two held queries. Horizontal marks are
four-world means, not confidence intervals. Writes only the dedicated figure
and its provenance/QA inputs; does not regenerate other manuscript figures.
"""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import figure_style
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parents[1]/'figures'
SOURCE=ROOT/'reports/complex-scenes-v18/held-physical-results.json'
D=json.loads(SOURCE.read_text())
METHODS=['ours','analytic-control','phystwin','spring-gaus']
LABELS=['Ours','Analytic','PhysTwin','Spring-\nGaus']
COLORS=['#008c88','#397bc1','#dc8032','#8766ac']
FAMILIES=['clamp_release','curved_push','container_drop','rigid_impact','deformable_pair','heterogeneous']
TITLES=['Clamps / release','Curved pushing','Container drop','Rigid impact','Two soft bodies','Heterogeneous']
rows=[r for r in D['rows'] if r['method'] in METHODS]
assert len(rows)==192 and all(r['state']=='complete' for r in rows)
for m in METHODS:assert sum(r['method']==m for r in rows)==48
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.5,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#183044','text.color':'#183044','axes.edgecolor':'#9aabb6','xtick.color':'#183044','ytick.color':'#183044'})
fig,axs=plt.subplots(2,3,figsize=(6.85,3.4),sharey=True)
records=[]
for a,family,title in zip(axs.flat,FAMILIES,TITLES):
 world_ids=sorted({r['world_id'] for r in rows if r['family']==family});assert len(world_ids)==4
 for j,(m,col) in enumerate(zip(METHODS,COLORS)):
  values=[]
  for w in world_ids:
   rr=[r for r in rows if r['family']==family and r['world_id']==w and r['method']==m];assert len(rr)==2 and {r['query_index'] for r in rr}=={0,1};values.append(float(np.mean([r['metrics']['vector_rms_mm'] for r in rr])))
  mean=float(np.mean(values));a.scatter(j+np.array([-.15,-.05,.05,.15]),values,s=17,facecolors='white',edgecolors=col,linewidths=.9,zorder=3);a.plot([j-.23,j+.23],[mean,mean],c=col,lw=2.2,zorder=4);a.text(j,max(values)+1.1,f'{mean:.2f}',ha='center',va='bottom',fontsize=7,color=col,weight='bold')
  records.append(dict(family=family,method=m,world_ids=world_ids,world_means_mm=values,mean_mm=mean))
 a.set(ylim=(0,32),xlim=(-.5,3.5),xticks=range(4),xticklabels=LABELS,yticks=[0,10,20,30]);a.set_title(title,fontsize=8.5,weight='bold',pad=5);a.tick_params(axis='x',length=0,pad=4,labelsize=7);a.tick_params(axis='y',length=2,labelsize=7);a.grid(axis='y',color='#e6ebef',lw=.7);a.set_axisbelow(True)
for a in axs[:,0]:a.set_ylabel('Motion RMS (mm)',fontsize=7.5)
fig.subplots_adjust(left=.065,right=.99,bottom=.16,top=.925,hspace=.60,wspace=.16)
fig.text(.5,.018,'Open points: 4 worlds per family, 2 queries per world. Colored bars: world means. Lower is better.',ha='center',fontsize=7,color='#65798a')
for ext in ['pdf','png']:fig.savefig(OUT/('cohesive-family-results.'+ext),bbox_inches='tight',pad_inches=.025,dpi=220)
plt.close(fig)
manifest=dict(scope=__doc__,source=str(SOURCE.relative_to(ROOT)),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),coverage={m:48 for m in METHODS},total_worlds=24,total_queries_per_method=48,aggregation='Exactly two queryvectorRMS values averaged within eachworld, then fourworldmean per family. No confidence intervals plotted.',rows=records,excluded='UniPhy has1completed,7failed,40unsupported; not plotted as comparable fullcohort method.',reference_caveat=D.get('numerical_reference_note'))
(OUT/'cohesive-family-results-provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('All4methods complete48/48; wrote familyfigure only')
