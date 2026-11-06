"""Full-cohort conditional temporal stress figure; no outcome-based selection."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .hierarchy_surface_figure import style,clean
from .temporal_hierarchy import ROOT,REPORT,verify
OUT=ROOT/'output/joint-field-temporal-v06'

def main():
 protocol,sha=verify();assert json.loads((REPORT/'completion-audit.json').read_text())['passed'];style();OUT.mkdir(parents=True,exist_ok=True)
 docs=[json.loads((REPORT/shape/f'case-{i:03d}.json').read_text()) for shape in ['block','step'] for i in range(32)]
 assert all(d['protocol_sha256']==sha for d in docs)
 rng=np.random.default_rng(260603);ix=np.concatenate([rng.integers(0,32,(2000,32)),32+rng.integers(0,32,(2000,32))],axis=1)
 models=['fixed-nominal','amplitude-only','scale-correlation','oracle'];labels=['Fixed\nnominal','Scale\nonly','Scale +\ncorr.','Oracle'];colors=['#9A9FA6','#D5984F','#278C8A','#6C7EA4'];records=[]
 fig,axes=plt.subplots(1,3,figsize=(7.2,2.9));fig.subplots_adjust(left=.075,right=.995,bottom=.27,top=.80,wspace=.43)
 specs=[('hidden_rmse_mm',1.,'a  Hidden original motion','Coordinate RMSE (mm)','{:.3f}'),('force_coverage90',100.,'b  Force interval coverage','Coverage (%)','{:.1f}'),('force_interval_width_N',1.,'c  Force interval width','Mean 90% interval width (N)','{:.3f}')]
 for ax,(metric,factor,title,ylabel,fmt) in zip(axes,specs):
  means=[];low=[];high=[]
  for model in models:
   v=np.array([next(r[metric] for r in d['rows'] if r['model']==model and r['condition']=='views') for d in docs])*factor;ci=np.quantile(v[ix].mean(1),[.025,.975]);means.append(v.mean());low.append(ci[0]);high.append(ci[1]);records.append(dict(model=model,metric=metric,factor=factor,mean=float(v.mean()),ci95=ci.tolist()))
  means=np.array(means);low=np.array(low);high=np.array(high);x=np.arange(4)
  ax.bar(x,means,color=colors,width=.65,zorder=2);ax.errorbar(x,means,yerr=np.array([means-low,high-means]),fmt='none',color='black',capsize=2,linewidth=.8,zorder=3)
  upper=max(high)*1.24 if metric!='force_coverage90' else 110
  ax.set_ylim(0,upper);ax.set_xticks(x,labels,fontsize=8);ax.set_ylabel(ylabel,labelpad=3);ax.set_title(title,loc='left',fontweight='bold',pad=6);clean(ax)
  for i,value in enumerate(means):ax.text(i,high[i]+upper*.025,fmt.format(value),ha='center',fontsize=8.5)
  if metric=='force_coverage90':ax.axhline(90,color='black',linestyle='--',linewidth=.8);ax.text(.98,.99,'Nominal 90%',transform=ax.transAxes,ha='right',va='top',fontsize=7.5)
 fig.text(.5,.97,'Temporal-prior stress: after the same four view measurements',ha='center',va='top',fontsize=11)
 fig.text(.5,.075,'64 fresh cases | True correlation length 0.10 | 95% case-bootstrap intervals',ha='center',fontsize=8)
 fig.text(.5,.025,'Post-hoc conditional stress test; coverage is not matched-prior calibration',ha='center',fontsize=8)
 for ext in ['pdf','png']:fig.savefig(OUT/f'comparison.{ext}',facecolor='white',bbox_inches='tight',pad_inches=.025)
 plt.close(fig)
 manifest=dict(protocol_sha256=sha,cases=64,condition='views',records=records,bootstrap='2000 paired case resamples stratified by two fixed geometries,seed260603',caption='Post-hoc temporal-prior stress study motivated by v04 failure, with separately frozen fresh cases. Four fixed view measurements; all64 retained. True length0.10, unknown scale drawn from five-point prior. Oracle knows true scale and length only. Force coverage is conditional on this fixed length, not matched-prior calibration.',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');verify();print(OUT)

if __name__=='__main__':main()
