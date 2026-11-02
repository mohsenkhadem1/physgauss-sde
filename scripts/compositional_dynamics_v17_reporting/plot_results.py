"""Publication figures from already-scored reports only; never opens raw targets."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];REPORT=ROOT/'reports/compositional-dynamics-v17';OUT=REPORT/'figures'
COLORS={'ours_fine':'#0072B2','ours_coarse':'#56B4E9','native_phystwin':'#D55E00'}

def save(fig,name,source):
 OUT.mkdir(exist_ok=True);fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(OUT/(name+'.svg'),bbox_inches='tight');plt.close(fig)
 sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
 (OUT/(name+'.json')).write_text(json.dumps(dict(source=str(source.relative_to(ROOT)),source_sha256=sha(source),plot_source_sha256=sha(__file__),png_sha256=sha(OUT/(name+'.png')),scope='Unmodified scored report values; no model fitting, target reads, alignment or case selection.'),indent=2)+'\n')

def development():
 path=REPORT/'matched-forecast-development-results.json';data=json.loads(path.read_text());native=data['primary_methods']['native_phystwin']
 fig,axes=plt.subplots(2,2,figsize=(10,6.5),constrained_layout=True);labels={'ours_fine':'Ours (fine inference)','ours_coarse':'Ours (coarse ablation)',native:'Native PhysTwin (calibration-selected)'}
 reference=None
 for name in ['ours_fine','ours_coarse',native]:
  row=data['methods'][name];g=row['geometry'];color=COLORS.get(name,COLORS['native_phystwin']);style='--'if name=='ours_coarse'else'-'
  if not g['rows']:continue
  t=np.array([r['time_s']for r in g['rows']]);label=labels[name]+(' [incomplete]'if not g['complete']else'')
  for ax,key in [(axes[0,0],'world_point_rms_m'),(axes[0,1],'shape_point_rms_m')]:ax.plot(t,[1000*r[key]for r in g['rows']],style,color=color,label=label,lw=1.8)
  p=row['physical_diagnostics']['prediction'];reference=row['physical_diagnostics']['reference']
  axes[1,0].plot(t,np.array(p['surface_centroid_z_m'])*1000,style,color=color,label=label,lw=1.8)
  axes[1,1].plot(t,p['volume_over_rest'],style,color=color,label=label,lw=1.8)
 if reference:
  t=np.arange(reference['frames'])*.01
  axes[1,0].plot(t,np.array(reference['surface_centroid_z_m'])*1000,color='#333333',label='Finite-resolution reference',lw=1.4)
  axes[1,1].plot(t,reference['volume_over_rest'],color='#333333',label='Finite-resolution reference',lw=1.4)
 for ax,y in zip(axes.flat,['World-position vector RMS (mm)','Rigid-aligned shape vector RMS (mm)','Surface-centroid z (mm)','Closed-surface volume / rest volume']):
  ax.set(xlabel='Physical time (s)',ylabel=y);ax.grid(alpha=.2);ax.spines[['right','top']].set_visible(False)
 axes[0,0].legend(fontsize=8);axes[1,0].legend(fontsize=8)
 fig.suptitle('One development object: 10 mm observed drop to 40 mm forecast',fontsize=13)
 save(fig,'development-physical-trajectories',path)

def cohort():
 path=REPORT/'fresh-cohort-geometry-recovery-results.json';data=json.loads(path.read_text());rows=data['rows'];fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True);x=np.arange(len(rows));width=.35
 for ax,key,label in [(axes[0],'world_point_rms_m','World-position vector RMS (mm)'),(axes[1],'shape_point_rms_m','Rigid-aligned shape vector RMS (mm)')]:
  for method,offset,color,title in [('ours',-width/2,COLORS['ours_fine'],'Ours'),('native_phystwin',width/2,COLORS['native_phystwin'],'Native PhysTwin')]:
   vals=np.array([1000*r['methods'].get(method,{}).get('geometry',{}).get(key,np.nan)if r['reference_complete']and r['methods'].get(method,{}).get('completed')else np.nan for r in rows]);valid=np.isfinite(vals)
   ax.bar(x[valid]+offset,vals[valid],width,color=color,label=title)
   for i in np.flatnonzero(~valid):ax.text(x[i]+offset,.02,'x',transform=ax.get_xaxis_transform(),ha='center',color='crimson',fontsize=14)
  ax.set_xticks(x,[r['world_id'].replace('world-','W')+' / '+r['role'].replace('forecast-','')for r in rows],rotation=45,ha='right');ax.set_ylabel(label);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.spines[['right','top']].set_visible(False);ax.legend(fontsize=8)
 fig.suptitle('Four fresh worlds, two forecast heights each | x = unavailable, never zero error',fontsize=12)
 save(fig,'fresh-cohort-physical-comparison',path)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['development','cohort']);args=parser.parse_args();development()if args.stage=='development'else cohort()
