"""Posthoc CDF illustration of sealed CONDITIONAL2D synthetic results only."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments import campaign_compliant_continuous_reference as model

ROOT=Path(__file__).resolve().parents[1]
FINAL_SHA='4bfa8aabee4545e7f203a9709ce29bc795c77bf63ec0adf7e8f7b0b110fba4f9'

def record(path):
 p=Path(path);return dict(path=p.resolve().relative_to(ROOT).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)

def plot(output,metadata):
 output,metadata=Path(output),Path(metadata)
 if output.exists() or metadata.exists():raise ValueError('Fresh figure/metadata required')
 base=ROOT/'runs/gaussian-campaign-v01';final=base/'compliant-smc-resolution-v04/summary.json'
 if record(final)['sha256']!=FINAL_SHA:raise ValueError('Sealed final six-run result changed')
 result=json.loads(final.read_text())
 if not result['completed'] or not result['passed'] or len(result['runs'])!=6:raise ValueError('Completed conditional2D gate required')
 summary=base/'compliant-continuous-reference-v02/conditional2D/summary.json';d=json.loads(summary.read_text());ref=d['adaptive_reference'][-1]
 inputs=[record(final),record(summary)];units=[]
 for count,folder in [(512,'compliant-continuous-reference-v02/conditional2D'),(32768,None)]:
  for seed in (2026121101,2026121102,2026121103):
   directory=folder or ('compliant-smc-resolution-v03' if seed==2026121101 else 'compliant-smc-resolution-v04')
   path=base/directory/f'smc-{count}-{seed}.npz';inputs.append(record(path))
   with np.load(path,allow_pickle=False) as raw:units.append((count,seed,raw['unit'].copy()))
 chain=base/'compliant-continuous-reference-v02/conditional2D/MCMC.npz';inputs.append(record(chain))
 with np.load(chain,allow_pickle=False) as raw:mc=raw['unit'].reshape(-1,2).copy()
 fig,axes=plt.subplots(1,2,figsize=(11.5,4.7),layout='constrained')
 colors={2026121101:'#2563eb',2026121102:'#d97706',2026121103:'#7c3aed'}
 knots=np.linspace(0,1,33);fine=np.linspace(0,1,401)
 for j,(axis,dimension,label) in enumerate(zip(axes,(0,3),('Young modulus E (Pa)','Fixture stiffness K (N/m)'))):
  physical=lambda z:np.exp(model.original.LOWER[dimension]+z*(model.original.UPPER-model.original.LOWER)[dimension])
  axis.plot(physical(knots),ref['unit_CDF'][j],color='black',linewidth=2.1,marker='.',markersize=4,label='Adaptive (33 CDF evaluations)',zorder=5)
  axis.plot(physical(fine),model.empirical_cdf(mc[:,j],fine),color='#059669',linestyle=':',linewidth=2,label='MCMC (8 prior-started chains)',zorder=4)
  for count,seed,unit in units:
   low=count==512
   axis.plot(physical(fine),model.empirical_cdf(unit[:,j],fine),color='#888888' if low else colors[seed],alpha=.45 if low else .9,
    linewidth=1 if low else 1.2,linestyle='-' if low else '--',label=('512 particles (3 seeds)' if seed==2026121101 else None) if low else '32768, seed '+str(seed))
  axis.set(xlabel=label,ylabel='Posterior cumulative probability',ylim=(0,1));axis.grid(alpha=.18)
 axes[0].legend(fontsize=7.5,loc='lower right',framealpha=.95)
 fig.suptitle('Conditional 2D posterior: independent integration and continuous samplers\n'
  'Center 0.029 m and width 0.031 m FIXED | synthetic linear model | not the full 4D posterior',fontsize=11)
 output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(output,dpi=170);plt.close(fig)
 value=dict(schema='conditional-compliant-CDF-figure-v0.1',completed=True,posthoc_descriptive_only=True,source=record(Path(__file__)),
  sealed_result_sha256=FINAL_SHA,inputs=inputs,figure=record(output),raw_PNG_LOCAL=True,
  no_new_inference_physics_or_cohort_values=True,conditional_center_width_fixed=True,
  old_GL_CC_4D_results_NOT_overlaid=True,adaptive_line_connects_only33_declared_CDF_evaluations=True,
  scope='Illustrates numerical CDF agreement and retained low-count variation; not physical calibration')
 metadata.parent.mkdir(parents=True,exist_ok=True);metadata.write_text(json.dumps(value,indent=2)+'\n')
 return record(metadata)

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--metadata',required=True);a=p.parse_args();print(json.dumps(plot(a.output,a.metadata)))
