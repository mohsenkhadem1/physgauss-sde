"""Static scientific plots and old-reference comparison from retained fields."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/phystwin-fixture-convergence-v04'
BASE=ROOT/'runs/phystwin-fixture-convergence-v04/fields'
s=json.loads((OUT/'summary.json').read_text())
fig,axes=plt.subplots(1,3,figsize=(13,4.2))
for ax,key,label,limit in zip(axes,['rms_mm','strain_relative','traction_relative'],['Displacement change (mm RMS)','Interior strain relative change','Support traction relative change'],[.02,.05,.02]):
    for action in s['protocol']['actions']:
        rows=[r for r in s['comparisons'] if r['action']==action and r['kind']=='spatial' and r['fine'].startswith('p2g-')]
        ax.loglog([int(r['fine'].split('-')[1][1:]) for r in rows],[r[key] for r in rows],'o-',label=action.replace('_',' '))
    ax.axhline(limit,color='black',ls='--',label='Frozen tolerance');ax.set_xlabel('Finer mesh level n');ax.set_ylabel(label);ax.grid(alpha=.25)
axes[0].legend(fontsize=8);fig.suptitle('Difficult half-face fixture: successive graded quadratic refinements')
fig.tight_layout();fig.savefig(OUT/'convergence.png',dpi=170);plt.close(fig)
old=np.load(ROOT/'runs/phystwin-active-support-v03/fields/development-11-n8.npz')['fields'];comparison={}
x=np.stack(np.meshgrid(np.linspace(0,.24,17),np.linspace(-.04,.04,5),np.linspace(-.04,.04,5),indexing='ij'),-1).reshape(-1,3)
motion={}
boundary_only={}
old12=np.load(ROOT/'runs/phystwin-active-support-v03/fields/development-11-n12.npz')['fields']
for i,action in enumerate(s['protocol']['actions']):
    files=sorted(BASE.glob(f'p2g-n*-q3-{action}.npz'),key=lambda f:int(f.name.split('-')[1][1:]))
    f=files[-1];u=np.load(f)['sampled_u'].reshape(17,5,5,3)[::2,::2,::2].reshape(81,3)
    comparison[action]=dict(reference=f.stem,rms_difference_mm=float(np.sqrt(np.mean((old[i]-u)**2))*1000),maximum_coordinate_difference_mm=float(abs(old[i]-u).max()*1000))
    z=np.load(f);y=x+z['sampled_u'];X=x-x.mean(0);Y=y-y.mean(0)
    v,_,w=np.linalg.svd(X.T@Y);D=np.eye(3);D[-1,-1]=np.linalg.det(v@w)
    nonrigid=float(np.sqrt(np.mean((X@v@D@w-Y)**2))*1000)
    total=float(np.sqrt(np.mean(z['sampled_u']**2))*1000)
    motion[action]=dict(reference=f.stem,motion_rms_mm=total,nonrigid_rms_mm=nonrigid,nonrigid_to_motion_rms_ratio=nonrigid/total,strain_rms=float(z['strain_rms']))
    same_mesh=BASE/f'p1-n12-q3-{action}.npz'
    if same_mesh.exists():
        consistent=np.load(same_mesh)['sampled_u'].reshape(17,5,5,3)[::2,::2,::2].reshape(81,3)
        boundary_only[action]=dict(rms_difference_mm=float(np.sqrt(np.mean((old12[i]-consistent)**2))*1000),note='Same P1 n=12 mesh: lumped versus consistent boundary integration')
(OUT/'legacy-reference-difference.json').write_text(json.dumps(comparison,indent=2)+'\n')
(OUT/'motion-diagnostic.json').write_text(json.dumps(motion,indent=2)+'\n')
(OUT/'boundary-integration-difference.json').write_text(json.dumps(boundary_only,indent=2)+'\n')
