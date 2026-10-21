"""Render completed TRAIN numerical audits without accessing reserved responses."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    campaign=Path('runs/gaussian-campaign-v01');hashes={}
    def read(path):
        path=Path(path);hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())
    rate=read(campaign/'numerical-timestep-v01/analysis.json')
    hardware=read(campaign/'numerical-hardware-pair-v01/analysis.json')
    rich=read(campaign/'numerical-force-polynomial-four-v01/results.json')
    fig,axes=plt.subplots(2,2,figsize=(11.4,7.5),layout='constrained')
    qs={}
    for hz in [480,960,1920]:
        root=campaign/f'numerical-timestep-v01/hz{hz}/captures/private/t0001'
        if not read(root/'validation.json')['passed']:raise ValueError('Unvalidated rate')
        with np.load(root/'reference.npz') as raw:rest=raw['rest_points_m']
        path=root/'evaluation/trajectory.npz';hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as raw:q,t=raw['positions_m'].astype(float),raw['times_s']
        qs[hz]=q;axes[0,0].plot(t,np.sqrt(np.mean((q-rest)**2,axis=(1,2)))*1e6,label=f'{hz} Hz')
    axes[0,0].axvline(1.65,color='#666',linestyle=':',linewidth=1);axes[0,0].set(title='Same diagonal force; timestep changes response',xlabel='Time (s)',ylabel='Coordinate RMS displacement (µm)');axes[0,0].legend()
    for low,high in [(480,960),(960,1920)]:
        axes[0,1].plot(t,np.sqrt(np.mean((qs[low]-qs[high])**2,axis=(1,2)))*1e6,label=f'{low} → {high} Hz')
    axes[0,1].set(title='Successive refinements diverge',xlabel='Time (s)',ylabel='Coordinate difference RMS (µm)');axes[0,1].legend()
    values=[]
    for variant in ['original','counterfactual']:
        root=campaign/f'numerical-hardware-pair-v01/{variant}/captures/private/t0000'
        if not read(root/'validation.json')['passed']:raise ValueError('Unvalidated hardware variant')
        with np.load(root/'evaluation/trajectory.npz') as raw:q,t=raw['positions_m'].astype(float),raw['times_s']
        with np.load(root/'zero_force/trajectory.npz') as raw:zero=raw['positions_m'].astype(float)
        values.append(q-zero)
    axes[1,0].plot(t,np.sqrt(np.mean(values[0]**2,axis=(1,2)))*1e6,label='Original response')
    axes[1,0].plot(t,np.sqrt(np.mean((values[1]-values[0])**2,axis=(1,2)))*1e6,label='Patch-change effect')
    axes[1,0].set(title='Same encoded input; attachment matters',xlabel='Time (s)',ylabel='Coordinate RMS (µm)');axes[1,0].legend()
    x=np.arange(len(rich['rows']));w=.36
    for degree,offset,label in [(1,-w/2,'Linear'),(2,w/2,'Quadratic')]:
        ratio=[r['models'][str(degree)]['held_diagonal_coordinate_rmse_m']/r['zero_baseline_coordinate_rmse_m'] for r in rich['rows']]
        axes[1,1].bar(x+offset,ratio,w,label=label)
    axes[1,1].set_xticks(x,[r['episode_id'] for r in rich['rows']]);axes[1,1].set(title='Held TRAIN diagonal: quadratic remains inaccurate',ylabel='Error / measured force-induced RMS');axes[1,1].legend()
    for ax in axes.flat:ax.grid(alpha=.16)
    fig.savefig(output/'numerical-audit.png',dpi=170);fig.savefig(output/'numerical-audit.pdf');plt.close(fig)
    manifest={'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'input_sha256':hashes,
        'qualification':'Completed TRAIN diagnostics only, no fitting/calibration/validation gate/confirmation data; no continuum claim'}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(output)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);main(**vars(p.parse_args()))
