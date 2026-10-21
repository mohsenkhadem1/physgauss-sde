"""Render completed TRAIN numerical audits without accessing reserved responses."""
import argparse
import hashlib
import json
import shutil
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
    axes[0,1].set(title='Finer tested difference is larger',xlabel='Time (s)',ylabel='Coordinate difference RMS (µm)');axes[0,1].legend()
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
    evidence={
        'timestep-analysis.json':campaign/'numerical-timestep-v01/analysis.json',
        'solver-factorial-analysis.json':campaign/'numerical-solver-coordinates-v02/failure-aware-analysis.json',
        'crossed-device-comparison.json':campaign/'numerical-solver-crossed-v01/cross-device-comparison.json',
        'capture-backaction-analysis.json':campaign/'numerical-capture-backaction-v01/analysis.json',
        'attachment-pair-analysis.json':campaign/'numerical-hardware-pair-v01/analysis.json',
        'attachment-geometry-audit.json':campaign/'numerical-attachment-audit-v02.json',
        'rich-force-polynomial-analysis.json':campaign/'numerical-force-polynomial-four-v01/results.json'}
    for name,path in evidence.items():read(path);shutil.copy2(path,output/name)
    links=' · '.join('<a href="'+name+'">'+name.replace('-',' ').replace('.json','')+'</a>' for name in evidence)
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Numerical reference audit</title><style>body{font:17px/1.5 system-ui;max-width:1120px;margin:32px auto;padding:0 20px;color:#203341}img{width:100%;height:auto}.note{padding:16px;background:#edf3f7;border-left:5px solid #b66d32}a{color:#176184}</style>
<h1>The current mechanical reference is repeatable, but numerically sensitive</h1>
<p>TRAIN diagnostics for the Gaussian boundary recovery campaign. No reserved response was used predictively. These are numerical experiments, not a new inference comparison or continuum-accuracy claim.</p>
<div class="note"><p>The finer tested timestep difference is larger: 181.2 → 275.4 µm coordinate RMSE. Increasing solver iterations also fails to establish convergence; one original-coordinate 128-iteration cell hits the unchanged compression guard. Every failed cell is preserved.</p></div>
<img src="numerical-audit.png" alt="Measured timestep sensitivity, hardware effect and held TRAIN force approximation errors">
<p>The coordinate comparison is independently replayed with GPUs 0 and 3 swapped: each condition reproduces bitwise, so that assignment does not explain the coordinate difference. Removing the combined intermediate display and capture block leaves the original trajectory bitwise unchanged. Composed sleep/settling settings are zero; this is not independent native-backend verification.</p>
<p>One controlled attachment-patch change preserves the entire encoded25D input yet changes the response by about30% of its RMS amplitude. Known hardware must condition every forward model equally, or remain an explicit nuisance variable. This does not explain all errors or prove a physical cause for nonlinearity.</p>
<p><a href="../../docs/GAUSSIAN_CAMPAIGN_NUMERICAL_AUDIT.md">Full scientific audit, source and test inventory</a> · <a href="numerical-audit.pdf">PDF figure</a> · <a href="manifest.json">Manifest</a></p>
<p>'''+links+'''</p>
<p>The next predeclared experiment isolates the documented external-forces-every-iteration option. It is not declared a fix; numerical acceptance across four support types remains required before physical calibration.</p></html>'''
    (output/'index.html').write_text(page)
    provenance=output/'provenance';provenance.mkdir();shutil.copy2(__file__,provenance/Path(__file__).name)
    manifest={'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'input_sha256':hashes,
        'files_sha256':{str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in output.rglob('*') if p.is_file()},
        'qualification':'Completed TRAIN diagnostics only, no fitting/calibration/validation gate/confirmation data; no continuum claim'}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(output)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);main(**vars(p.parse_args()))
