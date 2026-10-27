"""Package held-out uncertainty evidence; bootstrap independent objects, not pixels."""
from pathlib import Path
import hashlib
import json
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scipy.stats import beta
from scripts.phystwin.uncertainty_reference import BASE, ROOT
from scripts.phystwin.uncertainty_posterior import Z90


def main():
    report=ROOT/'reports/phystwin-uncertainty-v02';report.mkdir(exist_ok=True)
    manifest=json.loads((BASE/'analysis/pretest-manifest.json').read_text())
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in manifest.items()), 'Frozen inputs changed'
    for name in ['development.json','calibration-summary.json','calibration-seal.json','test-summary.json','pretest-manifest.json']:
        shutil.copyfile(BASE/'analysis'/name,report/name)
    shutil.copyfile(BASE/'protocol.json',report/'protocol.json')
    z=np.load(BASE/'analysis/test-predictions.npz');truth=z['truth'];n=24;repeats=4
    rng=np.random.default_rng(311);samples=rng.integers(0,n,size=(10000,n))
    per_object={};intervals={}
    for model in ['continuous_only','discrepancy']:
        per_object[model]={};intervals[model]={}
        for a in range(3):
            mean=z[f'{model}_{a}_mean'];sd=z[f'{model}_{a}_sd'];error=truth-mean
            mse=(error**2).reshape(n,repeats,243).mean((1,2))
            cov=(abs(error)<=Z90*sd).reshape(n,repeats,243).mean((1,2))
            per_object[model][str(a)]=dict(mse=mse.tolist(),coordinate_coverage=cov.tolist())
            intervals[model][str(a)]=dict(rmse95_m=np.quantile(np.sqrt(mse[samples].mean(1)),[.025,.975]).tolist(),
                coverage95=np.quantile(cov[samples].mean(1),[.025,.975]).tolist())
    m=per_object['discrepancy'];initial=np.array(m['0']['mse']);axial=np.array(m['1']['mse']);side=np.array(m['2']['mse']);random=(axial+side)/2
    comparisons=dict(initial_rmse_m=float(np.sqrt(initial.mean())),fixed_axial_rmse_m=float(np.sqrt(axial.mean())),
        fixed_sideways_rmse_m=float(np.sqrt(side.mean())),random_rmse_m=float(np.sqrt(random.mean())),
        fixed_vs_random_rmse_improvement95_m=np.quantile(np.sqrt(random[samples].mean(1))-np.sqrt(axial[samples].mean(1)),[.025,.975]).tolist(),
        fraction_rmse_reduction_after_axial=float(1-np.sqrt(axial.mean()/initial.mean())),
        calibrated_simultaneous_24_of_24_clopper_pearson95=[float(beta.ppf(.025,24,1)),1.])
    checks=[]
    for p in (BASE/'fields').glob('*.npz'):
        with np.load(p) as f:
            checks.append((float(f['max_residual_N']),float(f['minimum_J'])))
    verification=dict(frozen_input_hashes_verified=len(manifest),FEM_jobs=len(checks),
        maximum_residual_N=max(x[0] for x in checks),minimum_J=min(x[1] for x in checks),
        CPU_workers=8,GPU='CUDA_VISIBLE_DEVICES=0, float64 posterior likelihoods and moments',
        inference='FEM reference, not learned PhysTwin uncertainty')
    for name,obj in [('object-bootstrap.json',intervals),('policy-comparison.json',comparisons),('numerical-verification.json',verification)]:
        (report/name).write_text(json.dumps(obj,indent=2)+'\n')
    (BASE/'analysis/per-object-metrics.json').write_text(json.dumps(per_object,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    test=json.loads((report/'test-summary.json').read_text())
    fig,ax=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    labels=['Initial','Axial probe','Sideways probe'];x=np.arange(3)
    for model,color,label in [('continuous_only','#bb7840','Continuous, exact simulator assumed'),('discrepancy','#286da8','Continuous + numerical discrepancy')]:
        p=test[model]['raw']['policies']
        ax[0].plot(x,[v['coordinate_coverage']*100 for v in p],'o-',color=color,label=label)
        ax[1].plot(x,[v['rmse_m']*1000 for v in p],'o-',color=color)
        ax[2].plot(x,[v['mean_width_m']*1000 for v in test[model]['calibrated']['policies']],'o-',color=color)
    ax[0].axhline(90,color='grey',ls='--');ax[0].set_ylabel('Coordinate coverage (%)');ax[0].set_ylim(80,100)
    ax[1].set_ylabel('Target coordinate RMSE (mm)');ax[2].set_ylabel('Calibrated band width (mm)')
    for a in ax:a.set_xticks(x,labels);a.grid(alpha=.2)
    fig.suptitle('24 held-out material/support pairs; fixed geometry; four noise repeats each')
    fig.legend(*ax[0].get_legend_handles_labels(),loc='outside lower center',ncol=2)
    fig.savefig(report/'results.png',dpi=160)
    print(json.dumps(dict(comparisons=comparisons,verification=verification),indent=2))

if __name__=='__main__':main()
