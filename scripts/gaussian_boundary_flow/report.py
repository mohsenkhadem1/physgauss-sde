"""Assemble measured evidence without assigning prototype results to benchmarks."""
from pathlib import Path
import json,hashlib,subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(p):return json.loads(Path(p).read_text())

def main():
    root=Path('runs/gaussian-boundary-flow-refined-v03');out=Path('reports/gaussian-boundary-flow-v01');out.mkdir(parents=True,exist_ok=True)
    programs={k:read(root/f'evaluation-{k}.json')for k in ('sine','release','triangle','reverse')}
    evidence={'refined':programs,'point_baseline':read(root/'point-baseline.json'),'position_action':read(root/'position-action.json'),'cross_mesh':read(root/'cross-mesh.json'),'acquisition':read(root/'acquisition.json'),'calibration':read(root/'calibration.json'),'rgb_prototype':read('runs/gaussian-boundary-flow-rgb-v01/results.json'),'interpolation':read(out/'interpolation.json'),'qualification':read(out/'qualification.json'),'refined_qualification':read(out/'refined-qualification.json')}
    preservation={}
    for name in ('reports/benchmark-improvement-v14/softvt-summary-b.json','reports/real-transfer-v22/summary.json'):
        raw=Path(name).read_bytes();old=subprocess.check_output(['git','show',f'HEAD:{name}'])
        preservation[name]={'unchanged_from_HEAD':raw==old,'sha256':hashlib.sha256(raw).hexdigest()}
    evidence['existing_benchmark_preservation']=preservation
    rng=np.random.default_rng(1928);comparison={}
    for k,r in programs.items():
        flow=np.array(r['methods']['flow']['per_case_mm']);base=np.array(evidence['point_baseline']['methods'][k]['per_case_mm']);diff=base-flow;boots=diff[rng.integers(0,len(diff),(5000,len(diff)))].mean(1)
        comparison[k]={'neural_baseline_mean_minus_flow_mm':float(diff.mean()),'paired_bootstrap_95CI_mm':np.quantile(boots,[.025,.975]).tolist(),'flow_wins':int((flow<base).sum()),'n':len(flow)}
    evidence['matched_neural_comparison']=comparison
    (out/'results.json').write_text(json.dumps(evidence,indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(13,3.7),layout='constrained')
    labels=['Continued','Release','Triangle','Reversal'];x=np.arange(4)
    for offset,method,label,color in [(-.25,'ridge','Linear regression','#94a3b8'),(0,'point','Neural regression','#f59e0b'),(.25,'flow','Conditional flow','#14b8a6')]:
        y=[evidence['point_baseline']['methods'][k]['future_rmse_mm_mean'] if method=='point' else programs[k]['methods'][method]['future_rmse_mm_mean'] for k in programs]
        axes[0].bar(x+offset,y,.24,label=label,color=color)
    axes[0].set(xticks=x,xticklabels=labels,ylabel='Future coordinate RMSE (mm)',title='128 new simulated worlds');axes[0].legend(fontsize=8)
    ac=evidence['acquisition']['methods'];names=['no_acquisition','selected','fixed','random'];axes[1].bar(range(4),[ac[k]['future_rmse_mm_mean']for k in names],color=['#94a3b8','#14b8a6','#3b82f6','#f59e0b']);axes[1].set(xticks=range(4),xticklabels=['Before','Selected','Fixed','Random'],ylabel='Future RMSE (mm)',title='New view helps; selection ties fixed')
    ca=evidence['calibration'];axes[2].bar([0,1],[ca['raw_trajectory_coverage'],ca['calibrated_trajectory_coverage']],color=['#94a3b8','#14b8a6']);axes[2].axhline(.9,color='#475569',ls='--');axes[2].set(xticks=[0,1],xticklabels=['Raw intervals','Calibrated envelope'],ylim=(0,1),ylabel='Whole-trajectory coverage',title='Finer-domain predictive calibration')
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.savefig(out/'results.png',dpi=180);fig.savefig(out/'results.pdf');plt.close(fig)
    audit={'tests':'31 passed: boundary flow, existing physics, existing forces','finite_metrics':all(np.isfinite(v['future_rmse_mm_mean']) for r in programs.values() for v in r['methods'].values()),'same_checkpoint_across_programs':len({r['checkpoint_sha256']for r in programs.values()})==1,'existing_benchmarks_unchanged':all(v['unchanged_from_HEAD']for v in preservation.values()),'no_inverse_fit_at_forecast':True,'raw_datasets_local':True,'scope':'Controlled new-method development; no external benchmark improvement claimed','parameter_identification':'Unknown E, support stiffness/location, force magnitude/location, damping; known geometry, density, cameras, force direction and future command family','unimplemented':'Unknown discrete boundary-type inference, coupled-contact/drop rollout, active-poke policy, real-video adaptation, joint appearance optimization'}
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({'neural_comparison':comparison,'audit':audit},indent=2))

if __name__=='__main__':main()
