"""Local scalar-only plot; preserves all pilot and known-failure cases."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/operator-dev/postpoke-refresh-media-v01'


def disagreement(row):
    ratios=[row['fresh_joint_logZ_range']/.05,row['bridge_conditional_logZ_range']/.05]
    for c in row.get('SMC_vs_MH',row.get('SMC_vs_saved_MH',[]))+row['SMC_pairs']:
        ratios.extend([c['distribution']['max_CDF_difference']/.05,c['distribution']['max_quantile_difference']/.02,
                       c['field']['mean_RMS_difference_m']/c['field']['tolerance_m'],c['field']['SD_RMS_difference_m']/c['field']['tolerance_m'],
                       c['maximum_function_mean_difference_over_SD']/.1])
    return max(ratios)


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    base=ROOT/'runs/gaussian-campaign-v01'
    old=json.loads((base/'postpoke-refresh-v02/summary.json').read_text())
    new=json.loads((base/'postpoke-count-continuation-v01/summary.json').read_text())
    stress=json.loads((base/'postpoke-stress-regression-v01/summary.json').read_text())
    fig,ax=plt.subplots(1,2,figsize=(13,5),layout='constrained');x=np.arange(12);w=.38
    ax[0].bar(x-w/2,[disagreement(r)for r in old['per_object']],w,label='8,192 particles',color='#be6d54');ax[0].bar(x+w/2,[disagreement(r)for r in new['per_object']],w,label='32,768 particles',color='#27878d');ax[0].set(xticks=x,xlabel='All 12 pilot objects: selected poke, one fixed noise',ylabel='Worst disagreement / preset tolerance',title='Particle reference agreement');ax[0].legend()
    labels=[f"{r['case']['object_index']}/{r['case']['action']}/{r['case']['noise_index']}"for r in stress['per_case']]
    ax[1].bar(np.arange(9),[disagreement(r)for r in stress['per_case']],color=['#27878d'if r['passed']else'#be6d54'for r in stress['per_case']]);ax[1].set(xticks=np.arange(9),xticklabels=labels,xlabel='Original failing input: object / action / noise index',title='All nine original importance failures')
    for a in ax:a.axhline(1,color='black',ls='--',label='Acceptance limit');a.spines[['right','top']].set_visible(False);a.set_ylim(bottom=0)
    fig.suptitle('Numerical posterior diagnostics — below 1 passes reference agreement\nMixing and precision gates are checked separately; no policy superiority or physical calibration claim',fontsize=12)
    path=OUT/'posterior-reference-agreement.png';fig.savefig(path,dpi=160);plt.close(fig);print(path)

if __name__=='__main__':run()
