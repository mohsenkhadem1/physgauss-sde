"""Plot parameter and worst monitored motion traces; uses no future geometry."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pacnerf_comparison import cases,make_model,Target


@torch.no_grad()
def audit(raw,inputs,index):
    raw=Path(raw);scene,views,noise,force=cases()[index]
    name=f'{scene}__{views}views__{noise}__{force}'
    result=raw/f'results-{index}'/name;folder=raw/f'chains-{index}'/name
    summary=json.loads((result/'summary.json').read_text())
    diagnostic=json.loads((result/'diagnostics.json').read_text())
    reference=torch.load(folder/'reference.pt',weights_only=True)
    chains=torch.load(folder/'chains.pt',weights_only=True)
    model,_=make_model(inputs,index);m=model.modes
    pose=[2+m+t*2*m+i for t in range(13) for i in range(m)]
    velocity=[i+m for i in pose]
    worst=lambda indices:max(indices,key=lambda i:diagnostic['quantities'][f'z{i}']['rank_folded_split_rhat'] or 0)
    selected=[0,1,worst(pose),worst(velocity)]
    target=Target(model,reference['mean'],reference['precision'])
    traces=reference['mean'][selected]+chains['white']@target.transform[selected].T
    traces[:,:,2:]*=model.scale[selected[2:]]
    n=traces.shape[1];stride=max(1,n//2000)
    fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True,layout='constrained')
    labels=['Standardized log stiffness','Standardized log damping',
        f'Motion coordinate z{selected[2]} scaled (m)',f'Velocity coordinate z{selected[3]} scaled (m/s)']
    for j,ax in enumerate(axes.flat):
        for chain in range(traces.shape[0]):
            ax.plot(torch.arange(n)[::stride],traces[chain,::stride,j],lw=.5,alpha=.65,label=f'Chain {chain+1}')
        ax.set_ylabel(labels[j]);ax.set_xlabel('Retained draw')
    axes[0,0].legend(ncol=4,fontsize=8)
    status='pass' if summary['valid_for_ranking'] else 'FAIL'
    fig.suptitle(f"{name}: finite convergence checks {status}\nWorst Rhat {summary['worst_rhat']:.3f}, minimum bulk ESS {summary['minimum_bulk_ess']:.1f}; parameter traces alone can miss motion nonstationarity")
    filename=f'chain-trace-audit-{index}'
    fig.savefig(raw/(filename+'.png'),dpi=150);plt.close(fig)
    record={'case':name,'draws_per_chain':n,'finite_chain_checks_pass':summary['valid_for_ranking'],
        'selected_quantities':[f'z{i}' for i in selected],
        'worst_rhat':summary['worst_rhat'],'minimum_bulk_ess':summary['minimum_bulk_ess'],
        'raw_chains_sha256':hashlib.sha256((folder/'chains.pt').read_bytes()).hexdigest(),
        'qualification':'post-hoc trace diagnostic selected by reported scalar Rhat; no future geometry or inference tuning'}
    (raw/(filename+'.json')).write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',required=True);p.add_argument('--inputs',required=True);p.add_argument('--case-index',type=int,required=True)
    a=p.parse_args();torch.set_num_threads(2);audit(a.raw,a.inputs,a.case_index)
