"""Observed-only parameter traces; never opens evaluation images or particles."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from pacnerf_hybrid import conditional_reference_transform


def run(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(args.shards);gate=json.loads(Path(args.convergence).read_text())
    data=[torch.load(root/f'chain-{i}/chains.pt',weights_only=True) for i in range(4)]
    reference=torch.load(root/'chain-0/reference.pt',weights_only=True)
    w=torch.cat([d['white'] for d in data]);n=w.shape[1];d=w.shape[2];core=d-3
    if n!=gate['draws_per_chain']:raise ValueError('Plot budget differs from diagnostic gate')
    transform=conditional_reference_transform(reference['precision'],core)
    indices=[0,1,core,core+1,core+2]
    parameters=reference['mean'][indices]+w[...,indices]@transform[indices][:,indices].T
    # The omitted center and affine determinant are shared constants.
    density=torch.cat([v['log_ratio'] for v in data])-.5*w.square().sum(-1)
    values=torch.cat((parameters,density[...,None]),-1).numpy()
    labels=['Stiffness coordinate','Damping coordinate','Log added diffusion','Log smooth load amplitude','Renderer error hyperparameter','Log density (shared constant omitted)']
    fig,axes=plt.subplots(3,2,figsize=(12,8),layout='constrained')
    for j,(ax,label) in enumerate(zip(axes.flat,labels)):
        for chain in range(4):ax.plot(np.arange(n),values[chain,:,j],lw=.75,alpha=.8,label=f'Chain {chain+1}')
        ax.set_title(label);ax.set_xlabel('Retained draw');ax.grid(alpha=.15)
    axes[0,0].legend(ncol=2,fontsize=8)
    passed=gate['all_monitored_quantities_pass']
    status='CONVERGENCE GATE PASSED; uncertainty not yet validated' if passed else 'NOT CONVERGED — no uncertainty claim'
    fig.suptitle(f'Bird, four independent chains, {n} draws each\n{status}\nFull state/force gate: worst Rhat {gate["worst_rhat"]:.3f}, min bulk/tail ESS {gate["minimum_bulk_ess"]:.1f}/{gate["minimum_tail_ess"]:.1f}',fontsize=12)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=150);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['shards','convergence','output']:p.add_argument('--'+key,required=True)
    run(p.parse_args())
