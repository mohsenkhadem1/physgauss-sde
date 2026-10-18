"""Evaluation-only image renders at supplied true geometry and its modal projection.

No appearance is refitted. Future geometry never enters production inference.
These are diagnostic renders, not achievable RGB bounds or sparse reconstructions.
"""
import argparse
import json
from pathlib import Path
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pacnerf_comparison import make_model
from physgauss.pacnerf_model import PACRenderer
from physgauss.cli import source_fingerprint


@torch.no_grad()
def audit(inputs,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    rows=[];fig,axes=plt.subplots(2,4,figsize=(12,6),layout='constrained')
    for row,index in enumerate([4,12]):
        model,_=make_model(inputs,index);scene=model.data['scene']
        truth=torch.load(Path(inputs)/f'{scene}-evaluation.pt',weights_only=True)['particles']
        basis=model.renderer.gaussian_basis.flatten(0,1)
        q=torch.linalg.lstsq(basis,(truth-model.data['gaussians'][None]).flatten(1).T).solution.T
        errors={k:[] for k in ['true_centers','subspace_oracle']}
        region_errors={k:[] for k in errors};region_counts=[];frames={}
        for frame in range(1,14):
            cameras=[c for c in range(11) if (frame,c) not in model.specs]
            specs=[(frame,c) for c in cameras]
            target=torch.stack([model.data['images'][c,frame] for c in cameras])
            background=torch.stack([model.data['backgrounds'][c] for c in cameras])
            region=(target-background).abs().amax(-1)>.10;region_counts.append(region.sum()*3)
            direct=PACRenderer(dict(model.data,gaussians=truth[frame]),model.basis)
            direct.colors=model.renderer.colors
            rgb={'true_centers':direct(q.new_zeros(1,len(specs),model.modes),specs)[0],
                'subspace_oracle':model.renderer(q[frame][None,None].expand(1,len(specs),-1),specs)[0]}
            for key,value in rgb.items():
                errors[key].append((value-target).square().flatten(1).mean(-1))
                region_errors[key].append(((value-target).square()*region[...,None]).sum())
            if frame==6:
                c=cameras.index(0);frames={'target':target[c],**{k:v[c] for k,v in rgb.items()}}
        metrics={key:{'heldout_rgb_psnr_db':float(-10*torch.log10(torch.cat(errors[key]).mean())),
            'heldout_background_difference_region_psnr_db':float(-10*torch.log10(torch.stack(region_errors[key]).sum()/torch.stack(region_counts).sum().clamp_min(1)))} for key in errors}
        rows.append({'scene':scene,'schedule':'six-view labels include frame zero; five later dynamic images',
            'heldout_images':sum(len(v) for v in errors['true_centers']),
            'renderers':metrics,'appearance':'same fixed initial-only colors, radius .045, opacity .7, 24 slabs, 32px; no refit'})
        images=[frames['target'],frames['true_centers'],frames['subspace_oracle'],
            (frames['target']-frames['true_centers']).abs().mean(-1)]
        titles=['Official image','True particle centers','Projected geometry oracle','True-center RGB error (0–0.3)']
        for col,(value,title) in enumerate(zip(images,titles)):
            ax=axes[row,col]
            if col==3:ax.imshow(value,vmin=0,vmax=.3,cmap='magma')
            else:ax.imshow(value.clamp(0,1))
            ax.set_title(scene.title()+': '+title,fontsize=10);ax.axis('off')
    fig.suptitle('Evaluation-only renderer audit: frame 6, camera 0\nSupplied future geometry is used only here; fixed appearance; these are not sparse reconstructions',fontsize=12)
    fig.savefig(output/'renderer-reference-audit.png',dpi=150);plt.close(fig)
    result={'source_fingerprint':source_fingerprint(),'scenes':rows,
        'qualification':'post-hoc diagnostic at true centers / evaluation-only least-squares modal projection; no new training; these are not RGB lower bounds or reconstructions from sparse observations; production experiment unchanged'}
    (output/'renderer-reference-diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();torch.set_num_threads(2)
    import torch.backends.opt_einsum as backend
    backend.enabled=True;backend.strategy='greedy'
    audit(a.inputs,a.output)
