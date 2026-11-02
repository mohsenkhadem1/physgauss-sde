"""Development-only warm-start range repair after failed random-start fit."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import sys,time,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from scripts.compositional_neural_mechanics.interfaces import SharedInterfaceLaw,AnalyticInterfaceLaw,interface_derivatives
from scripts.compositional_neural_mechanics.train_interface_prototype import dataset,metrics

OUT=ROOT/'runs/compositional-neural-mechanics-v16/interface-wide-range-warm'
REPORT=ROOT/'reports/compositional-neural-mechanics-v16'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,data):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,indent=2)+'\n')

def main():
    torch.set_num_threads(1);torch.manual_seed(926401)
    if OUT.exists():raise ValueError('Preserve previous range study')
    OUT.mkdir(parents=True)
    checkpoint_start=ROOT/'runs/compositional-neural-mechanics-v16/interface-prototype/shared-interface.pt'
    files=[checkpoint_start,Path(__file__),ROOT/'scripts/compositional_neural_mechanics/interfaces.py',ROOT/'scripts/compositional_neural_mechanics/train_interface_prototype.py']
    plan=dict(role='Development-only range repair, not reserved scene evaluation',steps=6000,batch=512,
              train_strengths=[.1,.3,.5,.7,.9],test_strengths=[.2,.4,.6,.8],train_radii=[3.,16.],
              train_samples_per_radius=8192,test_radius=16.,extrapolation_radius=24.,
              seeds=dict(train=[926403,926407],test=[926411,926413],batch=926417),
              source_sha256={str(p.relative_to(ROOT)):sha(p) for p in files},
              loss='Per-sample force error/(K*delta_ref+norm(true_force)); energy/(K*delta_ref^2+abs(true_energy)); energy weight0.05',
              initialization='Original small-slip prototype; failed wide random initialization preserved',learning_rate=.003,cosine_final_fraction=.05,time_cap_seconds=1200,
              reason='Declared development scenes exceeded original 2delta_ref training /3delta_ref harder probes; original model retained',
              selection='Fixed final iterate; no response/test checkpoint selection')
    write(REPORT/'interface-wide-range-warm-plan.json',plan)
    pairs=[dataset(plan['train_strengths'],seed,8192,radius) for seed,radius in zip(plan['seeds']['train'],plan['train_radii'])]
    x=torch.cat([p[0] for p in pairs]);z=torch.cat([p[1] for p in pairs])
    teacher=AnalyticInterfaceLaw();truth=teacher.traction(x,z,False).detach();energy_truth=teacher.energy_density(x,z).detach()
    force_scale=z[:,0].exp()*.002+truth.norm(dim=-1)
    energy_scale=z[:,0].exp()*.002**2+energy_truth.abs()
    model=SharedInterfaceLaw();model.load_state_dict(torch.load(checkpoint_start,weights_only=False)['state_dict']);optimizer=torch.optim.Adam(model.parameters(),lr=plan['learning_rate'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,plan['steps'],eta_min=.0003)
    generator=torch.Generator().manual_seed(plan['seeds']['batch']);start=time.monotonic();trace=[]
    for step in range(plan['steps']):
        if time.monotonic()-start>plan['time_cap_seconds']:raise TimeoutError('Range training budget')
        ids=torch.randint(len(x),(plan['batch'],),generator=generator);xx=x[ids].detach().requires_grad_(True)
        W=model.energy_density(xx,z[ids]);P=torch.autograd.grad(W.sum(),xx,create_graph=True)[0]
        loss=((P-truth[ids])/force_scale[ids,None]).square().mean()+.05*((W-energy_truth[ids])/energy_scale[ids]).square().mean()
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();scheduler.step()
        if step%500==0:trace.append(dict(step=step,loss=float(loss.detach())))
    checkpoint=OUT/'shared-interface.pt';torch.save(dict(state_dict=model.state_dict(),reference_length=.002,bases=6,width=16,plan=plan),checkpoint)
    frozen=sha(checkpoint);results={}
    for label,seed,radius in zip(('interpolation','extrapolation'),plan['seeds']['test'],(16.,24.)):
        xx,zz=dataset(plan['test_strengths'],seed,4096,radius)
        entry=metrics(model,xx,zz);entry['families']={str(int(f)):metrics(model,xx[zz[:,2]==f],zz[zz[:,2]==f]) for f in (0.,1.)}
        _,P,H=interface_derivatives(model,xx,zz,True);T=teacher.traction(xx,zz,False)
        relative=(P-T).norm(dim=-1)/(zz[:,0].exp()*.002+T.norm(dim=-1))
        entry['per_sample_normalized_force_error_quantiles']=torch.quantile(relative.detach(),relative.new_tensor([.5,.95,.99,1.])).tolist()
        entry['minimum_sampled_tangent_eigenvalue_N_per_m']=float(torch.linalg.eigvalsh(H).min().detach())
        results[label]=entry
    if sha(checkpoint)!=frozen or any(sha(ROOT/p)!=h for p,h in plan['source_sha256'].items()):raise ValueError('Source/weights changed')
    report=dict(plan=plan,checkpoint_sha256=frozen,training_seconds=time.monotonic()-start,trace=trace,results=results,
                scope='Pointwise oracle attachment-law fit; broader coupons do not demonstrate scene/inverse/contact generalization')
    write(REPORT/'interface-wide-range-warm-results.json',report);print(json.dumps(results),flush=True)

if __name__=='__main__':main()
