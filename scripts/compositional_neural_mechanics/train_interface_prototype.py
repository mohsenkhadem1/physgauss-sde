"""Bounded bilateral interface-law prototype; no scene/reserved data reader."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import hashlib,json,time,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from scripts.compositional_neural_mechanics.interfaces import SharedInterfaceLaw,AnalyticInterfaceLaw,interface_derivatives
OUT=ROOT/'runs/compositional-neural-mechanics-v16/interface-prototype'
REPORT=ROOT/'reports/compositional-neural-mechanics-v16'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')


def dataset(strengths,seed,count,radius):
    rng=torch.Generator().manual_seed(seed)
    z=torch.tensor([[float(torch.log(torch.tensor(K))),s,f] for K in (300.,900.,1800.) for s in strengths for f in (0.,1.)],dtype=torch.float64)
    ids=torch.randint(len(z),(count,),generator=rng);z=z[ids]
    direction=torch.randn(count,3,generator=rng,dtype=torch.float64);direction/=direction.norm(dim=-1,keepdim=True)
    delta=.002*radius*torch.rand(count,1,generator=rng,dtype=torch.float64)*direction
    return delta,z


def metrics(law,delta,z):
    teacher=AnalyticInterfaceLaw();_,truth,_=interface_derivatives(teacher,delta,z)
    _,prediction,_=interface_derivatives(law,delta,z)
    linear=z[:,0].exp()[:,None]*delta
    norm=z[:,0].exp()[:,None]*.002
    def relative(value):return float(((value-truth)/norm).square().sum().sqrt()/((truth/norm).square().sum().sqrt()))
    return dict(normalized_traction_relative_l2=relative(prediction),oracle_linear_relative_l2=relative(linear),
                maximum_traction_error_N=float((prediction-truth).abs().max()),samples=len(delta))


def main():
    torch.set_num_threads(1);torch.manual_seed(926101)
    if OUT.exists():raise ValueError('Preserve existing interface prototype')
    OUT.mkdir(parents=True)
    sources={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'scripts/compositional_neural_mechanics/interfaces.py')}
    plan=dict(role='Pointwise bilateral-interface development prototype, not scene generalization',steps=2500,batch=512,
              train_strengths=[.1,.3,.5,.7,.9],test_strengths=[.2,.4,.6,.8],train_radius_over_ref=2.,hard_radius_over_ref=3.,
              reference_length_m=.002,train_seed=926103,test_seed=926107,hard_seed=926109,source_sha256=sources,
              latent='Oracle [logK, strength, family]; no inference or material-label-from-image claim',time_cap_seconds=600)
    write(REPORT/'interface-prototype-plan.json',plan)
    x,z=dataset(plan['train_strengths'],plan['train_seed'],6144,2.)
    teacher=AnalyticInterfaceLaw();truth=teacher.traction(x,z,False).detach();truth_energy=teacher.energy_density(x,z).detach()
    model=SharedInterfaceLaw();optimizer=torch.optim.Adam(model.parameters(),lr=.006)
    generator=torch.Generator().manual_seed(926113);started=time.monotonic();trace=[]
    for step in range(plan['steps']):
        if time.monotonic()-started>plan['time_cap_seconds']:raise TimeoutError('Interface prototype training cap')
        ids=torch.randint(len(x),(plan['batch'],),generator=generator)
        xx=x[ids].detach().requires_grad_(True);zz=z[ids];energy=model.energy_density(xx,zz)
        traction=torch.autograd.grad(energy.sum(),xx,create_graph=True)[0]
        K=zz[:,0].exp();loss=((traction-truth[ids])/(K[:,None]*.002)).square().mean()+.05*((energy-truth_energy[ids])/(K*.002**2)).square().mean()
        optimizer.zero_grad();loss.backward();optimizer.step()
        if step%250==0:trace.append(dict(step=step,loss=float(loss.detach())))
    checkpoint=OUT/'shared-interface.pt';torch.save(dict(state_dict=model.state_dict(),reference_length=.002,bases=6,width=16,plan=plan),checkpoint)
    checkpoint_sha=sha(checkpoint)
    # Fresh descriptor/strain probes are generated only after weights are fixed.
    results={}
    for label,seed,radius in [('interpolation',plan['test_seed'],2.),('harder_displacement',plan['hard_seed'],3.)]:
        xx,zz=dataset(plan['test_strengths'],seed,2048,radius);results[label]=metrics(model,xx,zz)
        results[label]['families']={str(int(f)):metrics(model,xx[zz[:,2]==f],zz[zz[:,2]==f]) for f in (0.,1.)}
        _,_,H=interface_derivatives(model,xx[:64],zz[:64],True)
        results[label]['minimum_sampled_tangent_eigenvalue_N_per_m']=float(torch.linalg.eigvalsh(H).min())
    if sha(checkpoint)!=checkpoint_sha or any(sha(ROOT/p)!=h for p,h in sources.items()):raise ValueError('Frozen prototype changed')
    write(REPORT/'interface-prototype-results.json',dict(plan=plan,checkpoint_sha256=checkpoint_sha,
          training_seconds=time.monotonic()-started,trace=trace,results=results,
          scope='Oracle latent pointwise attachment-law fit; no contact/friction, scene inference, or composition claim'))
    print(json.dumps(results),flush=True)


if __name__=='__main__':main()
