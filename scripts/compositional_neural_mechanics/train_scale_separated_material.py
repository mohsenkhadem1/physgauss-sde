"""Bounded oracle-latent constitutive prototype, not a scene benchmark."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,hashlib,itertools,json,time
from pathlib import Path
import numpy as np
import torch
from scripts.compositional_neural_mechanics.materials import teacher_energy
from scripts.compositional_neural_mechanics.scale_separated_material import ScaleSeparatedMaterialLaw
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/compositional-neural-mechanics-v16/scale-separated-prototype';REPORT=ROOT/'reports/compositional-neural-mechanics-v16'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')
def data(n,descriptors,strain,seed,device):
 rng=np.random.default_rng(seed);z=np.asarray(descriptors)[rng.integers(len(descriptors),size=n)]
 diagonal=np.exp(rng.uniform(-strain,strain,(n,3)));F=np.zeros((n,3,3))
 F[:,range(3),range(3)]=diagonal
 F[:,0,1]=rng.uniform(-2*strain,2*strain,n);F[:,0,2]=rng.uniform(-2*strain,2*strain,n);F[:,1,2]=rng.uniform(-2*strain,2*strain,n)
 F=torch.tensor(F,dtype=torch.float64,device=device,requires_grad=True);z=torch.tensor(z,dtype=torch.float64,device=device)
 W=torch.where(z[:,2]<0,teacher_energy(F,z[:,:2],'mooney_rivlin'),teacher_energy(F,z[:,:2],'yeoh'))
 P=torch.autograd.grad(W.sum(),F)[0]
 return F.detach(),z,W.detach(),P.detach()
def metrics(law,dataset):
 F,z,W,P=dataset;predW=law(F,z);predP=law.stress(F,z,create_graph=False)
 result={}
 for name,mask in [('all',torch.ones(len(F),dtype=torch.bool,device=F.device)),('mooney_rivlin',z[:,2]<0),('yeoh',z[:,2]>0)]:
  def relative(a,b):return float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b).clamp_min(1e-12))
  result[name]=dict(energy_relative_l2=relative(predW[mask],W[mask]),stress_relative_l2=relative(predP[mask],P[mask]),stress_rmse_Pa=float(torch.mean((predP[mask]-P[mask])**2).sqrt()))
 return result

def tangent_audit(law,F,z):
 values=[];rng=np.random.default_rng(69)
 for i in range(min(24,len(F))):
  x=F[i].detach().requires_grad_(True);P=law.stress(x,z[i]);a=torch.tensor(rng.normal(size=3),device=x.device,dtype=x.dtype);b=torch.tensor(rng.normal(size=3),device=x.device,dtype=x.dtype);H=torch.outer(a/a.norm(),b/b.norm())
  A_H=torch.autograd.grad((P*H).sum(),x)[0];values.append(float((A_H*H).sum()))
 return dict(sampled_rank_one_curvatures_Pa=values,minimum_Pa=min(values),negative_count=sum(v<0 for v in values),scope='Finite sampled directional checks only, not proof of rank-one convexity/stability.')
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--device',default='cuda:0');parser.add_argument('--steps',type=int,default=6000);args=parser.parse_args()
 if OUT.exists():raise ValueError('Preserve prior prototype; use separate declared experiment')
 torch.set_num_threads(1);torch.manual_seed(931222);np.random.seed(931222)
 trainz=list(itertools.product([-.5,0,.5],[-1.,0.,1.],[-1.,1.]));testz=list(itertools.product([-.25,.25],[-.5,.5],[-1.,1.]))
 sources={str(p.relative_to(ROOT)):sha(p)for p in (Path(__file__).resolve(),ROOT/'scripts/compositional_neural_mechanics/materials.py',ROOT/'scripts/compositional_neural_mechanics/scale_separated_material.py')}
 plan=dict(scope='Exact common-scale structural correction, not selected using prior hard-test outcomes. Fresh strain draws on the same declared18/8descriptorgrids; oracle constitutive prototype only. Fixed final6000-step checkpoint, no validation selection.',seed=931222,latent=['log common scale','generative nonlinear shape descriptor','family descriptor: -1 MR,+1 Yeoh'],training_materials=trainz,unseen_material_descriptors=testz,train_strain='upper-triangular diagonal log entries±.2, upper offdiagonal shear±.4; J>0',test_strain='interpolation same strainrange AND harder diagonal log entries±.35/shear±.7',counts=dict(train=4096,validation=512,test_interpolation=1024,test_harder=1024),steps=args.steps,optimizer='Adam lr.003 cosine to0, fixed6000steps, stressMSE+.1energyMSE normalized12000Pa; no test-based selection',batch=256,device=args.device,budget_seconds=1800,source_sha256=sources)
 write(REPORT/'scale-separated-prototype-plan.json',plan);OUT.mkdir(parents=True)
 train=data(4096,trainz,.2,101,args.device);val=data(512,trainz,.2,102,args.device)
 law=ScaleSeparatedMaterialLaw().double().to(args.device);optimizer=torch.optim.Adam(law.parameters(),lr=.003);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.steps)
 started=time.monotonic();history=[]
 for step in range(args.steps):
  if time.monotonic()-started>1800:raise TimeoutError('Constitutive training stage exceeded30minutes')
  ix=torch.randint(len(train[0]),(256,),device=args.device);F=train[0][ix].clone().requires_grad_(True);z=train[1][ix];W=law(F,z);P=torch.autograd.grad(W.sum(),F,create_graph=True)[0]
  loss=((P-train[3][ix])/12000).square().mean()+.1*((W-train[2][ix])/12000).square().mean()
  optimizer.zero_grad();loss.backward();optimizer.step();scheduler.step()
  if step%1000==0:history.append(dict(step=step,loss=float(loss)));print(history[-1],flush=True)
 # Freeze shared weights BEFORE generating the two declared constitutive test sets.
 law.eval()
 for p in law.parameters():p.requires_grad_(False)
 checkpoint=OUT/'shared-law.pt';torch.save(dict(state_dict=law.state_dict(),latent_dim=3,source_sha256=sources,steps=args.steps),checkpoint)
 freeze=dict(checkpoint_sha256=sha(checkpoint),source_sha256=sources,train_seconds=time.monotonic()-started,plan_sha256=sha(REPORT/'scale-separated-prototype-plan.json'))
 write(OUT/'weight-freeze.json',freeze)
 test=data(1024,testz,.2,103,args.device);hard=data(1024,testz,.35,104,args.device)
 results=dict(scope=plan['scope'],frozen_checkpoint=freeze,history=history,training=metrics(law,train),validation=metrics(law,val),unseen_descriptor_interpolation=metrics(law,test),harder_strain_unseen_descriptors=metrics(law,hard),adversarial_sample_tangents=tangent_audit(law,hard[0],hard[1]))
 if any(sha(ROOT/p)!=digest for p,digest in sources.items()):raise ValueError('Constitutive source changed duringrun')
 write(REPORT/'scale-separated-prototype-results.json',results);print(json.dumps(results,indent=2))
if __name__=='__main__':main()
