"""Local orientation-constrained nodal projection, replacing global shrinkage loss."""
import sys,argparse,json,hashlib,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.pidg_comparison.run import configure,EXTERNAL

def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--steps',type=int,default=80);a=p.parse_args();configure(a.gpu);sys.path.insert(0,str(EXTERNAL/'upstream'))
 import torch,numpy as np
 from scipy.signal import savgol_filter
 from scripts.pidg_solver_forecast.common import BASE,load,positions
 torch.set_num_threads(2);g,d=load(a.scene);out=BASE/a.scene/'solver';data=dict(np.load(out/('adapter-before-local-repair.npz' if (out/'adapter-before-local-repair.npz').exists() else 'adapter.npz')));ap=torch.load(out/'appearance.pt',weights_only=False);rest=torch.tensor(data['rest'],device='cuda',dtype=torch.float32);elements=torch.tensor(data['elements'],device='cuda');ids=torch.tensor(data['ids'],device='cuda');N=torch.tensor(data['N'],device='cuda',dtype=torch.float32);ref=(ap['centers'].cuda()-torch.tensor(data['origin'],device='cuda',dtype=torch.float32))/float(data['length']);w=ap['opacity'].cuda().flatten()/N.abs().max(1).values.square().clamp_min(1);dm=(rest[elements[:,1:]]-rest[elements[:,:1]]).transpose(1,2);inv=torch.linalg.inv(dm);grad=torch.cat([-inv.sum(1,keepdim=True),inv],1);results=[];rows=[];start=time.time()
 for frame,t in enumerate(data['times']):
  with torch.no_grad():target=(positions(g,d,float(t))-torch.tensor(data['origin'],device='cuda',dtype=torch.float32))/float(data['length'])
  x=torch.nn.Parameter(torch.tensor(data['x'][frame],device='cuda',dtype=torch.float32));opt=torch.optim.Adam([x],lr=.0015);initial=None;best=None
  for iteration in range(a.steps+1):
   F=torch.einsum('eai,eaj->eij',x[elements],grad);J=torch.linalg.det(F);pred=ref+torch.einsum('pa,pai->pi',N,(x-rest)[ids]);error=((pred-target).square().sum(1)*w).sum()/w.sum();barrier=1e-5*(-torch.log(((J-.1)/.9).clamp_min(1e-8))+(J-1)/.9).mean()+1e-5*(J-1).square().mean();loss=error+barrier
   if initial is None:initial=float(error.detach().sqrt())
   if best is None or float(loss.detach())<best[0]:best=(float(loss.detach()),x.detach().clone(),float(error.detach().sqrt()),float(J.detach().min()))
   if iteration==a.steps:break
   opt.zero_grad(set_to_none=True);loss.backward();before=x.detach().clone();opt.step()
   with torch.no_grad():
    delta=x-before;fraction=torch.ones(len(x),1,device='cuda')
    for backtrack in range(20):
     J=torch.linalg.det(torch.einsum('eai,eaj->eij',x[elements],grad));bad=(~torch.isfinite(J))|(J<=.2)
     if not bool(bad.any()):break
     affected=torch.unique(elements[bad]);fraction[affected]*=.5;x.copy_(before+delta*fraction)
    else:x.copy_(before)
  results.append(best[1].cpu().numpy());rows.append(dict(time=float(t),initial_weighted_rms=initial,final_weighted_rms=best[2],minimum_J=best[3]))
  if frame%20==0:print(frame,rows[-1],time.time()-start,flush=True)

 if not (out/'adapter-before-local-repair.npz').exists():(out/'adapter-before-local-repair.npz').write_bytes((out/'adapter.npz').read_bytes())
 data['x']=np.asarray(results);data['v']=savgol_filter(data['x'],9,3,deriv=1,delta=float(data['times'][1]-data['times'][0]),axis=0);np.savez_compressed(out/'adapter.npz',**data);report=dict(completed=True,steps_per_time=a.steps,rows=rows,seconds=time.time()-start,adapter_sha256=hashlib.sha256((out/'adapter.npz').read_bytes()).hexdigest(),scope='Prefix native Gaussian trajectories projected locally subject to positive tetrahedral orientation. No future images or states. Weighted errors are reconstruction proxies, not ground truth. Original globally shrunk trajectory retained.')
 (out/'projection-fit.json').write_text(json.dumps(report,indent=2));meta=json.loads((out/'adapter.json').read_text());meta['initial_adapter_sha256']=hashlib.sha256((out/'adapter-before-local-repair.npz').read_bytes()).hexdigest();meta['adapter_sha256']=report['adapter_sha256'];meta['local_projection_repair']=True;(out/'adapter.json').write_text(json.dumps(meta,indent=2));print('projection complete',flush=True)
if __name__=='__main__':main()
