"""Frozen-parameter inertia sensitivity within development prefixes only.

This is not dynamically refitted inference. Two substep counts expose numerical
sensitivity separately from whether adding inertia improves motion prediction.
"""
import argparse,json,numpy as np,torch
from .development_run import ROOT,BASE,load,write,sha
from scripts.real_transfer_v19.neural_fem import NeuralFEM
from scripts.external_benchmark.data import phystwin
from scripts.external_benchmark.evaluate import expand_prediction,score

class DynamicNeural(NeuralFEM):
    def evaluate(self,x,theta,frame,tangent=False):
        energy,residual,hessian=super().evaluate(x,theta,frame,tangent)
        if self.inertial_target is not None:
            delta=x-self.rest;energy-=.5e-5*(delta**2).sum();residual-=1e-5*delta
            if hessian is not None:hessian-=1e-5*torch.eye(self.ndof,dtype=self.dtype,device=self.device)
        return energy,residual,hessian

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',default='single_lift_sloth');p.add_argument('--device',default='cuda:0')
    p.add_argument('--substeps',type=int,choices=[2,4,8],default=2);args=p.parse_args();torch.set_num_threads(1)
    folder=BASE/'fits/development-control'/args.case/'neural'
    data,_=load(args.case,120,True);model=DynamicNeural(data,args.device)
    with np.load(folder/'best.npz') as f:theta=model.t(f['theta'])
    targets=model.targets.clone();x,_,_=model.solve(theta,0);velocity=torch.zeros_like(x)
    prediction=[x.cpu().numpy()*model.scale+model.origin];diagnostics=[]
    for frame in range(1,len(data['times'])):
        dt=float(data['times'][frame]-data['times'][frame-1])/args.substeps
        for sub in range(args.substeps):
            fraction=(sub+1)/args.substeps
            model.targets[frame]=targets[frame-1]*(1-fraction)+targets[frame]*fraction
            model.inertial_dt=dt;model.inertial_target=x+dt*velocity
            new,_,diagnostic=model.solve(theta,frame,x)
            velocity=(new-x)/dt;x=new
        model.targets[frame]=targets[frame]
        prediction.append(x.cpu().numpy()*model.scale+model.origin);diagnostics.append(diagnostic)
    output=BASE/'dynamic-diagnostic'/args.case/f'substeps{args.substeps}';output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'prediction.npz',predicted=np.array(prediction))
    evaluation=phystwin(args.case,120);dense=expand_prediction(evaluation,np.array(prediction))
    metrics=score(evaluation,dense,data['train_end'],len(data['times']))['summary']
    write(output/'complete.json',dict(case=args.case,substeps=args.substeps,metrics=metrics,diagnostics=diagnostics,
        source_fit_sha256=sha(folder/'fit.json'),prediction_sha256=sha(output/'prediction.npz'),
        scope='Development only: frozen quasistatically fitted parameters, implicit Euler inertia, no extra velocity damping. Not a dynamically trained model.'))
    print(json.dumps(metrics,indent=2))

if __name__=='__main__':main()
