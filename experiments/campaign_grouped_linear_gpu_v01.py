"""Vectorized float64 evaluation of the SAME grouped linear positive bank.

No parameter/FE approximation, discarded amplitude atom or Gaussian posterior.
Only the likelihood/moment algebra moves to torch. CPU tensor entry is solely
an independent synthetic parity control; real device use requires pinned0–2.
"""
import os
import numpy as np
from experiments import campaign_grouped_linear_decision_v01 as CPU
ALLOWED_UUIDS=('GPU-5b432ce5-a24a-68ba-150f-682265260a9b','GPU-c49707f6-24fd-38e4-3973-3543ec644ceb','GPU-571378aa-c44e-ffb8-b346-1e409f9821f9')
FORBIDDEN_UUID='GPU-9dbe9939-379d-9560-2028-462a2f8b3ac1'

def pin_config(physical_UUID):
    if physical_UUID not in ALLOWED_UUIDS or physical_UUID==FORBIDDEN_UUID or os.environ.get('CUDA_VISIBLE_DEVICES')!=physical_UUID:raise ValueError('Exactly one explicit GPU0–2 UUID; GPU3 forbidden')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise ValueError('Deterministic cuBLAS workspace required')

def allocation_projection(bank,chunk):
    R,J=bank.amplitude.shape;D=bank.target_ratio.shape[1];A=bank.candidate_ratio.shape[1]
    return int(8*(12*R*J*chunk+6*R*D*chunk+3*R*J+R*(A+D+8))+32*1024**2)

class _TensorGrouped:
    """CPU torch implementation for manufactured controls; no CUDA calls."""
    def __init__(self,bank,*,outcome_chunk=8,allocated_cap_bytes=512*1024**2,_device='cpu'):
        if type(outcome_chunk)is not int or not 1<=outcome_chunk<=8 or type(allocated_cap_bytes)is not int or not 0<allocated_cap_bytes<=512*1024**2:raise ValueError('Explicit bounded outcome chunk/allocation')
        if allocation_projection(bank,outcome_chunk)>allocated_cap_bytes:raise MemoryError('All-atom vectorization allocation preflight exceeds cap')
        import torch
        self.torch=torch;self.device=torch.device(_device);self.bank=bank;self.maximum_chunk=outcome_chunk;self.cap=allocated_cap_bytes
        self._guard()
        for name,array in[('a',bank.amplitude),('logconditional',bank.logconditional),('candidate',bank.candidate_ratio),('target',bank.target_ratio)]:setattr(self,name,torch.as_tensor(np.ascontiguousarray(array),dtype=torch.float64,device=self.device))
        p=np.full(len(bank.row_probability),-np.inf);np.log(bank.row_probability,out=p,where=bank.row_probability>0);self.logrow=torch.as_tensor(p,dtype=torch.float64,device=self.device);self._guard()
    def _guard(self):
        if self.device.type=='cuda'and self.torch.cuda.max_memory_allocated(self.device)>self.cap:raise MemoryError('Owned torch float64 allocation cap')
    def outcome_risk(self,action,y,sigma):
        y=np.asarray(y,float)
        if y.ndim!=1 or not 0<len(y)<=self.maximum_chunk or not np.isfinite(y).all()or not np.isfinite(sigma)or sigma<=0 or not 0<=action<self.bank.candidate_ratio.shape[1]:raise ValueError('Bounded finite scalar outcome input')
        self._guard();t=self.torch
        with t.no_grad():
            value=t.as_tensor(y,dtype=t.float64,device=self.device);ll=self.logconditional[:,:,None]-.5*((self.a[:,:,None]*self.candidate[:,action,None,None]-value)/sigma)**2-np.log(sigma*np.sqrt(2*np.pi));peak=t.max(ll,dim=1).values;w=t.exp(ll-peak[:,None,:]);mass=t.sum(w,dim=1)
            if not bool(t.isfinite(peak).all())or bool((mass<=0).any()):raise FloatingPointError('Unresolved positive tensor outcome mass')
            anchor=self.a[:,0,None];delta=self.a-anchor;md=t.sum(w*delta[:,:,None],dim=1)/mass;m=anchor+md;v=t.sum(w*(delta[:,:,None]-md[:,None,:])**2,dim=1)/mass;rowlog=peak+t.log(mass);joint=self.logrow[:,None]+rowlog;lp=t.logsumexp(joint,dim=0);rho=t.exp(joint-lp);rho=rho/t.sum(rho,dim=0)[None,:]
            rowtarget=m.T[:,:,None]*self.target[None,:,:];dt=rowtarget-rowtarget[:,0:1,:];mean=t.sum(rho.T[:,:,None]*dt,dim=1);variance=t.sum(rho.T[:,:,None]*(v.T[:,:,None]*self.target[None,:,:]**2+(dt-mean[:,None,:])**2),dim=(1,2))/self.target.shape[1]
            if not bool(t.isfinite(lp).all())or not bool(t.isfinite(variance).all())or bool((variance<0).any()):raise FloatingPointError('Invalid centered tensor target variance')
            answer=(lp.cpu().numpy(),variance.cpu().numpy())
        self._guard();return answer
    def expected_risk(self,action,sigma,*,order=16,maximum_outcome_nodes=200000):
        if type(order)is not int or order<2 or not np.isfinite(sigma)or sigma<=0 or not 0<=action<self.bank.candidate_ratio.shape[1]:raise ValueError('Positive fixed tensor outcome plan')
        lo,hi=self.bank.response_bounds(self.bank.candidate_ratio[:,action,None]);a=float(lo[0]-8*sigma);b=float(hi[0]+8*sigma);panels=max(1,int(np.ceil((b-a)/(.5*sigma))))
        if panels*order>maximum_outcome_nodes:raise MemoryError('Outcome node cap')
        nodes,weights=np.polynomial.legendre.leggauss(order);edges=np.linspace(a,b,panels+1);half=np.diff(edges)/2;ys=((edges[:-1]+edges[1:])[:,None]/2+half[:,None]*nodes).ravel();ws=(half[:,None]*weights).ravel();risk=mass=0.
        for start in range(0,len(ys),self.maximum_chunk):
            stop=start+self.maximum_chunk;lp,v=self.outcome_risk(action,ys[start:stop],sigma);p=ws[start:stop]*np.exp(lp);risk+=float(p@v);mass+=float(p.sum())
        from scipy.special import ndtr
        lower,upper=self.bank.response_bounds(self.bank.target_ratio);tail=float(2*ndtr(-8));prior=self.bank.prior_moments()['total_variance']
        if not np.isfinite(risk)or risk<0 or not 0<mass<=1+1e-10:raise FloatingPointError('Unresolved vectorized outcome integral')
        return dict(expected_target_variance=risk,prior_target_variance=prior,variance_reduction=prior-risk,predictive_mass=mass,omitted_mass_bound=tail,omitted_risk_bound=float(tail*np.mean((upper-lower)**2)/4),outcome_nodes=len(ys),order=order)

class TensorGroupedControl(_TensorGrouped):
    """Manufactured CPU tensor controls cannot request a CUDA device."""
    def __init__(self,bank,*,outcome_chunk=8,allocated_cap_bytes=512*1024**2):
        super().__init__(bank,outcome_chunk=outcome_chunk,allocated_cap_bytes=allocated_cap_bytes,_device='cpu')

class Float64GroupedGPU(_TensorGrouped):
    def __init__(self,bank,*,physical_UUID,outcome_chunk=8,allocated_cap_bytes=512*1024**2):
        pin_config(physical_UUID)
        if type(outcome_chunk)is not int or not 1<=outcome_chunk<=8 or type(allocated_cap_bytes)is not int or not 0<allocated_cap_bytes<=512*1024**2:raise ValueError('Explicit bounded GPU chunk/allocation')
        if allocation_projection(bank,outcome_chunk)>allocated_cap_bytes:raise MemoryError('GPU all-atom allocation preflight')
        import torch
        if not torch.cuda.is_available()or torch.cuda.device_count()!=1:raise RuntimeError('Exactly one explicitly pinned CUDA device required')
        torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;device=torch.device('cuda:0');actual=getattr(torch.cuda.get_device_properties(device),'uuid',None)
        if actual is None or str(actual).removeprefix('GPU-')!=physical_UUID.removeprefix('GPU-'):raise RuntimeError('Pinned physical hardware UUID mismatch/unavailable')
        free,_=torch.cuda.mem_get_info(device)
        if free<allocated_cap_bytes+512*1024**2:raise MemoryError('Declared allocation plus512MiB free VRAM reserve required')
        self.physical_UUID=physical_UUID
        super().__init__(bank,outcome_chunk=outcome_chunk,allocated_cap_bytes=allocated_cap_bytes,_device='cuda:0')
