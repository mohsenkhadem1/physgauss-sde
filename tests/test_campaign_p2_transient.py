import numpy as np
import torch
from experiments.campaign_p2_transient import P2State,step_p2

class CoupledOscillator:
    def __init__(self):
        self.M=torch.tensor([[2.,.3],[.3,1.]],dtype=torch.float64);self.K=torch.tensor([[9.,-2.],[-2.,5.]],dtype=torch.float64)
        self.mask=torch.zeros((2,1),dtype=torch.bool)
    def mass(self,v):return self.M@v
    def solve(self,force,initial,target,effective_dt,force_tolerance):
        q=torch.linalg.solve(self.K+self.M/effective_dt**2,force+self.M@target/effective_dt**2)
        gradient=self.K@q;residual=gradient+self.M@(q-target)/effective_dt**2-force
        return dict(displacement=q,residual=residual,cache=dict(gradient=gradient,energy=.5*(q*gradient).sum(),minimum_J=torch.tensor(1.)),iterations=1,history=[dict(cg_iterations=0)])


def test_consistent_mass_BDF2_matches_independent_first_order_block_oracle():
    t=CoupledOscillator();state=P2State(torch.zeros((2,1),dtype=torch.float64),torch.zeros((2,1),dtype=torch.float64));previous=None
    q=np.zeros(2);v=np.zeros(2);oldq=oldv=None;h=.01;f=np.array([.5,-.2]);M=t.M.numpy();K=t.K.numpy()
    for step in range(12):
        coefficient=1/h if step==0 else 3/(2*h)
        rhsq=q/h if step==0 else (4*q-oldq)/(2*h)
        rhsv=M@v/h if step==0 else M@(4*v-oldv)/(2*h)
        matrix=np.block([[coefficient*np.eye(2),-np.eye(2)],[K,coefficient*M]])
        expected=np.linalg.solve(matrix,np.r_[rhsq,f+rhsv]);new,row=step_p2(t,state,previous,h,torch.tensor(f[:,None]))
        np.testing.assert_allclose(new.u.numpy()[:,0],expected[:2],atol=2e-16,rtol=0)
        np.testing.assert_allclose(new.v.numpy()[:,0],expected[2:],atol=2e-14,rtol=0)
        assert row['original_physical_free_R_N']<1e-10
        previous,state=state,new;oldq,oldv=q,v;q,v=expected[:2],expected[2:]
