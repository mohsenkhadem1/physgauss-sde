"""Two-mesh continuum-response estimate with implicit derivatives on both meshes.

The leading h^2 correction is an approximation assessed on development oracle
cases; it is not an assertion that the extrapolated field is an exact discrete
equilibrium. Predictive calibration retains the remaining model discrepancy.
"""
import os
import numpy as np
import torch
from scipy import sparse
from scripts.position_ood.runtime import dense_model
from scripts.phystwin.tensor_fixture import basis

class ExtrapolatedPositionFEM:
    def __init__(self,coarse=2,fine=3,device=None,field_frame='world'):
        device=device or os.environ.get('POSITION_INVERSE_DEVICE','cuda:0')
        assert fine>coarse
        self.coarse=dense_model(coarse,device,field_frame=field_frame);self.fine=dense_model(fine,device,field_frame=field_frame);self.geo=self.fine.geo;self.rest=self.fine.rest;self.device=device;self.n=fine;self.field_frame=field_frame
        self.fine_weight=fine**2/(fine**2-coarse**2);self.coarse_weight=coarse**2/(fine**2-coarse**2)
        g=self.coarse.geo;q=self.geo.rest;cell=np.stack([np.clip(np.searchsorted(a,q[:,i],side='right')-1,0,len(a)-2) for i,a in enumerate(g.axes)],axis=1);lo=np.stack([a[cell[:,i]] for i,a in enumerate(g.axes)],axis=1);width=np.stack([np.diff(a)[cell[:,i]] for i,a in enumerate(g.axes)],axis=1);N,_=basis(2*(q-lo)/width-1);element=np.ravel_multi_index(cell.T,tuple(g.cells));cols=g.elements[element];rows=np.broadcast_to(np.arange(len(q))[:,None],cols.shape)
        self.interpolation=sparse.coo_matrix((N.ravel(),(rows.ravel(),cols.ravel())),shape=(len(q),len(g.rest))).tocsr();coo=self.interpolation.tocoo();self.P=torch.sparse_coo_tensor(torch.as_tensor(np.stack([coo.row,coo.col]),dtype=torch.long,device=device),self.tensor(coo.data),size=coo.shape,device=device,check_invariants=True).coalesce()
    def tensor(self,x,dtype=None):return self.fine.tensor(x,dtype)
    def predict(self,theta,poses,layout=0):
        u=self.fine_weight*self.fine.predict(theta,poses,layout)-self.coarse_weight*torch.sparse.mm(self.P,self.coarse.predict(theta,poses,layout))
        with torch.no_grad():
            F=self.fine.eye+torch.einsum('eai,qeaj->qeij',u.detach()[self.fine.elements],self.fine.check_grad)
            if float(torch.linalg.det(F).min())<=.1:raise ValueError('Extrapolated deformation Jacobian guard')
        return u
