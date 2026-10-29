"""Object-attached spatial material field for the position-driven reference."""
import numpy as np
from scipy import sparse
from scripts.position_ood.fixture import PositionFixture
from scripts.phystwin.tensor_fixture import GPUHex,basis

class MaterialFieldGPUHex(GPUHex):
    def elastic(self,positions,tangent=False):
        t=self.t;m=self.model;u=self.tensor(positions)-self.rest;ue=u[self.elements]
        N,_=basis(m.quad);X=t.einsum('qa,eai->qei',self.tensor(N),self.rest[self.elements])
        feature=t.exp(-((X[...,0]-.06)/.07)**2)-t.exp(-((X[...,0]-.18)/.07)**2)
        E=m.young*t.exp(m.material_gradient*feature);mu=E/(2*1.35);lam=E*.35/(1.35*.3)
        energy=t.zeros((),dtype=t.float64,device=self.device);ge=t.zeros_like(ue);Js=[]
        for start in range(0,len(self.check_bary),self.chunk):
            F=self.I+t.einsum('eai,qeaj->qeij',ue,self.gradients(self.check_bary[start:start+self.chunk]));Js.append(t.linalg.det(F).reshape(-1))
        for start in range(0,len(self.bary),self.chunk):
            q=slice(start,start+self.chunk);g=self.gradients(self.bary[q]);v=self.weights[q]
            G=t.einsum('eai,qeaj->qeij',ue,g);F=self.I+G;J=t.linalg.det(F);Js.append(J.reshape(-1))
            if not bool(t.isfinite(J).all()) or float(J.min())<=m.minimum_J:raise ValueError('Quadrature Jacobian guard')
            tr=t.diagonal(G,dim1=-2,dim2=-1).sum(-1);second=.5*(tr*tr-t.einsum('qeij,qeji->qe',G,G));detG=t.linalg.det(G);jm=tr+second+detG;log=t.log1p(jm);invT=t.linalg.inv(F).transpose(-1,-2)
            small=sum((-1.)**k*jm**k/k for k in range(2,10));xm=t.where(abs(jm)<1e-3,small,jm-log)
            W=mu[q]*(.5*(G*G).sum((-1,-2))-second-detG+xm)+.5*lam[q]*log*log
            P=mu[q][...,None,None]*G+(lam[q]*log-mu[q])[...,None,None]*(invT-self.I)+(lam[q]*log)[...,None,None]*self.I
            energy+=(v*W).sum();ge+=t.einsum('qe,qeiJ,qeaJ->eai',v,P,g)
        J=t.cat(Js)
        if not bool(t.isfinite(J).all()) or float(J.min())<=m.minimum_J:raise ValueError('Nodal Jacobian guard')
        grad=t.zeros_like(self.rest);grad.index_add_(0,self.elements.reshape(-1),ge.reshape(-1,3));H=None
        if tangent:
            values=np.empty((len(ue),27,3,27,3),dtype=np.float64)
            for first in range(0,len(ue),8192):
                sl=slice(first,first+8192);he=t.zeros((len(ue[sl]),27,3,27,3),dtype=t.float64,device=self.device)
                for q0 in range(0,len(self.bary),4):
                    q=slice(q0,q0+4);g=self.gradients(self.bary[q],self.linear[sl]);v=self.weights[q,sl]
                    G=t.einsum('eai,qeaj->qeij',ue[sl],g);F=self.I+G;invT=t.linalg.inv(F).transpose(-1,-2);log=t.log(t.linalg.det(F));D=t.einsum('qeiJ,qeaJ->qeai',invT,g)
                    he+=t.einsum('qe,qeaJ,qebJ,ik->eaibk',v*mu[q,sl],g,g,self.I)
                    he+=t.einsum('qe,qeak,qebi->eaibk',v*(mu[q,sl]-lam[q,sl]*log),D,D)
                    he+=t.einsum('qe,qeai,qebk->eaibk',v*lam[q,sl],D,D)
                values[sl]=he.cpu().numpy()
            H=sparse.coo_matrix((values.ravel(),(m.rows,m.cols)),shape=(m.rest.size,m.rest.size)).tocsr()
        return float(energy),grad.cpu().numpy(),H,J.cpu().numpy()

class MaterialPositionFixture(PositionFixture):
    def __init__(self,*args,material_gradient=0.,**kwargs):
        super().__init__(*args,**kwargs);self.material_gradient=float(material_gradient)
        self.engine=MaterialFieldGPUHex(self,kwargs.get('device','cuda:0'))
