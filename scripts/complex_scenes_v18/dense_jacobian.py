"""Independent dense element-coordinate orientation audit (sampled, not a proof)."""
import numpy as np
from scripts.phystwin.tensor_fixture import basis


class DenseJacobianAudit:
    def __init__(self,model,samples=9):
        if samples<5 or samples%2!=1:raise ValueError('Odd dense grid with at least five points per axis')
        self.elements=model.elements;self.blocks=[];q=np.stack(np.meshgrid(*[np.linspace(-1,1,samples)]*3,indexing='ij'),-1).reshape(-1,3);_,G=basis(q)
        for start in range(0,len(q),32):
            g=G[start:start+32];jac=np.einsum('eai,qaj->qeij',model.rest[model.elements],g);self.blocks.append((g,np.linalg.inv(jac)))
        self.sample_count=len(q)*len(model.elements)

    def minimum(self,u):
        ue=u[self.elements];value=float('inf')
        for g,inv in self.blocks:
            local=np.einsum('eai,qaj->qeij',ue,g);F=np.eye(3)+local@inv;J=np.linalg.det(F)
            if not np.isfinite(J).all():raise FloatingPointError('Nonfinite dense orientation sample')
            value=min(value,float(J.min()))
        return value
