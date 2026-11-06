"""Exact reference binding on the structured tetrahedral volume.

Unlike nearest-centroid association, points inside the block use their actual
containing tetrahedron. Outside appearance splats use explicit affine extension.
"""
import itertools
import numpy as np
import torch

class DenseBinding:
    def __init__(self,model,centers,covariance):
        x=np.asarray(centers);dims=np.asarray(model.dimensions);cells=np.asarray(model.cells)
        q=(x+np.r_[0,dims[1:]/2])/dims*cells
        cell=np.clip(np.floor(q).astype(int),0,cells-1);frac=q-cell
        perm=np.argsort(-np.clip(frac,0,1),axis=1,kind='stable')
        perms=list(itertools.permutations(range(3)));lookup={p:i for i,p in enumerate(perms)}
        cube=np.ravel_multi_index(cell.T,tuple(cells));indices=cube*6+np.array([lookup[tuple(p)] for p in perm])
        self.ids=model.elements[model.t(indices).long()]
        self.grad=model.grad[model.t(indices).long()]
        vertices=model.rest[self.ids];dm=(vertices[:,1:]-vertices[:,:1]).transpose(-1,-2)
        bary=torch.linalg.solve(dm,(model.t(x)-vertices[:,0])[...,None]).squeeze(-1)
        self.N=torch.cat([1-bary.sum(-1,keepdim=True),bary],-1)
        self.cov=model.t(covariance)
        self.rest=model.rest
        self.centers=model.t(centers)
        self.extrapolated_fraction=float(((self.N< -1e-9)|(self.N>1+1e-9)).any(-1).float().mean())
    def transport(self,nodes):
        dx=(nodes-self.rest)[self.ids];F=torch.eye(3,device=nodes.device,dtype=nodes.dtype)+torch.einsum('pai,paj->pij',dx,self.grad);J=torch.linalg.det(F)
        if not bool(torch.isfinite(J).all()) or float(J.detach().min())<=0:raise ValueError('Invalid material transport')
        R=F
        for _ in range(12):R=.5*(R+torch.linalg.inv(R).transpose(-1,-2))
        if not torch.allclose(R.transpose(-1,-2)@R,torch.eye(3,device=R.device,dtype=R.dtype).expand_as(R),atol=1e-5,rtol=1e-5):raise ValueError('Polar rotation did not converge')
        return dict(centers=self.centers+torch.einsum('pa,pai->pi',self.N,dx),covariance=F@self.cov@F.transpose(-1,-2),rotation=R,determinant=J)
