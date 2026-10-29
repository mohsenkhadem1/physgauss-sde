"""Finite smooth attachment footprints for boundary-condition diagnostics.

A Gaussian surface footprint tapers to zero at the end-face perimeter. This
removes abrupt idealized traction transitions without altering the material law.
Total attachment stiffness retains units N/m and the same normalization.
"""
import numpy as np
from scipy import sparse
from numpy.polynomial.legendre import leggauss
from scripts.position_ood.material_field import MaterialPositionFixture
from scripts.position_ood.differentiable import DensePositionFEM
from scripts.phystwin.tensor_fixture import basis

def footprint(y,z,kind):
    cy,cz={'lower':(0.,-.025),'upper':(0.,.025),'yminus':(-.025,0.),'yplus':(.025,0.)}[kind]
    return np.maximum(0,1-(y/.04)**2)*np.maximum(0,1-(z/.04)**2)*np.exp(-((y-cy)**2+(z-cz)**2)/(2*.03**2))

class SmoothMaterialFixture(MaterialPositionFixture):
    def patch(self,left,kind):
        if kind=='full':return super().patch(left,kind)
        q,w=leggauss(6);l=np.stack([q*(q-1)/2,1-q*q,q*(q+1)/2],axis=1);N=np.einsum('ia,jb->ijab',l,l).reshape(36,9)
        cell=np.stack(np.meshgrid(np.arange(self.cells[1]),np.arange(self.cells[2]),indexing='ij'),-1).reshape(-1,2);offset=np.stack(np.meshgrid(np.arange(3),np.arange(3),indexing='ij'),-1).reshape(-1,2);yz=2*cell[:,None]+offset;ix=np.full(yz.shape[:2],0 if left else self.grid_shape[0]-1);nodes=np.ravel_multi_index((ix,yz[:,:,0],yz[:,:,1]),self.grid_shape)
        ya=self.axes[1][cell[:,0]];za=self.axes[2][cell[:,1]];dy=np.diff(self.axes[1])[cell[:,0]];dz=np.diff(self.axes[2])[cell[:,1]];yy=ya[:,None]+(q+1)[None]*dy[:,None]/2;zz=za[:,None]+(q+1)[None]*dz[:,None]/2
        weights=(np.outer(w,w)[None]*dy[:,None,None]*dz[:,None,None]/4*footprint(yy[:,:,None],zz[:,None,:],kind)).reshape(len(cell),36);local=np.einsum('qa,eq,qb->eab',N,weights,N);integral=weights@N
        rows=np.broadcast_to(nodes[:,:,None],local.shape);cols=np.broadcast_to(nodes[:,None,:],local.shape);M=sparse.coo_matrix((local.ravel(),(rows.ravel(),cols.ravel())),shape=(len(self.rest),len(self.rest))).tocsr();load=np.zeros(len(self.rest));np.add.at(load,nodes,integral)
        return M,load,float(weights.sum())

class SmoothDensePositionFEM(DensePositionFEM):
    def patch(self,left,kind):
        g=self.geo;q,w=leggauss(6);M=np.zeros((len(g.rest),len(g.rest)));area=0.
        for iy in range(g.cells[1]):
            for iz in range(g.cells[2]):
                ya,yb=g.axes[1][iy:iy+2];za,zb=g.axes[2][iz:iz+2];yy=(q+1)*(yb-ya)/2+ya;zz=(q+1)*(zb-za)/2+za;points=np.stack(np.meshgrid(yy,zz,indexing='ij'),-1).reshape(-1,2)
                local=np.c_[np.full(36,-1. if left else 1.),2*(points[:,0]-ya)/(yb-ya)-1,2*(points[:,1]-za)/(zb-za)-1];N,_=basis(local);ww=np.outer(w,w).ravel()*(yb-ya)*(zb-za)/4*footprint(points[:,0],points[:,1],kind);idx=np.ravel_multi_index((0 if left else g.cells[0]-1,iy,iz),tuple(g.cells));ids=g.elements[idx];M[np.ix_(ids,ids)]+=N.T@(ww[:,None]*N);area+=ww.sum()
        return M/area
