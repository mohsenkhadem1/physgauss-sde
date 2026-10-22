"""Exact P1 consistent left-face spring variant; v01 lumped source unchanged."""
import numpy as np
from scipy import sparse
from experiments.campaign_static_uniaxial_bar import PrismaticBar,face_area_weights,SIZE


class ConsistentPrismaticBar(PrismaticBar):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        _,triangles=face_area_weights(self.reference_local,self.faces,0,0.)
        corners=self.reference_local[triangles]
        areas=.5*np.linalg.norm(np.cross(corners[:,1]-corners[:,0],corners[:,2]-corners[:,0]),axis=1)
        # Exact integral of P1 NiNj over each reference triangle. Only local-x
        # displacements couple: lateral fixture stiffness remains exactly zero.
        local=self.spring/(SIZE[1]*SIZE[2])*areas[:,None,None]/12*(np.ones((3,3))+np.eye(3))
        dofs=3*triangles
        rows=np.repeat(dofs,3,axis=1).ravel();cols=np.tile(dofs,(1,3)).ravel()
        self.spring_matrix=sparse.coo_matrix((local.ravel(),(rows,cols)),shape=(self.rest.size,self.rest.size)).tocsr()
    def evaluate(self,u,tangent=False):
        u=np.asarray(u,dtype=float)
        if u.shape!=self.rest.shape or not np.isfinite(u).all():raise ValueError('Finite local displacement field required')
        positions=self.rest+u@self.rotation.T;elastic,grad,H,J=self.elastic(positions,tangent)
        spring_force=(self.spring_matrix@u.ravel()).reshape(u.shape)
        spring_energy=.5*float(np.sum(u*spring_force));work=float(np.sum(self.load*u))
        return dict(potential_J=elastic+spring_energy-work,elastic_J=elastic,spring_J=spring_energy,work_J=work,
            residual=grad@self.rotation+spring_force-self.load,
            tangent=None if H is None else self.T.T@H@self.T+self.spring_matrix,J=J,positions=positions)
