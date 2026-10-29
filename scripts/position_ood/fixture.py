"""Position-driven nonlinear solid with material- and world-frame interactions.

The controller target positions are inputs. Reaction forces are outputs and must
not be exposed to predictors in the primary unknown-environment benchmark.
"""
from pathlib import Path
import sys
import numpy as np
from scipy import sparse
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.phystwin.v08_reference import StrongFixture
from scripts.phystwin.sequential_fixture import points

class PositionFixture(StrongFixture):
    def __init__(self,n,E,Kleft,Kright,layout,device='cuda:0',order=3,
                 gravity=(0,0,-9.81),density=625.,field_amplitude_J=0.,
                 field_center=(.12,0,-.09),field_width=.08):
        super().__init__(n,E,Kleft,Kright,layout,device=device,order=order)
        self.gravity=np.asarray(gravity,dtype=float);self.density=float(density)
        if self.gravity.shape!=(3,) or not np.isfinite(self.gravity).all():raise ValueError('finite gravity vector required')
        self.total_mass=self.density*float(self.volumes.sum())
        self.field_amplitude_J=float(field_amplitude_J);self.field_center=np.asarray(field_center,dtype=float);self.field_width=float(field_width)
        if self.field_width<=0 or self.field_center.shape!=(3,):raise ValueError('invalid spatial field')
        kinds={'parallel_lower':('lower','lower'),'crossed':('lower','upper'),'opposite_y':('yminus','yplus')}[layout]
        self.anchor_matrices=[]
        for left,K,kind in zip([True,False],[Kleft,Kright],kinds):
            M,_,area=self.patch(left,kind);self.anchor_matrices.append(K*M/area)
        self.set_controls(np.zeros((2,6)))
    def set_action(self,action):
        # Parent construction calls this before positional controllers exist.
        self.load=np.zeros_like(self.rest)
    def set_controls(self,poses):
        """Two [translation xyz (m), rotation-vector xyz (rad)] tool poses.

        Rotations are about end-face centers in the rest configuration. Each
        surface quadrature point follows the prescribed rigid tool transform.
        """
        p=np.asarray(poses,dtype=float)
        if p.shape!=(2,6) or not np.isfinite(p).all():raise ValueError('two finite 6D tool poses required')
        self.poses=p.copy();self.anchor_targets=[];self.target_constant=0.
        self.load=self.volume_weights[:,None]*self.total_mass*self.gravity
        for i,M in enumerate(self.anchor_matrices):
            center=np.array([0. if i==0 else .24,0.,0.]);R=Rotation.from_rotvec(p[i,3:]).as_matrix()
            d=(self.rest-center)@R.T+center+p[i,:3]-self.rest
            self.anchor_targets.append(d);self.load+=M@d;self.target_constant+=.5*float(np.sum(d*(M@d)))
        self.clear_linear_cache()
    def spatial_field(self,x,tangent=False):
        delta=x-self.field_center;s2=self.field_width**2
        value=self.field_amplitude_J*self.volume_weights*np.exp(-np.sum(delta*delta,axis=1)/(2*s2))
        grad=-value[:,None]*delta/s2
        H=None
        if tangent:
            blocks=value[:,None,None]*(delta[:,:,None]*delta[:,None,:]/s2**2-np.eye(3)[None]/s2)
            ids=np.arange(x.size).reshape(-1,3);rows=np.broadcast_to(ids[:,:,None],blocks.shape);cols=np.broadcast_to(ids[:,None,:],blocks.shape)
            H=sparse.coo_matrix((blocks.ravel(),(rows.ravel(),cols.ravel())),shape=(x.size,x.size)).tocsr()
        return float(value.sum()),grad,H
    def evaluate(self,u,tangent=False):
        state=super().evaluate(u,tangent)
        if not hasattr(self,'field_amplitude_J'):return state
        energy,grad,H=self.spatial_field(self.rest+u,tangent)
        state['potential_J']+=self.target_constant+energy;state['spring_J']=sum(.5*float(np.sum((u-d)*(M@(u-d)))) for M,d in zip(self.anchor_matrices,self.anchor_targets))
        state['spatial_potential_J']=energy;state['residual']+=grad
        if tangent:state['tangent']+=H
        return state
    def sampled(self,state):
        u=state['displacement_local_m'];x=points();g=self.query_field(u,x,True)
        reactions=np.array([(M@(u-d)).sum(0) for M,d in zip(self.anchor_matrices,self.anchor_targets)])
        _,field_grad,_=self.spatial_field(self.rest+u)
        body=self.total_mass*self.gravity
        return dict(field=self.query_field(u,x),strain=(g+g.transpose(0,2,1))/2,
                    residual_N=float(np.linalg.norm(state['residual'])),minimum_J=float(state['J'].min()),
                    reaction_forces_N=reactions,body_force_N=body,spatial_force_N=-field_grad.sum(0),
                    reaction_balance_N=float(np.linalg.norm(reactions.sum(0)-body+field_grad.sum(0))),
                    elastic_J=state['elastic_J'],spring_J=state['spring_J'])

def controls(name,amplitude=.006):
    p=np.zeros((2,6))
    if name=='rest':pass
    elif name=='translate_y':p[1,1]=amplitude
    elif name=='translate_z':p[1,2]=amplitude
    elif name=='stretch':p[0,0]=-amplitude/2;p[1,0]=amplitude/2
    elif name=='opposed_y':p[0,1]=-amplitude;p[1,1]=amplitude
    elif name=='twist_x':p[0,3]=-.15;p[1,3]=.15
    elif name=='bend_z':p[1,5]=.15
    elif name=='diagonal':p[1,:3]=amplitude*np.array([.5,-1.,.5])
    else:raise ValueError(name)
    return p
