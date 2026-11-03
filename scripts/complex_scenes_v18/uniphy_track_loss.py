"""Masked surface-track loss; the author native MPM adjoint consumes its x grads."""
import numpy as np
import taichi as ti


@ti.data_oriented
class TrackedLoss:
    def __init__(self,sim,indices,weights,observed,frame_steps):
        observed=np.asarray(observed,np.float32);self.frames,self.points,_=observed.shape;self.sim=sim
        if indices.shape!=(self.points,8)or weights.shape!=indices.shape or len(frame_steps)!=self.frames:raise ValueError('Matched fixed binding and observation clock required')
        self.indices=ti.field(ti.i32,shape=indices.shape);self.indices.from_numpy(indices)
        self.weights=ti.field(ti.f32,shape=weights.shape);self.weights.from_numpy(weights)
        self.observed=ti.Vector.field(3,ti.f32,shape=observed.shape[:2]);self.observed.from_numpy(observed)
        self.steps=ti.field(ti.i32,shape=self.frames);self.steps.from_numpy(np.asarray(frame_steps,np.int32))

    @ti.kernel
    def compute(self):
        for f,p in ti.ndrange(self.frames-1,self.points):
            x=ti.Vector.zero(ti.f32,3)
            for j in ti.static(range(8)):
                x+=self.weights[p,j]*self.sim.x[self.indices[p,j],self.steps[f+1]]
            self.sim.loss[None]+=(x-self.observed[f+1,p]).norm_sqr()*1e6/((self.frames-1)*self.points)
