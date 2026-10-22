from types import SimpleNamespace
import numpy as np
import pytest
from experiments.campaign_newton_static_initialization import accepted_start_solver_class

class Array:
 def __init__(self,x):self.x=np.asarray(x,dtype=float).copy()
 def zero_(self):self.x[:]=0
class Warp:
 @staticmethod
 def copy(a,b):a.x[:]=b.x
class Base:
 def __init__(self):self.model=SimpleNamespace(body_count=0);self.particle_enable_self_contact=False
 def _initialize_particles(self,state_in,state_out,contacts,dt):
  self.particle_q_prev=Array(state_in.particle_q.x);self.pos_prev_collision_detection=Array(state_in.particle_q.x)
  self.inertia=Array(state_in.particle_q.x+dt**2*np.array([[16.,0,0],[0,0,0]]))
  self.particle_displacements=Array(self.inertia.x-state_in.particle_q.x);state_in.particle_q.x[:]=self.inertia.x


def test_initial_guess_only_objective_inertia_preserved():
 old=np.array([[.24,0,0],[0,0,0]]);s=SimpleNamespace(particle_q=Array(old));solver=accepted_start_solver_class(Base,Warp)()
 solver._initialize_particles(s,None,None,1.)
 np.testing.assert_array_equal(s.particle_q.x,old);np.testing.assert_array_equal(solver.particle_q_prev.x,old)
 np.testing.assert_array_equal(solver.inertia.x,old+np.array([[16.,0,0],[0,0,0]]))
 np.testing.assert_array_equal(solver.pos_prev_collision_detection.x,old);assert np.max(abs(solver.particle_displacements.x))==0
 # The exact original inertial force at arbitrary candidate is unaltered.
 q=old+.01;np.testing.assert_array_equal(solver.inertia.x-q,old+np.array([[16.,0,0],[0,0,0]])-q)


def test_undeclared_contact_or_rigid_scope_rejected():
 cls=accepted_start_solver_class(Base,Warp);s=SimpleNamespace(particle_q=Array([[0,0,0]]))
 a=cls();a.model.body_count=1
 with pytest.raises(ValueError):a._initialize_particles(s,None,None,1.)
 a=cls();a.particle_enable_self_contact=True
 with pytest.raises(ValueError):a._initialize_particles(s,None,None,1.)
