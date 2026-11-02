import numpy as np
import pytest
import torch
from scripts.compositional_neural_mechanics.inference import implicit_gradient

def test_implicit_direct_term_and_no_newton_graph():
 z=torch.tensor([.4,-.2],dtype=torch.float64);u=torch.tensor([.4*np.exp(-.2)],dtype=torch.float64)
 residual=lambda u,z:u*torch.exp(-z[1])-z[0]
 # Exact equilibrium is z0*exp(z1).
 objective=lambda u,z:.5*(u[0]-.1)**2+.7*z[1]**2
 result=implicit_gradient(u,z,residual,objective)
 expected=torch.tensor([(u[0]-.1)*torch.exp(z[1]),(u[0]-.1)*u[0]+1.4*z[1]],dtype=z.dtype)
 torch.testing.assert_close(result['gradient'],expected)
 assert not result['gradient'].requires_grad
 with pytest.raises(ValueError,match='tangent disagrees'):implicit_gradient(u,z,residual,objective,tangent=np.eye(1)*99)
 with pytest.raises(ValueError,match='Equilibrium'):implicit_gradient(u+1,z,residual,objective)

def test_fixed_dof_and_singular_adjoint_rejection():
 z=torch.tensor([.2],dtype=torch.float64);u=torch.tensor([0.,.2],dtype=z.dtype)
 result=implicit_gradient(u,z,lambda u,z:torch.stack([u[0],u[1]-z[0]]),lambda u,z:u.square().sum(),free_dofs=[1])
 torch.testing.assert_close(result['gradient'],torch.tensor([.4],dtype=z.dtype))
 with pytest.raises(np.linalg.LinAlgError):implicit_gradient(u,z,lambda u,z:torch.zeros_like(u)+z*0,lambda u,z:u.sum())

@pytest.mark.parametrize('kind',['log_modulus_support','shared_material_latent'])
def test_q2_equilibrium_adjoint_matches_centered_refits(kind):
 from scripts.compositional_neural_mechanics.mechanics import Q2Mechanics,GeometrySpec,AttachmentSpec
 from scripts.compositional_neural_mechanics.materials import SharedMaterialLaw
 from scripts.compositional_neural_mechanics.inference import observation_matrix,observe_displacement,linear_reaction_wrenches
 torch.set_num_threads(1);torch.manual_seed(81)
 law=SharedMaterialLaw(latent_dim=2).double()
 with torch.no_grad():law.coefficient_net[-1].weight.normal_(0,.15)
 for parameter in law.parameters():parameter.requires_grad_(False)
 theta=np.array([np.log(12000.),np.log(6000.)])if kind=='log_modulus_support'else np.array([.2,-.3])
 pose=np.zeros((2,6));pose[1,0]=.0015;pose[1,5]=.025
 coords=np.array([[.2,.3,.4],[.6,.7,.3],[.85,.4,.8]])
 def material_field(q,z):
  if kind=='log_modulus_support':return z[0].expand(*q.shape[:-1],1)
  return torch.stack([z[0]+z[1]*(q[...,0]-.5),z[1].expand(q.shape[:-1])],-1)
 def stiffness(z):return torch.stack([z.new_tensor(6000.),z[1].exp()])if kind=='log_modulus_support'else z.new_tensor([6000.,6000.])
 def build(v):
  z=torch.tensor(v,dtype=torch.float64);K=stiffness(z)
  model=Q2Mechanics(GeometrySpec(taper=.1,arch=.004),n=1,bulk_energy=law.energy if kind=='shared_material_latent'else None,material_latent_field=lambda q:material_field(q,z),attachments=tuple(AttachmentSpec(float(k))for k in K))
  model.set_controls(pose);return model,z
 model,z=build(theta);state=model.solve(force_tolerance_N=1e-10);M=observation_matrix(model,coords)
 def loss(u,z):
  position=observe_displacement(u,M);forces,torques=linear_reaction_wrenches(model,u,stiffness(z))
  return (position/.003).square().mean()+.2*((forces[1]-u.new_tensor([-.1,0.,0.]))/.2).square().mean()+.1*(torques[1]/.02).square().mean()
 def residual(u,z):
  model.material_latent_field=lambda q:material_field(q,z)
  return model.torch_residual(u,stiffness(z))
 H=model.evaluate(state['displacement_local_m'],True)['tangent']
 result=implicit_gradient(state['displacement_local_m'],z,residual,loss,tangent=H,equilibrium_tolerance=1e-8)
 values=[];h=2e-4
 for j in range(len(theta)):
  pair=[]
  for sign in (-1,1):
   v=theta.copy();v[j]+=sign*h;other,other_z=build(v);solved=other.solve(force_tolerance_N=1e-10)
   # Identical geometry/pose and observation map; only declared parameters change.
   pair.append(float(loss(torch.tensor(solved['displacement_local_m'],dtype=torch.float64),other_z)))
  values.append((pair[1]-pair[0])/(2*h))
 print(kind,dict(implicit=result['gradient'].tolist(),centered_refit=values,adjoint_relative_residual=result['adjoint_relative_residual'],tangent_check=result['tangent_direction_relative_error']))
 np.testing.assert_allclose(result['gradient'].numpy(),values,rtol=5e-4,atol=2e-6)
 np.testing.assert_allclose(observe_displacement(torch.tensor(state['displacement_local_m']),M).numpy(),model.sample_material(state['displacement_local_m'],coords),atol=1e-14)
 # Optional wrench direct dependence matters; the logK partial loss is not dropped.
 if kind=='log_modulus_support':assert abs(result['direct_gradient'][1])>1e-5

def test_q2_attachment_center_width_implicit_gradient():
 from scripts.compositional_neural_mechanics.mechanics import Q2Mechanics
 torch.set_num_threads(1)
 pose=np.zeros((2,6));pose[1,5]=.035;pose[1,0]=.001
 theta=np.array([.3,np.log(.32)])
 def geometry(z):
  left=z.new_tensor([.5,.1875,np.log(.375),np.log(.375)])
  right=torch.stack([z.new_tensor(.5),z[0],z.new_tensor(np.log(.375)),z[1]])
  return torch.stack([left,right])
 def energy(delta,z):return .5*torch.exp(z[...,0])*delta.square().sum(-1)
 def field(q):return torch.full((*q.shape[:-1],1),np.log(6000.),dtype=q.dtype,device=q.device)
 def build(v):
  z=torch.tensor(v,dtype=torch.float64)
  model=Q2Mechanics(n=1,interface_energy=energy,interface_latent_fields=(field,field),interface_geometry_parameters=lambda:geometry(z))
  model.set_controls(pose);return model,z
 model,z=build(theta);state=model.solve(force_tolerance_N=1e-10)
 def set_parameter(z):model.interface_geometry_parameters=lambda:geometry(z)
 def residual(u,z):
  set_parameter(z);return model.torch_residual(u)
 def objective(u,z):
  set_parameter(z);_,_,_,(forces,torques)=model.interface_response(u,create_graph=True)
  return (u/.003).square().mean()+.2*(forces[1]/.2).square().mean()+.1*(torques[1]/.02).square().mean()
 H=model.evaluate(state['displacement_local_m'],True)['tangent']
 result=implicit_gradient(state['displacement_local_m'],z,residual,objective,tangent=H,equilibrium_tolerance=1e-8)
 finite=[];h=1e-4
 for j in range(2):
  pair=[]
  for sign in (-1,1):
   v=theta.copy();v[j]+=sign*h;other,zz=build(v);s=other.solve(force_tolerance_N=1e-10)
   pair.append(float(objective(torch.tensor(s['displacement_local_m'],dtype=torch.float64,requires_grad=True),zz)))
  finite.append((pair[1]-pair[0])/(2*h))
 print('attachment_center_logwidth',dict(implicit=result['gradient'].tolist(),centered_refit=finite,adjoint_relative_residual=result['adjoint_relative_residual']))
 np.testing.assert_allclose(result['gradient'].numpy(),finite,rtol=5e-4,atol=2e-6)
