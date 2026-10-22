"""CPU-only manufactured parity; never loads any campaign responses."""
import unittest
from pathlib import Path
import warp as wp
wp.config.kernel_cache_dir=str(Path(__file__).resolve().parents[1]/"runs/operator-dev/p2-warp-rhs-v01/warp-cache")
import numpy as np
from numpy.polynomial.legendre import leggauss
from experiments.campaign_p2_warp_rhs import WarpP2Force
EDGES=((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))
def shape_gradients(b,g):
 return np.concatenate([(4*b[...,None]-1)*g,np.stack([4*(b[...,i,None]*g[...,j,:]+b[...,j,None]*g[...,i,:]) for i,j in EDGES],-2)],-2)
def duffy(order):
 z,w=leggauss(order);z=(z+1)/2;w=w/2;b=[];weights=[]
 for i,a in enumerate(z):
  for j,c in enumerate(z):
   for k,d in enumerate(z):
    r=a;s=(1-a)*c;t=(1-a)*(1-c)*d;b.append([1-r-s-t,r,s,t]);weights.append(6*w[i]*w[j]*w[k]*(1-a)**2*(1-c))
 return np.asarray(b),np.asarray(weights)
def stable_material(F,E,nu):
 mu=E/(2*(1+nu));kap=E*nu/((1+nu)*(1-2*nu))+mu;J=np.linalg.det(F);cof=J[...,None,None]*np.linalg.inv(F).swapaxes(-1,-2)
 W=.5*mu*((F*F).sum((-1,-2))-3)-mu*(J-1)+.5*kap*(J-1)**2;P=mu*F+(kap*(J-1)-mu)[...,None,None]*cof
 A=np.empty(F.shape[:-2]+(3,3,3,3))
 for k in range(3):
  for l in range(3):
   dF=np.zeros_like(F);dF[...,k,l]=1
   dc=np.stack([np.cross(dF[..., :,1],F[..., :,2])+np.cross(F[..., :,1],dF[..., :,2]),np.cross(dF[..., :,2],F[..., :,0])+np.cross(F[..., :,2],dF[..., :,0]),np.cross(dF[..., :,0],F[..., :,1])+np.cross(F[..., :,0],dF[..., :,1])],-1)
   A[..., :, :,k,l]=mu*dF+kap*cof[...,k,l,None,None]*cof+(kap*(J-1)-mu)[...,None,None]*dc
 return W,P,A

def fixture():
 X=np.array([[0.,0.,.1],[.2,0.,.1],[0.,.2,.1],[0.,0.,.3]])
 X=np.r_[X,np.array([X[list(e)].mean(0) for e in EDGES])];elements=np.arange(10)[None];lg=np.array([[[-5.,-5.,-5.],[5.,0.,0.],[0.,5.,0.],[0.,0.,5.]]]);b,w=duffy(3)
 grad=np.stack([shape_gradients(bb[None],lg) for bb in b]);check=np.stack([shape_gradients(bb[None],lg) for bb in np.r_[np.eye(4),np.array([(np.eye(4)[i]+np.eye(4)[j])/2 for i,j in EDGES])]])
 z,ww=leggauss(8);z=(z+1)/2;ww=ww/2;r,s=np.meshgrid(z,z,indexing='ij');wa,wb=np.meshgrid(ww,ww,indexing='ij');r=r.ravel();ss=((1-r.reshape(8,8))*s).ravel();bary=np.c_[1-r-ss,r,ss];N=np.c_[bary*(2*bary-1),4*bary[:,0]*bary[:,1],4*bary[:,0]*bary[:,2],4*bary[:,1]*bary[:,2]]
 return dict(rest=X,elements=elements,grad=grad,check_grad=check,weights=w[:,None]*(.2**3/6),surface_elements=np.array([[0,1,2,4,5,7]]),surface_shapes=N,surface_weights=(2*wa*wb*(1-r.reshape(8,8))).reshape(-1,1)*.02)

def reference(a,u,floor=.102):
 F=np.eye(3)+np.einsum('eai,qeaj->qeij',u[a['elements']],a['grad']);W,P,A=stable_material(F,5000.,.35);g=np.einsum('qe,qeij,qeaj->eai',a['weights'],P,a['grad'])[0];U=float((a['weights']*W).sum());H=np.einsum('qe,qeijkl,qeaj,qebl->eaibk',a['weights'],A,a['grad'],a['grad'])[0].reshape(30,30)
 f=a['surface_elements'];N=a['surface_shapes'];gap=N@(a['rest']+u)[f[0],2]-floor;neg=np.minimum(gap,0);sw=a['surface_weights'][:,0];U+=.5*1e4*np.dot(sw,neg**2);np.add.at(g[:,2],f[0],1e4*N.T@(sw*neg));Hs=1e4*np.einsum('q,qa,qb->ab',sw*(gap<0),N,N);dof=3*f[0]+2;H[np.ix_(dof,dof)]+=Hs
 return U,g,H,float(min(np.linalg.det(F).min(),np.linalg.det(np.eye(3)+np.einsum('eai,qeaj->qeij',u[a['elements']],a['check_grad'])).min()))

class ManufacturedWarpRHS(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.a=fixture();cls.obj=WarpP2Force(cls.a,young=5000.,poisson=.35,kappa_A=1e4,floor_z=.102,device='cpu')
 def test_energy_force_J_parity_and_rigid_translation(self):
  rng=np.random.default_rng(8)
  for u in [np.zeros((10,3)),np.tile([.03,-.02,.04],(10,1)),rng.normal(size=(10,3))*3e-4,rng.normal(size=(10,3))*.003]:
   actual=self.obj.evaluate(u);U,g,H,j=reference(self.a,u)
   self.assertLess(abs(actual['energy']-U),1e-12);np.testing.assert_allclose(actual['gradient'],g,atol=1e-10,rtol=1e-13);self.assertLess(abs(actual['minimum_J']-j),2e-14)
 def test_energy_gradient_and_force_jacobian_finite_differences(self):
  rng=np.random.default_rng(9);u=rng.normal(size=(10,3))*1e-4;d=rng.normal(size=(10,3));d/=np.linalg.norm(d);eps=1e-7
  U,g,H,j=reference(self.a,u);plus=self.obj.evaluate(u+eps*d);minus=self.obj.evaluate(u-eps*d)
  self.assertLess(abs((plus['energy']-minus['energy'])/(2*eps)-np.sum(g*d)),1e-7)
  np.testing.assert_allclose((plus['gradient']-minus['gradient'])/(2*eps),(H@d.ravel()).reshape(10,3),atol=2e-5,rtol=1e-7)
 def test_all_guard_points_and_fp64_contract(self):
  with self.assertRaises(ValueError):self.obj.evaluate(np.zeros((10,3),dtype=np.float32))
  u=np.zeros((10,3));u[:,0]=-2*self.a['rest'][:,0]
  with self.assertRaisesRegex(ValueError,'Jacobian'):self.obj.evaluate(u)
  u=np.zeros((10,3));u[4,0]=.047
  F=np.eye(3)+np.einsum('eai,qeaj->qeij',u[self.a['elements']],self.a['grad'])
  self.assertGreater(np.linalg.det(F).min(),.1)
  Fc=np.eye(3)+np.einsum('eai,qeaj->qeij',u[self.a['elements']],self.a['check_grad'])
  self.assertLess(np.linalg.det(Fc).min(),.1)
  with self.assertRaisesRegex(ValueError,'Jacobian'):self.obj.evaluate(u)

if __name__=='__main__':unittest.main()
