import numpy as np
import pytest
from experiments.campaign_newton_vbd_probe_v02 import material,element_reference,block_inputs,actual_input_reference


def test_exact_mass_rightforce_and_leftfixed_geometry():
 X,T,F,m,ids,force=block_inputs();assert len(X)==45 and len(T)==96 and len(ids)==9
 assert m.sum()==pytest.approx(.96);assert (m>0).all()
 np.testing.assert_allclose(force.sum(0),[1,0,0],atol=2e-16)
 np.testing.assert_allclose(np.cross(X,force).sum(0),0,atol=1e-17)
 assert np.all(X[ids,0]==0)


def test_stable_reference_energy_gradient_and_selfblock_fd():
 X,T,_,_,_,_=block_inputs();q=X@np.array([[1.03,.01,0],[0,.99,.005],[0,0,.995]]).T
 r=element_reference(X,T,q);e=2;a=1;n=int(T[e,a]);direction=np.array([.2,-.3,.4]);eps=1e-7
 qp=q.copy();qm=q.copy();qp[n]+=eps*direction;qm[n]-=eps*direction
 plus=element_reference(X,T,qp);minus=element_reference(X,T,qm)
 assert (plus['energy_J']-minus['energy_J'])/(2*eps)==pytest.approx(-np.sum(r['local_force_N'][T==n]*direction),rel=1e-7,abs=1e-9)
 np.testing.assert_allclose(-(plus['local_force_N'][e,a]-minus['local_force_N'][e,a])/(2*eps),r['self_H_N_m'][e,a]@direction,rtol=1e-7,atol=2e-7)


def test_reststress_rigidobjectivity_and_lame_coefficients():
 X,T,_,_,_,_=block_inputs();r=element_reference(X,T,X)
 assert np.max(abs(r['local_force_N']))<1e-12
 mu,lam=material();assert mu==pytest.approx(1851.85185185185);assert lam==pytest.approx(4320.987654320987)
 a=.35;Q=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
 rotated=element_reference(X,T,X@Q.T+[.2,-.1,.05]);assert abs(rotated['energy_J'])<1e-12;assert np.max(abs(rotated['local_force_N']))<1e-12


def test_actual_readback_inputs_reference_and_idealquantization_are_distinct():
 X,T,_,_,_,_=block_inputs();q=(X@np.array([[1.,.02,0],[0,.99,.01],[0,0,1.02]]).T).astype(np.float32).astype(float)
 inv=element_reference(X,T,X)['poses'];mu,lam=material();r=actual_input_reference(q,T,inv,mu,lam);expected=element_reference(X,T,q)
 np.testing.assert_allclose(r['local_force_N'],expected['local_force_N'],atol=1e-14)
 np.testing.assert_allclose(r['self_H_N_m'],expected['self_H_N_m'],atol=1e-12)
 angle=.4;Q=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1.]])
 native=actual_input_reference((X@Q.T).astype(np.float32),T,inv.astype(np.float32),np.float32(mu),np.float32(lam))
 assert abs(native['local_force_N']).max()>1e-8
