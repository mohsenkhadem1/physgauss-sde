import json
import numpy as np
import pytest
from experiments import isaac_block_gravity_suite as s


def test_rectangular_domain_mass_positive_tets_shared_topology_and_boundaries():
 for c in s.cases():
  X,T,F,ids=s.geometry_and_fixed(c);mass,w,det=s.mass_and_reference(X,T)
  assert X.shape==(325,3) and T.shape==(1152,4)
  np.testing.assert_allclose(np.ptp(X,axis=0),[.24,.08,.08],atol=1e-16)
  assert det.min()>0;assert mass.sum()==pytest.approx(.96)
  assert w.sum()==pytest.approx(1)
  allfaces=np.sort(np.concatenate([T[:,[1,2,3]],T[:,[0,2,3]],T[:,[0,1,3]],T[:,[0,1,2]]]),axis=1)
  values,count=np.unique(allfaces,axis=0,return_counts=True);assert set(count)=={1,2}
  assert set(map(tuple,values[count==1]))==set(map(tuple,np.sort(F,axis=1)))
  local=X-X.min(0)
  if c['boundary']=='left':assert len(ids)==25 and np.max(abs(local[ids,0]))<1e-12
  elif c['boundary']=='right':assert len(ids)==25 and np.max(abs(local[ids,0]-.24))<1e-12
  elif c['boundary']=='bothends':assert len(ids)==50
  elif c['boundary']=='bottom':assert len(ids)==65
  elif c['boundary']=='two_bottom_patches':
   assert len(ids)==40;assert np.all((local[ids,0]<=.06+1e-12)|(local[ids,0]>=.18-1e-12));assert np.max(abs(local[ids,2]))<1e-12
  else:assert len(ids)==0
  assert X[:,2].min()==pytest.approx(.05 if c['kind']=='drop' else .3)


def test_exact_eight_cases_no_floor_in_freefall_and_two_drop_materials():
 c=s.cases();assert len(c)==8 and len({r['id'] for r in c})==8
 assert [(r['young_Pa'],r['duration_s']) for r in c if r['kind']=='drop']==[(300000.,.6),(600000.,.6)]
 free=next(r for r in c if r['kind']=='freefall');assert free['duration_s']==.2 and free['boundary']=='none'
 for r in c:
  assert round(r['duration_s']*240)%8==0


def test_native_flags_distinguish_disabled_targets_from_inverse_mass():
 _,_,_,fixed=s.geometry_and_fixed(s.cases()[0]);N=325;e=np.zeros((1,N,1),bool);e[0,fixed]=True;r=np.zeros((1,N,4));r[0,:,3]=(~e[0,:,0]).astype(float)
 assert s.check_flags(e,r,N,fixed)['enabled_fixed_count']==25
 r[0,fixed[0],3]=1
 with pytest.raises(AssertionError):s.check_flags(e,r,N,fixed)


def test_native_rest_tets_may_reorder_but_not_change_oriented_elements():
 X,T,_,_=s.geometry_and_fixed(s.cases()[0]);s.native_geometry(X[None].astype(np.float32),T[::-1][None],X,T)
 wrong=T.copy();wrong[0,[0,1]]=wrong[0,[1,0]]
 with pytest.raises(AssertionError):s.native_geometry(X[None].astype(np.float32),wrong[None],X,T)


def test_ballistic_actual_float32_clock_and_two_position_conventions():
 dt=np.full(48,float(np.float32(1/240)));g=np.array([0,0,-9.81]);v,exact,kick=s.ballistic_reference(g,dt);t=dt.cumsum()
 np.testing.assert_allclose(v,t[:,None]*g,atol=2e-15)
 np.testing.assert_allclose(exact,.5*t[:,None]**2*g,atol=3e-16)
 np.testing.assert_allclose(kick-exact,.5*np.cumsum(dt**2)[:,None]*g,atol=3e-16)
 with pytest.raises(ValueError):s.ballistic_reference(g,[0])


def test_frozen_protocol_hash_and_source_change_guard(tmp_path,monkeypatch):
 monkeypatch.setattr(s,'ROOT',tmp_path)
 for name in s.SOURCE_NAMES:
  p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('frozen source\n')
 r=s.prepare();p=s.validate_protocol(r['path'],r['sha256']);assert p['global_wall_cap_s']==600 and p['app_wall_cap_s']==120
 assert p['gpu']==0 and p['CUDA_VISIBLE_DEVICES']=='0'
 (tmp_path/s.SOURCE_NAMES[0]).write_text('changed\n')
 with pytest.raises(ValueError,match='source'):s.validate_protocol(r['path'],r['sha256'])
