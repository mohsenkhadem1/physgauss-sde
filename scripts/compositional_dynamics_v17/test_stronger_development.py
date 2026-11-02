import json,time
import numpy as np
from scripts.compositional_dynamics_v17 import stronger_development as s

def parameters():
 return dict(n=1,dt_s=.0001,volume_order=3,surface_order=8,sample_interval_s=.001,dimensions_m=[.12,.06,.06],young_Pa=12000,poisson=.35,density_kg_m3=625,gravity_m_s2=9.81,penalty_N_m3=4e7,source_sha256={},input_sha256={},scope='synthetic fixture',diagnostic_gates=dict(minimum_J=.1,energy_relative=.01,maximum_penetration_m=.0012))
def striker():return dict(name='finite-striker',family='striker',duration_s=.001,sphere_radius_m=.025,sphere_mass_kg=.08,sphere_center_m=[.06,0,.08],sphere_velocity_m_s=[0,0,-1])
def test_striker_mass_energy_and_external_support_impulse():
 m,e,_=s.build(parameters(),striker());u=np.zeros_like(m.rest);u[:,2]=-.001;a=e.initial_state(u=u);b=e.step(a,.00001)
 assert np.isclose(m.total_mass,.27)
 f0,_=e.total_force_torque(a);f1,_=e.total_force_torque(b)
 np.testing.assert_allclose(e.momentum(b)-e.momentum(a),.000005*(f0+f1),atol=1e-13)
 assert b.diagnostics['momentum_step_balance_Ns']<1e-13
 assert np.isclose(e.energy(e.initial_state()),.04)
def test_actual_sphere_serialization(tmp_path,monkeypatch):
 p=parameters();plan=tmp_path/'plan.json';plan.write_text(json.dumps(p));monkeypatch.setattr(s,'PLAN',plan);monkeypatch.setattr(s,'RAW',tmp_path/'out');monkeypatch.setattr(s,'ROOT',tmp_path)
 row=s.worker((striker(),time.monotonic()+10))
 assert row['completed']and not row['reference_qualified']
 with np.load(tmp_path/row['path']/'trajectory.npz')as d:
  assert d['sphere_centers_m'].shape==(2,1,3)
  np.testing.assert_allclose(d['sphere_centers_m'][-1,0],[.06,0,.079],atol=1e-13)
  assert d['tool_wrenches_N_Nm'].shape==(2,2,6)
 assert row['sphere_names']==['striker']and row['radii_m']==[.025]
