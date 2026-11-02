import numpy as np
import pytest
from scripts.compositional_dynamics_v17 import cohort_reference_sensitivity as m
from scripts.compositional_dynamics_v17.prototype import face_grid

def fixture():
 settings=dict(sample_interval_s=.001,mass_kg=.27,gravity_m_s2=9.8,dimensions_m=[.12,.06,.06],reference_diagnostics=dict(minimum_J=.1,maximum_penetration_m=.0012,energy_drift_relative=.01,spatial_world_coordinate_RMS_m=.0003,spatial_shape_coordinate_RMS_m=.00015,temporal_world_coordinate_RMS_m=.0003,temporal_shape_coordinate_RMS_m=.00015))
 job=dict(duration_s=.002,dt_s=.001,clearance_m=.01)
 rest=(face_grid(17)*[.12,.06,.06]+[0,-.03,-.03]).reshape(6,17,17,3)
 ledger=np.zeros((3,21));ledger[:,0]=np.arange(3)*.001;ledger[:,7]=1
 data=dict(times_s=np.arange(3)*.001,surface_rest=rest,surface_frames=np.repeat(rest[None],3,0),ledger=ledger,saved_dense_penetration_m=np.zeros(3))
 return settings,job,data

def test_rigid_shift_and_rotation_separate_shape():
 s,j,a=fixture();b={k:v.copy()for k,v in a.items()};theta=.1;R=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
 b['surface_frames']=b['surface_frames']@R+[.001,0,0]
 ar={'diagnostics':m.trajectory_diagnostics(a,j,s,True)};br={'diagnostics':m.trajectory_diagnostics(b,j,s,True)}
 result=m.compare_pair((ar,a),(br,b),'spatial',s)
 assert result['shape_coordinate_RMS_m']<1e-15 and result['world_coordinate_RMS_m']>.0003 and not result['passed']

def test_energy_failure_and_partial_trajectory_never_pass():
 s,j,a=fixture();a['ledger'][-1,1]=.1
 assert not m.trajectory_diagnostics(a,j,s,True)['passed']
 a['ledger'][-1,1]=0
 assert not m.trajectory_diagnostics(a,j,s,False)['passed']
 a['saved_dense_penetration_m'][-1]=np.nan
 with pytest.raises(ValueError,match='Nonfinite'):m.trajectory_diagnostics(a,j,s,True)

def test_identical_pair_and_missing_pair():
 s,j,a=fixture();r={'diagnostics':m.trajectory_diagnostics(a,j,s,True)}
 assert m.compare_pair((r,a),(r,a),'temporal',s)['passed']
 assert not m.compare_pair(None,(r,a),'temporal',s)['passed']

def test_no_target_access_without_explicit_release(tmp_path,monkeypatch):
 path=tmp_path/'release.json';m.write(path,{'role':'wrong'})
 def forbidden(*a):raise AssertionError('Prediction or reference reader called too early')
 monkeypatch.setattr(m,'committed',forbidden)
 with pytest.raises(ValueError,match='release'):m.release_guard(dict(protocol_sha='p',freeze_sha='f'),path)
