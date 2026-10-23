import numpy as np
import pytest
from experiments import run_p2_contact_root128_time_control as c

def toy(delta=0.,different=True):
 a=dict(times_s=np.array([0.,1.]),positions_m=np.zeros((2,3,3)),velocities_m_s=np.zeros((2,3,3)),output_mechanical_energies_J=np.zeros(2),query_positions_m=np.zeros((2,2,3)),accepted_times_s=np.array([0.,.5,1.]))
 b={k:v.copy() for k,v in a.items()};b['velocities_m_s'][1,0,2]=delta
 if different:b['accepted_times_s']=np.array([0.,.25,.5,1.])
 return a,b

def test_separate128_time_gate_cannot_inherit_surface_or_identical_partition():
 a,b=toy(9e-5);assert c.time_pair(a,b,1e-6)['time_reference_qualified']
 a,b=toy(1.01e-4);assert not c.time_pair(a,b,1e-6)['time_reference_qualified']
 a,b=toy(different=False);r=c.time_pair(a,b,1e-6);assert r['passed'] and not r['time_reference_qualified']
 b['times_s'][1]=.9
 with pytest.raises(AssertionError):c.time_pair(a,b,1e-6)

def test_exact_new_scope_and_legacy_immutable_lineage():
 assert c.CONFIG==dict(name='root128_accuracy_1e9',rtol=1e-9,displacement_atol_m=1e-12,velocity_atol_m_s=1e-10,impulse_atol_Ns=1e-13)
 assert c.GPU=='GPU-5b432ce5-a24a-68ba-150f-682265260a9b'
 assert c.CONTROL_CAPTURE.name=='trajectory.npz' and 'order128-trajectory' in str(c.CONTROL_CAPTURE)
 assert c.EXPECTED=='5a2c638041780380c3e7a9dc3afc56036bdb0aec48b1deff4416ff94247a0fee'
