import numpy as np
import pytest
from experiments import run_p2_contact_spatial_pairs as c

def test_exact_new_mesh_phase_resources_and_no_shared_output_ids():
 assert c.PHASES['324tets']['cells']==[6,3,3] and c.PHASES['768tets']['cells']==[8,4,4]
 assert c.PHASES['324tets']['case_caps_s']==[900,1800] and c.PHASES['768tets']['case_caps_s']==[1800,3600]
 assert c.PHASES['324tets']['CPU_affinity_cores']==[2,3] and c.PHASES['768tets']['CPU_affinity_cores']==[4,5]
 assert len({x['GPU_uuid'] for x in c.PHASES.values()})==2

def test_common_queries_q_AND_v_unchanged_and_ratio_correct():
 a=dict(times_s=np.array([0.,1.]),material_probes_m=np.zeros((81,3)),query_positions_m=np.zeros((2,81,3)),query_velocities_m_s=np.zeros((2,81,3)))
 b={k:v.copy() for k,v in a.items()};b['query_positions_m'][1,0,2]=7e-6;b['query_velocities_m_s'][1,0,2]=3e-5
 r=c.spatial_pair(a,b,2.);assert r['q_passed'] and r['v_passed']
 r=c.spatial_pair(a,b,3.);assert not r['q_passed'] and r['v_passed']
 b['material_probes_m'][0,0]=.1
 with pytest.raises(AssertionError):c.spatial_pair(a,b,3.)
