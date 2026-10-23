from experiments import run_p2_soft_floor_screen as c

def test_explicit_newphysical_screen_resources_and_no_new_timepairclaim():
 assert [j['cells'] for j in c.JOBS]==[[4,2,2],[6,3,3],[8,4,4]]
 assert [j['wall_cap_s'] for j in c.JOBS]==[600,900,1800]
 assert [j['CPU_affinity_cores'] for j in c.JOBS]==[[0,1],[2,3],[4,5]]
 assert len({j['GPU_uuid'] for j in c.JOBS})==3
 assert c.CONFIG==dict(name='accuracy_1e10',rtol=1e-10,displacement_atol_m=1e-13,velocity_atol_m_s=1e-11,impulse_atol_Ns=1e-13)
 assert 'soft-floor-screen' in str(c.OUT)
