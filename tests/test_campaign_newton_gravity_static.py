from experiments.campaign_newton_gravity_static import gates,JOBS

def test_original_force_stationarity_velocity_are_independent_gates():
 assert gates(.001,1e-8,1e-8,.9,0)
 assert not gates(.02,1e-8,1e-8,.9,0)
 assert not gates(.001,1e-6,1e-8,.9,0)
 assert not gates(.001,1e-8,1e-6,.9,0)
 assert not gates(.001,1e-8,1e-8,.09,0)
 assert not gates(.001,1e-8,1e-8,.9,1e-16)
 assert not gates(float('nan'),1e-8,1e-8,.9,0)

def test_scope_no_contact_or_law_retuning():
 assert len(JOBS)==5
 assert {j['E'] for j in JOBS}=={300000,600000}
 assert {j['tets'] for j in JOBS}=={96,768}
 assert sum(j['iterations']==128 for j in JOBS)==1
