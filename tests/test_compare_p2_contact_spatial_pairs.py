from experiments.compare_p2_contact_spatial_pairs import combine,contraction

def row(q,v,qpass=True,vpass=True):return dict(maximum_common81_q_difference_m=q,maximum_common81_v_difference_m_s=v,q_passed=qpass,v_passed=vpass)
def test_finest_gate_requires_contraction_and_retains_q_v_separately():
 a=row(8e-6,1e-3);b=row(4e-6,8e-4,True,False);r=combine(a,b)
 assert r['finest_conditional_q_gate_met'] and not r['finest_conditional_v_gate_met']
 assert not r['continuum_or_contact_dataset_admitted']
 r=combine(a,row(9e-6,2e-3));assert not r['q_errors_contract'] and not r['v_errors_contract']
 assert contraction(0.,0.) and not contraction(1.,1.)
