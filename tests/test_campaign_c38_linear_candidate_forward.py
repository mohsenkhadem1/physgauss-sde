import numpy as np
import pytest
from experiments.campaign_c38_linear_candidate_forward import LinearCandidatePointForward
from experiments.run_campaign_c38_static_prequalification import cases
from experiments.campaign_c38_table_validation import POINTS


def rows():
    out=[]
    for c in cases():
        for nx in (32,40):
            a=1 if c['support']=='both_ends' else 2
            g=POINTS*.02*a;u=POINTS*.003*a
            q=POINTS+(75000./c['young_Pa'])*(g+c['Fz_N']*u)
            out.append(dict(case=c,cells=[nx,nx//2,nx//2],material_probes_m=POINTS.copy(),probe_positions_m=q))
    return out


def test_new_target_force_and_material_from_independent_linear_law():
    f=LinearCandidatePointForward(rows());E=120000.;ids=[11,14,17]
    predicted=f.predict_displacement('separated_bottom_patches',1.,E,ids)
    np.testing.assert_allclose(predicted,(75000./E)*POINTS[ids]*(.02+.003)*2,atol=1e-16,rtol=0)
    assert np.all(f.observation_coefficients('both_ends',1.,ids)[[0,2]]==0)


def test_requires_complete_queries_and_rejects_non_linear_tamper():
    a=rows()
    with pytest.raises(ValueError):LinearCandidatePointForward(a[:-1])
    a[-1]['probe_positions_m'][20,0]+=1e-6
    with pytest.raises(ValueError,match='scaling'):LinearCandidatePointForward(a)


def test_physical_and_point_binding_tamper_rejected():
    a=rows();a[-1]['cells']=[41,20,20]
    with pytest.raises(ValueError):LinearCandidatePointForward(a)
    a=rows();a[0]['material_probes_m'][0,0]+=1e-6
    with pytest.raises(ValueError):LinearCandidatePointForward(a)


def test_stores_own_coefficients_and_no_force_extrapolation():
    a=rows();f=LinearCandidatePointForward(a);p=f.predict_displacement('both_ends',0.,100000.,[11])
    a[0]['probe_positions_m'][:]=0
    np.testing.assert_array_equal(p,f.predict_displacement('both_ends',0.,100000.,[11]))
    with pytest.raises(ValueError):f.observation_coefficients('both_ends',.1,[11])
