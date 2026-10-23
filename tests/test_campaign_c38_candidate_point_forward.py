import copy
import numpy as np
import pytest
from experiments.campaign_c38_candidate_point_forward import CandidatePointForward
from experiments.campaign_c38_nonlinear_table_catalog import catalog
from experiments.campaign_c38_table_validation import POINTS


def archive():
    rows=[]
    for i,c in enumerate(catalog()):
        if c['material_use'] == 'TABLE_NODE':
            for k,n in enumerate((32,40)):
                rows.append((2*i+k,c,n))
    q=[]
    for _,c,n in rows:
        x=75000./c['young_Pa']; scale=1 if c['support']=='both_ends' else 2
        q.append(POINTS+POINTS*(c['Fz_N']+4)*scale*(.01*x+.002*x*x))
    return dict(source_index=np.array([r[0] for r in rows]),case_ids=np.array([r[1]['id'] for r in rows]),material_use=np.array(['TABLE_NODE']*48),information_role=np.array('CANDIDATE_POINT_FORWARD_TABLE_NOT_ADMITTED'),cells=np.array([[n,n//2,n//2] for _,c,n in rows]),support=np.array([c['support'] for _,c,n in rows]),young_Pa=np.array([c['young_Pa'] for _,c,n in rows]),Fz_N=np.array([c['Fz_N'] for _,c,n in rows]),material_probes_m=POINTS.copy(),probe_positions_m=np.array(q))


def test_independent_quadratic_and_archive_ownership():
    a=archive(); f=CandidatePointForward(a); E=89190.;ids=np.array([11,14,17]);x=75000./E
    expected=POINTS[ids]*6*2*(.01*x+.002*x*x)
    np.testing.assert_allclose(f.predict_displacement('separated_bottom_patches',2.,E,ids),expected,atol=1e-16,rtol=0)
    a['probe_positions_m'][:]=0
    np.testing.assert_allclose(f.predict_displacement('separated_bottom_patches',2.,E,ids),expected,atol=1e-16,rtol=0)


def test_source_index_joins_independent_of_archive_order():
    a=archive();b={k:(v[::-1] if v.shape and len(v)==48 else v) for k,v in a.items()}
    f=CandidatePointForward(a);g=CandidatePointForward(b)
    np.testing.assert_array_equal(f.observation_coefficients('both_ends',0.,[3,6]),g.observation_coefficients('both_ends',0.,[3,6]))


def test_reject_truth_role_or_index_tamper():
    a=archive();a['information_role']=np.array('PRIVATE_DIRECT_WITHHELD_INTERPOLATION_VALIDATION')
    with pytest.raises(ValueError):CandidatePointForward(a)
    a=archive();a['source_index'][0]=16
    with pytest.raises(ValueError):CandidatePointForward(a)
    a=archive();a['support'][0]='bad'
    with pytest.raises(ValueError):CandidatePointForward(a)


def test_no_unqualified_force_material_or_points():
    f=CandidatePointForward(archive())
    for F in [.5,1.01,np.nan]:
        with pytest.raises(ValueError):f.observation_coefficients('both_ends',F,[11])
    for ids in [[63],[11,11],[-1]]:
        with pytest.raises(ValueError):f.observation_coefficients('both_ends',1.,ids)
    with pytest.raises(ValueError):f.predict_displacement('both_ends',1.,74999.,[11])
