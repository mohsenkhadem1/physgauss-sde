import numpy as np
import pytest
from scipy.sparse.linalg import splu
from experiments import campaign_fixture_gaussian_linear_screen_v01 as m


def test_fixed72_center_first_paired_worklist_all_layouts_parameters_loads_retained():
    w=m.worklist();assert len(w)==72 and len({(r['layout'],r['E_Pa'],r['K_N_m'],r['n']) for r in w})==72
    assert [r['n'] for r in w[:2]]==[3,6]
    assert all(r['E_Pa']==4000 and r['K_N_m']==np.sqrt(150) for r in w[:2])
    assert len(m.bank.load_specs())==11 and len(m.bank.measurement_specs())==60
    assert len({r['layout'] for r in w})==4


def test_new_measurement_modal_full_modes_particular_and_direct_field_parity_n1():
    # Manufactured operator callback control, not an actual development row.
    f=m.geometry.make_forward(1,m.bank.LAYOUTS[0]['id']);loads,_=m.geometry.load_catalog(1)
    model=m.make_measurement_modal(f,loads);labels=[s['id'] for s in m.bank.measurement_specs()]
    assert model.modal.diagnostics['all_boundary_modes_retained']
    assert not model.modal.diagnostics['interior_basis_truncated']
    assert model.modal.diagnostics['exact_particular_1_over_E_term'] and model.modal.eigenvalues[-1]==0
    for E,K in [(2000.,5.),(4000.,np.sqrt(150)),(8000.,30.)]:
        rhs=np.column_stack([v.ravel() for v in loads.values()]);u=np.zeros_like(rhs);u[f.free]=splu(E*f.Kunit_free+K*f.W_free).solve(rhs[f.free]);fields=u.T.reshape(11,len(f.X),3)
        direct=m.measurement_response(1,{name:fields[j:j+1] for j,name in enumerate(loads)})
        unit=np.log([E/2000,K/5])/np.log([4,6]);np.testing.assert_allclose(model.select(labels)(unit[None]),direct,atol=2e-12,rtol=0)
    assert model.modal.output_shape==(1,60)


def test_all_category_separation_has_noise_units_and_missing_category_blocks():
    rows=[]
    for r in m.worklist():rows.append(dict(r,response=np.full(60,20e-6*next(i for i,s in enumerate(m.bank.LAYOUTS) if s['id']==r['layout']))))
    s=m.separation(rows);assert len(s)==108
    assert s[0]['prefix4_L2_over_sigma']==2 and s[0]['held48_RMS_over_sigma']==1
    with pytest.raises(ValueError,match='four categories'):m.separation(rows[:-1])
