import numpy as np
import pytest
from experiments.campaign_world_plane_contact import plane_contact


def inputs():
    return dict(positions_m=np.array([[.02,0.,-.001],[0.,.03,.002],[0.,0.,-.003]]),
        velocities_m_s=np.array([[.1,.2,-.3],[0.,0.,-1.],[.2,0.,.4]]),
        surface_areas_m2=np.array([.001,.002,0.]),plane_origin_m=np.zeros(3),
        plane_normal=np.array([0.,0.,1.]),plane_velocity_m_s=np.zeros(3),
        stiffness_Pa_per_m=1e5,damping_Pa_s_per_m=200.,friction_coefficient=.4,
        slip_regularization_m_s=.001)


def test_separation_and_interior_nodes_have_no_force_and_no_adhesion():
    p=inputs();r=plane_contact(**p)
    np.testing.assert_array_equal(r['forces_world_N'][1:],0.)
    p['velocities_m_s'][0,2]=10.;r=plane_contact(**p)
    assert r['normal_force_N'][0]>0
    np.testing.assert_array_equal(r['damping_forces_world_N'],0.)


def test_elastic_energy_gradient_matches_force_away_from_switch():
    p=inputs();p['damping_Pa_s_per_m']=0.;p['friction_coefficient']=0.
    f=plane_contact(**p)['forces_world_N'];h=1e-7
    for i in range(3):
        for j in range(3):
            a={**p,'positions_m':p['positions_m'].copy()};b={**p,'positions_m':p['positions_m'].copy()}
            a['positions_m'][i,j]+=h;b['positions_m'][i,j]-=h
            derivative=(plane_contact(**a)['elastic_energy_J']-plane_contact(**b)['elastic_energy_J'])/(2*h)
            assert derivative==pytest.approx(-f[i,j],abs=2e-12)


def test_passive_rotation_translation_and_velocity_frame_covariance():
    p=inputs();r=plane_contact(**p);angle=.7
    q=np.array([[np.cos(angle),0.,np.sin(angle)],[0.,1.,0.],[-np.sin(angle),0.,np.cos(angle)]])
    shift=np.array([.4,-.1,.2]);boost=np.array([.2,.3,-.1]);t={**p}
    t['positions_m']=p['positions_m']@q.T+shift;t['plane_origin_m']=p['plane_origin_m']@q.T+shift
    t['velocities_m_s']=p['velocities_m_s']@q.T+boost;t['plane_velocity_m_s']=p['plane_velocity_m_s']@q.T+boost
    t['plane_normal']=q@p['plane_normal'];s=plane_contact(**t)
    np.testing.assert_allclose(s['forces_world_N'],r['forces_world_N']@q.T,atol=2e-15)
    for k in ['elastic_energy_J','damping_dissipation_W','friction_dissipation_W']:
        assert s[k]==pytest.approx(r[k],abs=1e-15)


def test_dissipation_and_coulomb_bound_for_random_nodes():
    rng=np.random.default_rng(43);p=inputs();p.update(positions_m=rng.normal(size=(128,3))*.002,
        velocities_m_s=rng.normal(size=(128,3)),surface_areas_m2=rng.uniform(0,.01,128))
    r=plane_contact(**p)
    assert r['damping_dissipation_W']>=0 and r['friction_dissipation_W']>=0
    assert np.all(np.linalg.norm(r['friction_forces_world_N'],axis=1)<=p['friction_coefficient']*r['normal_force_N']+1e-14)
    assert np.all(r['normal_force_N']>=0)
    np.testing.assert_allclose(r['friction_forces_world_N']@p['plane_normal'],0.,atol=1e-14)


def test_invalid_normal_and_parameters_fail_without_repair():
    p=inputs();p['plane_normal']=np.array([0.,0.,2.])
    with pytest.raises(ValueError):plane_contact(**p)
    p=inputs();p['damping_Pa_s_per_m']=-1.
    with pytest.raises(ValueError):plane_contact(**p)
