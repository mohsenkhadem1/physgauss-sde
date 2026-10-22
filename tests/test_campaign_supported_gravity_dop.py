from pathlib import Path
import numpy as np
import pytest
from scipy.linalg import expm
from experiments.campaign_p2_keast_quadrature import KeastBulkP2Geometry
from experiments.campaign_p2_exact_quadrature import ExactBulkP2Geometry
from experiments.campaign_newton_stable_material import StableP2Block
from experiments.campaign_supported_gravity_dop import (WarpBulkP2Force,SupportedBulkDynamics,AcceptedDOPCheckpoint,read_dop_checkpoint,streaming_dop853)


@pytest.fixture(scope='module')
def controls(tmp_path_factory):
    import warp as wp
    wp.config.kernel_cache_dir=str(tmp_path_factory.mktemp('warp-cpu-cache'))
    m=KeastBulkP2Geometry((4,2,2),young=100000.,gravity=9.81,support='both_ends')
    w=WarpBulkP2Force(m,device='cpu');d=SupportedBulkDynamics(m,w)
    old=ExactBulkP2Geometry((4,2,2),young=100000.,gravity=9.81,support='both_ends')
    cpu=StableP2Block((4,2,2),young=100000.,poisson=.35,density=625.,gravity=0.,traction=0.,order=3)
    for name in ['bary','weights','N','check_bary','load','mask','free']:setattr(cpu,name,getattr(old,name).copy())
    return m,w,d,cpu


def test_pure_bulk_cpu_force_energy_at_finite_deformation_and_static_equilibrium(controls):
    m,w,d,cpu=controls;X=m.rest
    u=np.column_stack([.02*X[:,0]+.1*X[:,1]*X[:,2],-.03*X[:,1]+.3*X[:,0]*X[:,2],.03*X[:,2]+.1*X[:,0]*X[:,1]])
    UE,g,_,J=cpu.elastic(X+u);actual=w.evaluate(u)
    assert w.integration_points==24 and w.additional_J_guard_points==90
    assert not hasattr(w,'contact_E') and not hasattr(w,'floor_z')
    np.testing.assert_allclose(actual['gradient_N'],g,atol=2e-10,rtol=2e-12)
    assert abs(actual['elastic_energy_J']-UE)<2e-12
    answer=cpu.solve(force_tolerance_N=1e-10);state=d.mechanics(d.pack(answer['positions']-m.rest,np.zeros_like(m.rest)))
    assert np.linalg.norm((w.evaluate(answer['positions']-m.rest)['gradient_N']-m.load)[~m.mask])<=1e-10
    assert state['mass_residual_N']<=1e-9 and state['minimum_J']>.1


def test_exact_full_mass_fixed_reactions_and_coordinate_constraints(controls):
    m,w,d,cpu=controls;y=d.pack(np.zeros_like(m.rest),np.zeros_like(m.rest));a=d.mechanics(y);free=d.free_nodes;fixed=d.fixed_nodes
    expected=d.mass_lu.solve(m.load[free]);np.testing.assert_allclose(a['acceleration_m_s2'][free],expected,atol=1e-12,rtol=1e-13)
    inertial_fixed=d.mass[fixed][:,free]@expected
    np.testing.assert_allclose(a['reaction_N'][fixed],inertial_fixed-m.load[fixed],atol=1e-12,rtol=1e-13)
    assert np.linalg.norm(inertial_fixed)>1e-3  # M_cf is physically material here.
    assert a['mass_residual_N']<=1e-9 and not np.any(a['acceleration_m_s2'][fixed])
    np.testing.assert_allclose((d.mass@a['acceleration_m_s2']).sum(0),m.load.sum(0)+a['reaction_N'][fixed].sum(0),atol=2e-12,rtol=0)
    assert np.isfinite(d.rhs(0.,y)).all()
    with pytest.raises(ValueError,match='exact-fixed'):d.pack(np.ones_like(m.rest),np.zeros_like(m.rest))


def test_public_dop_discrete_clock_energy_and_bounded_memory_checkpoint(tmp_path):
    M=np.array([[2.,.3],[.3,1.]]);K=np.array([[7.,-1.],[-1.,4.]]);A=np.block([[np.zeros((2,2)),np.eye(2)],[-np.linalg.solve(M,K),np.zeros((2,2))]])
    y0=np.array([.1,-.2,.3,.15]);times=np.linspace(0,.1,25)
    def rhs(t,y):return A@y
    def telemetry(y):return dict(mechanical_energy_J=.5*(y[:2]@K@y[:2]+y[2:]@M@y[2:]),mass_residual_N=0.,minimum_J=1.)
    policy=dict(every_accepted_steps=32,on_initial=True,on_new_common_output=True,on_final=True,max_uncommitted_accepted_steps_bound=32)
    metadata={'source_sha256':{'synthetic':'notaphysicalprotocol'},'model':'independentcoupledlinearoscillator','force':'zero','checkpoint_policy':policy}
    writer=AcceptedDOPCheckpoint(tmp_path,metadata);answer=streaming_dop853(rhs,telemetry,y0,times,rtol=1e-10,atol=1e-12,max_step_s=.01,energy_tolerance_J=1e-10,checkpoint=writer,checkpoint_every_accepted_steps=32)
    exact=np.array([expm(A*t)@y0 for t in times]);np.testing.assert_allclose(answer['common_states'],exact,atol=2e-12,rtol=2e-12)
    np.testing.assert_array_equal(answer['times_s'],times)
    assert 'accepted_states' not in answer and len(answer['accepted_scalar_ledger'])==answer['accepted_steps']+1
    saved=read_dop_checkpoint(tmp_path,metadata);np.testing.assert_allclose(saved['current_y'],exact[-1],atol=2e-12,rtol=0);np.testing.assert_allclose(saved['current_f'],rhs(.1,saved['current_y']),atol=0,rtol=0)
    assert 'accepted_states' not in saved and len(saved['common_states'])==25 and float(saved['next_h_abs_s'])>0
    with pytest.raises(ValueError,match='metadata'):read_dop_checkpoint(tmp_path,{'different':'physicalidentity'})
    pointer=__import__('json').loads((tmp_path/'latest.json').read_text());file=tmp_path/pointer['file'];file.write_bytes(file.read_bytes()+b'tamper')
    with pytest.raises(ValueError,match='prefix'):read_dop_checkpoint(tmp_path,metadata)


def test_throttled_even_interval_alternates_on_commits_not_step_count(tmp_path):
    import json
    policy=dict(every_accepted_steps=32,on_initial=True,on_new_common_output=True,on_final=True,max_uncommitted_accepted_steps_bound=32)
    metadata={'synthetic':True,'checkpoint_policy':policy};slots=[]
    class Recording(AcceptedDOPCheckpoint):
        def __call__(self,*args):
            super().__call__(*args);slots.append(json.loads((self.directory/'latest.json').read_text())['file'])
    writer=Recording(tmp_path,metadata);y0=np.array([1.,0.])
    def rhs(t,y):return np.array([y[1],-y[0]])
    def telemetry(y):return dict(mechanical_energy_J=.5*float(y@y),mass_residual_N=0.,minimum_J=1.)
    answer=streaming_dop853(rhs,telemetry,y0,[0.,.1],rtol=1e-10,atol=1e-12,max_step_s=.001,energy_tolerance_J=1e-10,checkpoint=writer,checkpoint_every_accepted_steps=32)
    assert len(slots)>=4 and all(a!=b for a,b in zip(slots[:-1],slots[1:]))
    assert answer['checkpoint_commits']==len(slots)<answer['accepted_steps']/8
    assert answer['maximum_observed_uncommitted_accepted_steps']<=32
    saved=read_dop_checkpoint(tmp_path,metadata)
    assert len(saved['accepted_times_s'])==answer['accepted_steps']+1
    # An interrupted write to the other slot does not invalidate latest.
    (tmp_path/'prefix-1.npz.tmp').write_bytes(b'incomplete')
    read_dop_checkpoint(tmp_path,metadata)
    with pytest.raises(FileExistsError,match='Fresh'):AcceptedDOPCheckpoint(tmp_path,metadata)
