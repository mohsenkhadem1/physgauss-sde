import copy,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from experiments import audit_fixture_balanced_held_nh_refinement_v01 as a


def good_scalars():
    d=dict(minimum_J=.98,maximum_absolute_symmetric_strain_eigenvalue=.02,nonrigid_RMS_m=200e-6,nonrigid_fraction=.6)
    s=dict(passed=True,smallest_generalized_Ritz_ratio=.9,rest_norm_squared=1.,normalization_passed=True,generalized_dual_residual=1e-10,relative_pair_residual=1e-10/.9,dimensionless_Ritz_lower_endpoint=.9-1e-10,exact_rest_solve_residual=1e-17,exact_rest_solve_tolerance=1e-13,explicit_physical_pair_residual_norm=1e-10,matrix_modified=False,rest_SPD_externally_assumed=True,rigorous_global_eigenvalue_bound=False)
    e=dict(passed=True,free_residual_N=1e-12,tolerance_N=1e-10,force_balance_error_N=1e-12,force_balance_tolerance_N=1e-8,moment_balance_error_Nm=1e-12,moment_balance_tolerance_Nm=1e-8,exact_clamp=True)
    row=dict(original_free_R_l2_N=1e-12,linear_original_R_l2_N=1e-12,equilibrium=e,stability=s,physical_passed=True,excitation_goal_met=True,weak_prefix_control=False,load_eligible=True,bank_admitted=False)
    return row,d


def test_independent_physics_excitation_and_stability_scalar_flags():
    row,d=good_scalars();assert a.scalar_status(row,d)['load_eligible']
    row['equilibrium']['force_balance_error_N']=1.
    with pytest.raises(ValueError,match='equilibrium'):a.scalar_status(row,d)
    row,d=good_scalars();row['stability']['relative_pair_residual']=.1
    with pytest.raises(ValueError,match='Ritz residual'):a.scalar_status(row,d)
    row,d=good_scalars();row['stability']['matrix_modified']=True
    with pytest.raises(ValueError,match='stability'):a.scalar_status(row,d)
    row,d=good_scalars();d['nonrigid_RMS_m']=99e-6;row.update(excitation_goal_met=False,load_eligible=False)
    assert a.scalar_status(row,d)['physical_passed']and not a.scalar_status(row,d)['load_eligible']


def test_all_six_membership_no_majority_or_physical_only_rescue():
    rows=[dict(n=n,label=l,completed=True,physical_passed=True,load_eligible=True,excitation_goal_met=True,weak_prefix_control=False)for n in a.stage.LEVELS for l in a.stage.PILOT_LOADS];spatial={l:dict(spatial_development_passed=True)for l in a.stage.PILOT_LOADS}
    assert a.classification(rows,spatial)==a.stage.classification(rows,spatial)
    for mutate in(lambda r:r.pop(),lambda r:r[0].update(load_eligible=False),lambda r:r[0].update(physical_passed=False)):
        bad=copy.deepcopy(rows);mutate(bad);assert not a.classification(bad,spatial)['pilot_qualified']
    bad=copy.deepcopy(spatial);bad[a.stage.PILOT_LOADS[0]]['spatial_development_passed']=False
    assert a.classification(rows,bad)['all_physical_passed']and not a.classification(rows,bad)['pilot_qualified']


def test_geometry_only_gradients_match_tiny_original_without_solver():
    f=a.geometry_forward(1);g=a.stage.resolution.make_forward(1,a.stage.PILOT_LAYOUT)
    assert np.array_equal(f.X,g.X)and np.array_equal(f.elements,g.elements)and np.array_equal(f.linear_gradients,g.linear_gradients)
    assert not hasattr(f,'Kunit')and not hasattr(f,'W')
    u=np.zeros_like(f.X);u[:,0]=.01*f.X[:,0]
    d=a.stage.resolution.deformation(f,u,finest=1)
    assert abs(d['minimum_J']-1.01)<1e-13 and abs(d['maximum_absolute_symmetric_strain_eigenvalue']-.01)<1e-13


def test_saved_capture_query_force_topology_and_wrench_controls(monkeypatch,tmp_path):
    n=1;f=a.geometry_forward(n);label=a.stage.PILOT_LOADS[0];loads,info=a.stage.bank.load_catalog(n);u=np.zeros_like(f.X);u[:,1]=1e-6*np.sin(f.X[:,0]*np.pi/.36);u[f.fixed]=0;points=a.stage.bank.measurement_points()
    real=a.stage.resolution.deformation;monkeypatch.setattr(a.stage.resolution,'deformation',lambda f,u,finest:real(f,u,finest=1))
    d=real(f,u,finest=1);path=tmp_path/'capture.npz';values=dict(displacement_m=u,linear_displacement_m=u,rest_m=f.X,force_N=loads[label],elements=f.elements,exact_constraint_nodes=f.fixed,query_points_m=points,query_displacement_m=a.stage.resolution.field_queries(n,u,points));np.savez_compressed(path,**values)
    row=dict(n=n,label=label,capture=dict(path=str(path)),known_load=info[label],actual_deformed_wrench=a.stage.prior.force_wrench(f.X,loads[label],u),deformation=d)
    out,_=a.check_capture(row);assert np.array_equal(out,u)
    values['force_N']=values['force_N'].copy();values['force_N'][0,0]+=1e-6;np.savez_compressed(path,**values)
    with pytest.raises(ValueError,match='force'):a.check_capture(row)
    values['force_N']=loads[label];values['query_displacement_m']=values['query_displacement_m']+1e-12;np.savez_compressed(path,**values)
    with pytest.raises(ValueError,match='query'):a.check_capture(row)


def fake_metadata(monkeypatch,tmp_path):
    monkeypatch.setattr(a.stage,'LOCAL',tmp_path);monkeypatch.setattr(a.stage,'ROOT',tmp_path);monkeypatch.setattr(a.stage,'record',lambda p:dict(path=str(p),sha256=a.stage.sha(p)))
    rows=[]
    for n in a.stage.LEVELS:
        for label in a.stage.PILOT_LOADS:
            path=tmp_path/'raw-LOCAL'/f'n{n}-{label}.npz';path.parent.mkdir(exist_ok=True);path.write_bytes(b'not-an-array')
            rows.append(dict(n=n,label=label,completed=True,layout=a.stage.PILOT_LAYOUT,E_Pa=2000.,K_N_m=5.,capture=a.stage.record(path)))
    for name in('protocol.json','release.json'):(tmp_path/name).write_text('{}')
    release=a.stage.record(tmp_path/'release.json');digest=a.stage.sha(tmp_path/'protocol.json');p=dict(source_SHA256={a.SOURCE:'s',a.TEST:'t'});monkeypatch.setattr(a.stage,'validate',lambda d:p)
    r=dict(completed=True,protocol_SHA256=digest,release=release,worker_pid=102,rows=rows,wall_s=10.,peak_RSS_bytes=100,no_noise_or_truth_draws=True,bank_admitted=False);a.stage.write(tmp_path/'result-LOCAL.json',r);cache=tmp_path/'cache.json';cache.write_bytes((tmp_path/'result-LOCAL.json').read_bytes())
    e=dict(terminal=True,completed=True,worker_exit_code=0,cleanup_exit_code=0,controller_pid=101,worker_pid=102,protocol_SHA256=digest,release=release,result=a.stage.record(tmp_path/'result-LOCAL.json'),external_cache=a.stage.record(cache),wall_s=11.,peak_aggregate_RSS_bytes=200);a.stage.write(tmp_path/'execution.json',e)
    c=dict(root_reaped=True,root_exit_code=0,all_owned_PID_and_groups_absent=True,protocol_SHA256=digest,source_SHA256=p['source_SHA256'],execution_SHA256=a.stage.sha(tmp_path/'execution.json'),owned_pids=[101,102]);a.stage.write(tmp_path/'root-closure.json',c)
    seen=[];monkeypatch.setattr(a.stage.prior.shared.old,'absent',lambda pid:seen.append(pid));monkeypatch.setattr(a.np,'load',lambda *a,**k:pytest.fail('No NPZ before metadata barrier'))
    return p,e,r,c,seen


def test_external_reap_and_all_six_hashes_precede_npz(monkeypatch,tmp_path):
    p,e,r,c,seen=fake_metadata(monkeypatch,tmp_path);a.metadata_barrier();assert seen==[101,102]
    c['root_reaped']=False;a.stage.write(tmp_path/'root-closure.json',c)
    with pytest.raises(ValueError,match='reap'):a.metadata_barrier()
    c['root_reaped']=True;a.stage.write(tmp_path/'root-closure.json',c);Path(r['rows'][-1]['capture']['path']).write_bytes(b'tampered')
    with pytest.raises(ValueError,match='capture'):a.metadata_barrier()


def test_independent_spatial_arithmetic_and_public_query_gate(monkeypatch):
    fields={(n,label):np.zeros((1,3))for n in a.stage.LEVELS for label in a.stage.PILOT_LOADS}
    monkeypatch.setattr(a.stage.resolution,'common_rms',lambda fields,finest:np.array([[0,0,0],[100e-6,0,0],[0,5e-6,0]]))
    monkeypatch.setattr(a.stage.resolution,'field_queries',lambda n,u,p:np.full((len(p),3),n*10e-6))
    result=a.spatial_results(fields);assert result==a.stage.spatial_results(fields)
    assert all(v['medium_adequate']and v['spatial_passed']and not v['public_point20um_passed']and not v['spatial_development_passed']for v in result.values())
