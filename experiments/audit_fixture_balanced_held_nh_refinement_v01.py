"""Independent saved-field six-control audit; no equilibrium/PDE/RNG replay.

All process, source and six capture commitments precede deserialization.
Sampled deformation, known loads/wrenches, public queries and spatial metrics
are recomputed. Residual and tangent eigenpair scalar witnesses are checked;
large sparse equilibrium/tangent operators are deliberately not reconstructed.
"""
import json
from types import SimpleNamespace
import numpy as np
from experiments import run_fixture_balanced_held_nh_refinement_v01 as stage
SOURCE='experiments/audit_fixture_balanced_held_nh_refinement_v01.py'
TEST='tests/test_audit_fixture_balanced_held_nh_refinement_v01.py'


def need(v,msg):
    if not v:raise ValueError(msg)
def finite(x):return isinstance(x,(float,int))and not isinstance(x,bool)and np.isfinite(x)
def le(x,limit):return finite(x)and 0<=x<=limit

def metadata_barrier():
    base=stage.LOCAL;digest=stage.sha(base/'protocol.json');p=stage.validate(digest)
    e=json.loads((base/'execution.json').read_text());r=json.loads((base/'result-LOCAL.json').read_text());c=json.loads((base/'root-closure.json').read_text())
    need(e.get('terminal')is True and e.get('completed')is True and e.get('worker_exit_code')==0 and e.get('cleanup_exit_code')==0 and 'failure'not in e and r.get('completed')is True,'Complete clean six-control process barrier')
    need(e['protocol_SHA256']==r['protocol_SHA256']==digest and e['result']==stage.record(base/'result-LOCAL.json')and e['release']==r['release']==stage.record(base/'release.json'),'Exact protocol/release/result')
    need(e['external_cache']==stage.record(stage.ROOT/e['external_cache']['path'])and(stage.ROOT/e['external_cache']['path']).read_bytes()==(base/'result-LOCAL.json').read_bytes(),'Immutable external cache')
    need(c.get('root_reaped')is True and c.get('root_exit_code')==0 and c.get('all_owned_PID_and_groups_absent')is True,'External root reap required')
    need(c.get('protocol_SHA256')==digest and c.get('source_SHA256')==p['source_SHA256']and c.get('execution_SHA256')==stage.sha(base/'execution.json'),'Exact root closure source/protocol/execution')
    pids=[e['controller_pid'],e['worker_pid']];need(len(c.get('owned_pids',[]))==2 and set(c['owned_pids'])==set(pids)and r['worker_pid']==e['worker_pid'],'Exact two owned identities')
    for pid in pids:stage.prior.shared.old.absent(pid)
    need(e['wall_s']<=stage.LIMITS['wall_cap_s']and r['wall_s']<=stage.LIMITS['wall_cap_s']and r['peak_RSS_bytes']<=stage.LIMITS['worker_RSS_cap_bytes']and e['peak_aggregate_RSS_bytes']<=stage.LIMITS['aggregate_RSS_cap_bytes'],'Frozen time and memory limits')
    expected={(n,label)for n in stage.LEVELS for label in stage.PILOT_LOADS};rows=r['rows']
    need(len(rows)==6 and{(v['n'],v['label'])for v in rows}==expected,'All6 exact new held memberships')
    for row in rows:
        need(row.get('completed')is True and row['layout']==stage.PILOT_LAYOUT and(row['E_Pa'],row['K_N_m'])==stage.PILOT_PARAMETERS,'Exact held support/material contexts')
        expected_path=base/'raw-LOCAL'/f"n{row['n']}-{row['label']}.npz"
        need(stage.ROOT/row['capture']['path']==expected_path and row['capture']==stage.record(expected_path),'All6 capture byte commitments before first NPZ')
    need(r.get('no_noise_or_truth_draws')is True and r.get('bank_admitted')is False,'Six controls only, no draws/bank admission')
    need({SOURCE,TEST}<=set(p['source_SHA256']),'Prospectively frozen independent audit/test')
    return p,e,r,c


def geometry_forward(n):
    X,elements,_,fixed,free=stage.resolution.geometry(n)
    dm=(X[elements[:,1:4]]-X[elements[:,:1]]).transpose(0,2,1);g=np.linalg.inv(dm)
    need(np.all(np.linalg.det(dm)>0),'Positive affine element volumes')
    return SimpleNamespace(n=n,X=X,elements=elements,fixed=fixed,free=free,linear_gradients=np.concatenate([-g.sum(1,keepdims=True),g],axis=1))


def check_capture(row):
    """Loads and geometry only: never constructs K/W, factors, or solves."""
    n=row['n'];f=geometry_forward(n);loads,info=stage.bank.load_catalog(n);force=loads[row['label']];path=stage.ROOT/row['capture']['path']
    with np.load(path,allow_pickle=False)as z:
        expected={'displacement_m','linear_displacement_m','rest_m','force_N','elements','exact_constraint_nodes','query_points_m','query_displacement_m'}
        need(set(z.files)==expected and all(np.isfinite(z[k]).all()for k in expected),'Exact finite saved-field schema')
        u=z['displacement_m'].copy();uL=z['linear_displacement_m'];q=z['query_displacement_m'];points=z['query_points_m']
        need(u.shape==f.X.shape and uL.shape==f.X.shape and np.array_equal(z['rest_m'],f.X)and np.array_equal(z['elements'],f.elements)and np.array_equal(z['exact_constraint_nodes'],f.fixed),'Exact versioned geometry/topology/clamps')
        need(np.all(u[f.fixed]==0)and np.all(uL[f.fixed]==0),'Exact nonlinear and linear clamp zeros')
        need(np.array_equal(z['force_N'],force)and np.array_equal(points,stage.bank.measurement_points())and points.shape==(16,3),'Exact new force and unchanged query16')
        need(np.array_equal(stage.resolution.field_queries(n,u,points),q),'Saved query field reproduction')
    need(row['known_load']==info[row['label']],'Exact normalized Gaussian known-load metadata')
    need(row['actual_deformed_wrench']==stage.prior.force_wrench(f.X,force,u),'Actual deformed deadload wrench reproduction')
    d=stage.resolution.deformation(f,u,finest=12)
    for key in('minimum_J','maximum_absolute_symmetric_strain_eigenvalue','nonrigid_RMS_m','nonrigid_fraction'):
        need(np.isclose(d[key],row['deformation'][key],rtol=1e-11,atol=1e-14),'Saved-field sampled strain/J/volume-rigid deformation reproduction')
    return u,d


def scalar_status(row,d):
    """Independent gates from scalars; no sparse residual/eigenpair replay."""
    g=stage.metrics.GATES;eq=row['equilibrium'];s=row['stability'];need(all(finite(d.get(k))for k in('minimum_J','maximum_absolute_symmetric_strain_eigenvalue','nonrigid_RMS_m','nonrigid_fraction')),'Finite recomputed deformation metrics')
    eq_ok=all(le(eq.get(error),eq.get(tolerance,-1))for error,tolerance in(('free_residual_N','tolerance_N'),('force_balance_error_N','force_balance_tolerance_N'),('moment_balance_error_Nm','moment_balance_tolerance_Nm')))and eq.get('exact_clamp')is True
    need(eq.get('passed')is bool(eq_ok),'Recomputed reaction/equilibrium flags')
    # The generalized Ritz pair residual is a bound producer numerical witness,
    # never an independently verified global spectral enclosure.
    for key in('smallest_generalized_Ritz_ratio','rest_norm_squared','generalized_dual_residual','relative_pair_residual','dimensionless_Ritz_lower_endpoint','exact_rest_solve_residual','exact_rest_solve_tolerance','explicit_physical_pair_residual_norm'):
        need(finite(s.get(key)),'Finite generalized Ritz scalar witness')
    lam=s['smallest_generalized_Ritz_ratio'];dual=s['generalized_dual_residual'];need(dual>=0 and s['rest_norm_squared']>0 and np.isclose(s['dimensionless_Ritz_lower_endpoint'],lam-dual,rtol=1e-12,atol=1e-30)and np.isclose(s['relative_pair_residual'],dual/max(abs(lam),np.finfo(float).tiny),rtol=1e-12,atol=1e-30),'Generalized Ritz residual identities')
    need(s['explicit_physical_pair_residual_norm']>=0 and s['exact_rest_solve_tolerance']==max(1e-13,1e-11*s['explicit_physical_pair_residual_norm']),'Original rest solve tolerance')
    normal=abs(s['rest_norm_squared']-1)<=1e-6;stability=bool(normal and le(s['exact_rest_solve_residual'],s['exact_rest_solve_tolerance'])and le(s['relative_pair_residual'],1e-6)and s['dimensionless_Ritz_lower_endpoint']>0)
    need(s.get('normalization_passed')is normal and s.get('passed')is stability and s.get('matrix_modified')is False and s.get('rest_SPD_externally_assumed')is True and s.get('rigorous_global_eigenvalue_bound')is False,'Independent generalized stability predicate/scope')
    material=bool(finite(d['minimum_J'])and finite(d['maximum_absolute_symmetric_strain_eigenvalue'])and d['minimum_J']>=g['minimum_J']and d['maximum_absolute_symmetric_strain_eigenvalue']<=g['maximum_absolute_symmetric_strain_eigenvalue'])
    fraction=bool(d['nonrigid_fraction']>=g['minimum_nonrigid_fraction']);excitation=bool(d['nonrigid_RMS_m']>=g['minimum_nonrigid_RMS_m']and fraction)
    physical=bool(material and le(row['original_free_R_l2_N'],1e-10)and eq_ok and stability)
    status=dict(physical_passed=physical,excitation_goal_met=excitation,weak_prefix_control=False,load_eligible=bool(physical and fraction and excitation),bank_admitted=False)
    need(all(row.get(k)is v for k,v in status.items()),'Independent original physical/excitation eligibility flags')
    need(le(row.get('linear_original_R_l2_N'),1e-10),'Original linear warm-start residual gate')
    return status


def spatial_results(fields):
    out={};g=stage.metrics.GATES
    for label in stage.PILOT_LOADS:
        values=[(n,fields[n,label])for n in stage.LEVELS];dist=stage.resolution.common_rms(values,finest=12);a,b=float(dist[1,0]),float(dist[2,1]);need(le(a,np.inf)and le(b,np.inf),'Finite positive common-volume distances')
        order=float(np.log2(a/b))if a>0 and b>0 else None
        fine=b/(2**order-1)if order is not None and order>=g['spatial_order_floor']else None;medium=2*b
        spatial=bool(fine is not None and fine<=g['maximum_conditional_fine_indicator_m']);adequate=bool(order is not None and order>=g['spatial_order_floor']and medium<=g['maximum_medium_indicator_m'])
        query=[stage.resolution.field_queries(n,u,stage.bank.measurement_points())for n,u in values];coarse=float(np.max(abs(query[1]-query[0])));last=float(np.max(abs(query[2]-query[1])))
        out[label]=dict(coarse_medium_RMS_m=a,medium_fine_RMS_m=b,observed_order=order,conditional_fine_indicator_m=fine,medium_indicator_m=medium,spatial_passed=spatial,medium_adequate=adequate,conditional_only=True,public_point_n3_n6_max_coordinate_m=coarse,public_point_n6_n12_max_coordinate_m=last,public_point20um_passed=last<=20e-6,spatial_development_passed=bool(spatial and adequate and last<=20e-6),bank_admitted=False)
    return out

def classification(rows,spatial):
    expected={(n,label)for n in stage.LEVELS for label in stage.PILOT_LOADS}
    complete=bool(len(rows)==6 and{(v['n'],v['label'])for v in rows}==expected and all(v.get('completed')is True for v in rows))
    physical=complete and all(v['physical_passed']for v in rows);eligible=complete and all(v['load_eligible']and v['excitation_goal_met']and not v['weak_prefix_control']for v in rows)
    space=bool(set(spatial)==set(stage.PILOT_LOADS)and all(v['spatial_development_passed']for v in spatial.values()))
    return dict(all6_retained=complete,all_physical_passed=bool(physical),all_load_eligible=bool(eligible),both_spatial_passed=space,passed=bool(physical),pilot_qualified=bool(physical and eligible and space),bank_admitted=False)


def audit():
    base=stage.LOCAL;need(not(base/'independent-review.json').exists()and not(base/'compact-summary.json').exists(),'No audit overwrite or implicit replay')
    p,e,r,c=metadata_barrier();fields={};scalars=[]
    for row in r['rows']:
        u,d=check_capture(row);status=scalar_status(row,d);fields[row['n'],row['label']]=u
        scalars.append(dict(n=row['n'],label=row['label'],completed=True,**status,minimum_sampled_J=d['minimum_J'],maximum_symmetric_strain=d['maximum_absolute_symmetric_strain_eigenvalue'],nonrigid_coordinate_RMS_m=d['nonrigid_RMS_m'],nonrigid_fraction=d['nonrigid_fraction'],original_free_R_l2_N=row['original_free_R_l2_N'],force_balance_N=row['equilibrium']['force_balance_error_N'],moment_balance_Nm=row['equilibrium']['moment_balance_error_Nm'],stability_passed=row['stability']['passed'],Ritz_lower_endpoint=row['stability']['dimensionless_Ritz_lower_endpoint'],Newton_iterations=row['Newton']['iterations'],capture=row['capture']))
    spatial=spatial_results(fields);need(spatial==r['spatial'],'Saved-field commonfinest12 Keast24 spatial and query metrics')
    flags=classification(scalars,spatial);need(flags==stage.classification(scalars,spatial)and all(r[k]is v for k,v in flags.items()),'Independent all6/bothspatial classification')
    need(e['physical_passed']is flags['all_physical_passed']and e['pilot_qualified']is flags['pilot_qualified'],'External physical/pilot flags')
    compact=dict(schema='balanced-held-nh-six-control-compact-v01',completed=True,**flags,rows=scalars,spatial=spatial,wall_s=e['wall_s'],peak_RSS_bytes=r['peak_RSS_bytes'],peak_aggregate_RSS_bytes=e['peak_aggregate_RSS_bytes'],scope='Only six NEW balanced-held NH80 DEVELOPMENT controls. Not full bank/calibration/policy admission. J sampled, generalized Ritz numerical witness, conditional spatial indicators not continuum error bounds. Raw fields LOCAL.')
    stage.write(base/'compact-summary.json',compact)
    review=dict(audit_passed=True,scientific_admission=False,source_SHA256=p['source_SHA256'],audit_source=stage.record(stage.ROOT/SOURCE),audit_test=stage.record(stage.ROOT/TEST),source_result=stage.record(base/'result-LOCAL.json'),root_closure=stage.record(base/'root-closure.json'),compact_summary=stage.record(base/'compact-summary.json'),checks=dict(all6_capture_hashes_before_NPZ=True,external_root_reaped_and_owned_groups_absent=True,no_PDE_RNG_or_equilibrium_replay=True,known_force_topology_query_wrench_reproduced=True,sampled_deformation_reproduced=True,common_volume_spatial_reproduced=True,original_physical_excitation_flags_reproduced=True,sparse_residual_and_tangent_witnesses_only=True,all_negatives_retained=True))
    stage.write(base/'independent-review.json',review);return compact

if __name__=='__main__':print(json.dumps(audit()))
