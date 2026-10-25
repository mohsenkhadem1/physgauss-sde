import builtins,inspect,json,symtable
import numpy as np
import pytest
from experiments import audit_fresh64_nonlinear_calibration_v01 as a


def toy_lean():
    u=np.array([.25,.75]);q=np.array([[.05,.5,.95]]*2)
    return dict(mean=np.full(96,2e-6),SD=np.ones(96)*1e-5,unit_mean=u+np.array([.01,-.02]),unit_quantiles=q,unit_CDF=np.array([np.linspace(0,1,33)]*2),PIT=u.copy(),covered=np.ones(2,dtype=bool),intrinsic_interval_width=np.array([.9,.9]),intrinsic_signed_bias=np.array([.01,-.02]),MAP_prediction=np.zeros(96)),u,np.zeros(96)


def test_lean_metrics_recompute_field_and_material_errors_and_reject_tampering():
    z,u,target=toy_lean();m=a.lean_metrics(z,u,target)
    assert abs(m['target_RMS_m']-2e-6)<1e-20 and m['MAP_target_RMS_m']==0
    np.testing.assert_allclose(m['intrinsic_signed_bias'],[.01,-.02],atol=1e-16)
    z['covered'][0]=False
    with pytest.raises(ValueError,match='calibration field'):a.lean_metrics(z,u,target)
    z,u,target=toy_lean();z['PIT'][0]=.8
    with pytest.raises(ValueError,match='CDF bracket'):a.lean_metrics(z,u,target)
    z,u,target=toy_lean();z['PIT'][0]=np.nextafter(1.,np.inf)
    with pytest.raises(ValueError,match='Strict PIT'):a.lean_metrics(z,u,target)


def toy_objects():
    result=[]
    for i in range(64):
        posts={}
        for model in a.MODELS:
            posts[model]={}
            for key in a.KEYS:
                reset=0 if key=='prefix'else int(key[-1]);posts[model][key]=dict(qualified=True,physical_valid=True,target_RMS_m=float(1+reset),covered=[True,True],PIT=[(i+.5)/64]*2,intrinsic_interval_width=[.4,.5],intrinsic_signed_bias=[.01,-.01])
        result.append(dict(index=i,physical_valid=True,completed=True,posteriors=posts,oracle_RMS_m={m:0. for m in a.MODELS}))
    return result


def test_missing_entire_object_keeps64_denominator_and_exact896_slots():
    records=toy_objects();records.pop(17);rows=a.planned_records(records)
    assert len(rows)==64 and rows[17]['missing'] and sum(len(v)for r in rows for v in r['posteriors'].values())==896
    cal,pred,primary=a.aggregate(records)
    assert cal['comparison_blocked']and pred['comparison_blocked']
    row=cal['nonlinear']['entries'][0]
    assert row['object_count']==64 and row['covered']==63 and row['unresolved_coverage']==1
    assert row['coverage_bounds']==[63/64,1.]and row['intervals']is None
    assert np.isnan(primary['nonlinear_PIT'][17]).all()
    with pytest.raises(ValueError,match='Duplicate'):a.planned_records(records+[records[0]])


def test_reset_means_stay_within_objects_shared_bootstrap_has64_clusters(monkeypatch):
    monkeypatch.setattr(a.design,'BOOTSTRAP_SEED',123)
    cal,pred,_=a.aggregate(toy_objects())
    b=pred['shared_bootstrap'];assert b['object_clusters']==64 and b['same_object_resamples_across_all_metrics']
    entries={r['label']:r for r in b['entries']}
    assert entries['nonlinear_normal_reset0_RMS_m']['mean']==1.
    assert entries['nonlinear_normal_mean3_RMS_m']['mean']==2.
    assert entries['nonlinear_prefix_minus_normal_mean3_RMS_m']['mean']==-1.
    assert cal['nonlinear']['object_clusters']==64 and not cal['comparison_blocked']


def test_early_terminal_seal_rejection_precedes_any_array_read(monkeypatch,tmp_path):
    monkeypatch.setattr(a.fresh,'LOCAL',tmp_path);monkeypatch.setattr(a.fresh,'release_gate',lambda _:dict(models=list(a.MODELS),phases=list(a.design.PHASES),truth_labels=list(a.design.TRUTH_LABELS),target_labels=list(a.design.TARGETS)))
    (tmp_path/'execution.json').write_text('{}');(tmp_path/'summary-LOCAL.json').write_text('{}')
    monkeypatch.setattr(a.np,'load',lambda *x,**k:pytest.fail('Premature actual archive read'))
    with pytest.raises(ValueError,match='Exact terminal seals'):a.terminal_metadata('unreleased','wrong','wrong')


def test_commit_context_and_bytes_cannot_be_swapped(tmp_path):
    p=tmp_path/'draws.npz';p.write_bytes(b'synthetic archive placeholder');c=p.with_suffix('.commit.json')
    c.write_text(json.dumps(dict(context={'index':2},raw_SHA256=a.sha(p))))
    manifest={str(p):a.sha(p),str(c):a.sha(c)}
    a.commit(p,{'index':2},manifest)
    with pytest.raises(ValueError,match='context'):a.commit(p,{'index':3},manifest)
    p.write_bytes(b'changed bytes')
    with pytest.raises(ValueError,match='committed'):a.commit(p,{'index':2},manifest)
    c.write_text(json.dumps(dict(context={'index':2},raw_SHA256=a.sha(p))))
    with pytest.raises(ValueError,match='committed'):a.commit(p,{'index':2},manifest)


def test_full_truth_field_and_queries_checked_on_manufactured_quadratic():
    from experiments.campaign_surface_p2_static_v02 import geometry
    from experiments.campaign_surface_p2_study_design_v01 import observation_points
    X,_,_,_,_=geometry(1);u=np.zeros_like(X);u[:,2]=X[:,0]*(.36-X[:,0])*.001
    points=observation_points();response=np.zeros_like(points);response[:,2]=points[:,0]*(.36-points[:,0])*.001
    z=dict(unit=np.array([.2,.3]),displacement_m=u,linear_displacement_m=u.copy(),response_m=response.ravel())
    a.truth_array_checks(z,z['unit'],n=1)
    z['response_m'][0]+=2e-9
    with pytest.raises(ValueError,match='query/field'):a.truth_array_checks(z,z['unit'],n=1)
    z['response_m'][0]-=2e-9;z['displacement_m'][0,0]=1e-8
    with pytest.raises(ValueError,match='fixed nodes'):a.truth_array_checks(z,z['unit'],n=1)


def test_audit_implementation_never_replays_model_or_parameter_noise_draws():
    for fn in(a.audit,a.object_arrays,a.terminal_metadata,a.truth_array_checks,a.numerical_record_checks,a.canonical_modal_identity):
        text=inspect.getsource(fn)
        assert 'default_rng'not in text and '.qualify('not in text and '.solve('not in text and '.cdf('not in text
        table=symtable.symtable(text,'fresh64audit','exec')
        def check(t):
            for s in t.get_symbols():
                if s.is_referenced()and s.is_global():assert s.get_name()in vars(a)or hasattr(builtins,s.get_name()),s.get_name()
            for c in t.get_children():check(c)
        check(table)


def numerical_record():
    rows=[]
    for coarse,fine in((128,256),(256,512)):
        rows.append(dict(coarse=coarse,fine=fine,passed=True,log_evidence_difference=0.,maximum_CDF_difference=0.,maximum_unit_quantile_difference=0.,maximum_predictive_mean_difference_m=1e-9,maximum_predictive_SD_difference_m=1e-8,maximum_unit_mean_difference=0.))
    adaptive=dict(passed=True,epsrel=[1e-6,1e-8],CDF_checks_per_epsrel=6,log_evidence_difference=0.,maximum_CDF_error=0.,adaptive_refinement_log_evidence_difference=0.,adaptive_refinement_maximum_CDF_difference=0.,maximum_QUADPACK_relative_error_estimates={'1e-06':1e-7,'1e-08':1e-9})
    import copy
    d=dict(schema='nonlinear-conditional-reference-v03',qualified=True,independent_adaptive_per_posterior_required=True,gates=dict(logZ=.01,CDF=.005,quantile=.002,unit_mean=.001,target_absolute_m=2e-6,target_relative_SD=.01),successive={'inner':copy.deepcopy(rows),'outer':copy.deepcopy(rows)},levels=[dict(outer=o,inner=i,status='computed',log_evidence=0.,weighted_hypotheses=100)for o,i in((512,128),(512,256),(512,512),(128,512),(256,512))],independent_adaptive=adaptive)
    return dict(diagnostics=d,strict_absolute_checks=a.fresh.cost.absolute_engine_witnesses(dict(qualified=True,diagnostics=d)))


def test_original_gate_and_order_plan_tampering_rejected_coarse_negatives_retained():
    r=numerical_record();assert a.numerical_record_checks(r,'nonlinear')
    r['diagnostics']['gates']['CDF']=.5
    with pytest.raises(ValueError,match='original numerical gates'):a.numerical_record_checks(r,'nonlinear')
    r=numerical_record();r['diagnostics']['successive']['inner'][-1]['fine']=256
    with pytest.raises(ValueError,match='order pairs'):a.numerical_record_checks(r,'nonlinear')
    r=numerical_record();r['diagnostics']['independent_adaptive_per_posterior_required']=False
    with pytest.raises(ValueError,match='adaptive declaration'):a.numerical_record_checks(r,'nonlinear')
    r=numerical_record();r['diagnostics']['levels'][0]=dict(outer=512,inner=128,status='unresolved',reason='Preserved coarse failure');r['diagnostics']['successive']['inner'][0]=dict(coarse=128,fine=256,passed=False,reason='Missing unresolved level')
    assert a.numerical_record_checks(r,'nonlinear')


def test_modal_identity_is_anchored_to_qualified_cost_not_self_consistency():
    p={'cost_receipt':{'memo':{'model_fingerprint':'a'*64}}}
    assert a.canonical_modal_identity('a'*64,p)=='a'*64
    # All objects could agree with each other yet still use a different model.
    with pytest.raises(ValueError,match='Canonical'):a.canonical_modal_identity('b'*64,p)
