"""Analytic corrected-library/GP tests, with no real library or cohort access."""
import hashlib,json
from types import MappingProxyType
import numpy as np
import pytest
from experiments import campaign_rotation_posterior as posterior
from experiments.campaign_joint_gaussian_discrepancy import JointGaussianDiscrepancy
from experiments.censored_range_observation import CensoredRanges


def arrays(*,offset=False,inverted=False):
    p={4:.8,9:.2};ids=np.repeat([4,9],3).astype(np.int64)
    physical=np.tile([5000.,625.,.35,.6],(6,1))
    visible=np.zeros((6,72),bool);visible[:,0]=True
    ranges=np.zeros((6,72));ranges[ids==9,0]=.0004 if offset else 0.
    minimum=np.ones(6);inversions=np.zeros(6,dtype=np.int64)
    if inverted:minimum[-1]=-.1;inversions[-1]=2
    data=dict(geometry_ids=ids,physical_coordinates=physical,predicted_visible=visible,
        predicted_residual_m=ranges,prefix_indices_256=np.array([0,3],dtype=np.int64),
        prior_mass_1024=np.repeat([.8/3,.2/3],3),prior_mass_256=np.array([.8,.2]),
        minimum_J=minimum,nonpositive_tet_frame_count=inversions)
    return data,p


def libraries(*,offset=False):
    data,p=arrays(offset=offset)
    refs=[]
    for seed in posterior.SEEDS:
        values=posterior.validate_corrected_arrays(data,p,count=3,prefix=1)
        refs.append(posterior.CorrectedReference(seed,*values))
    return posterior.CorrectedLibraries(tuple(refs),MappingProxyType(p),(1,3),
        {'path':'SYNTHETIC-summary','sha256':'synthetic'},{'path':'SYNTHETIC-plan','sha256':'synthetic'})


def model():
    loading=np.zeros((73,1));loading[[0,72],0]=.0003
    return JointGaussianDiscrepancy(np.zeros(73),loading,np.zeros(1),np.array([[.0001]]))


def observed(y=0.):
    visible=np.zeros(72,bool);visible[0]=True
    residual=np.zeros(72);residual[0]=y
    return CensoredRanges(visible,residual)


def test_constant_marginal_likelihood_recovers_prior_for_all3_seeds_both_counts():
    result=posterior.condition_rotation_reference(libraries(),observed(),model())
    assert len(result['seeds'])==3 and len(result['summaries'])==6
    assert [(r['seed'],r['count_per_mask']) for r in result['summaries']]==[
        (seed,count) for seed in posterior.SEEDS for count in [1,3]]
    for seed in result['seeds']:
        for prefix in seed['prefixes']:
            count=prefix['count_per_mask']
            np.testing.assert_allclose(prefix['weights'],np.repeat([.8/count,.2/count],count))
            np.testing.assert_allclose([c['posterior_probability'] for c in prefix['summary']['support_categories']],[.8,.2])
    assert result['stability']['maximum_independent_seed_support_total_variation']==0
    assert result['old_likelihood_or_posterior_reused'] is False


def test_analytic_integrated_gaussian_bayes_factor_and_fixed_shared_latent_indexing():
    libs=libraries(offset=True);gp=model();obs=observed(.0001)
    result=posterior.condition_rotation_reference(libs,obs,gp)
    variance=.0002**2+.0003**2
    local=np.exp(-.5*np.array([.0001**2,(.0001-.0004)**2])/variance)
    category=np.array([.8,.2])*local;category/=category.sum()
    weights=np.repeat(category/3,3)
    np.testing.assert_allclose(result['seeds'][0]['prefixes'][-1]['weights'],weights,rtol=2e-14)
    direct=gp.condition(CensoredRanges(libs.references[0].predicted_visible,
        libs.references[0].predicted_residual_m),obs,rows=np.arange(72))
    empirical=result['empirical_posterior'];conditional=empirical['base_gp_conditional']
    assert conditional is result['primary_base_gp_conditional']
    assert empirical['reference_seed']==20261112 and empirical['resampling_seed']==2026120700
    selected=posterior.systematic_resample(weights,2048,2026120700)
    np.testing.assert_array_equal(empirical['reference_indices'],selected)
    np.testing.assert_array_equal(empirical['geometry_ids'],libs.references[0].geometry_ids[selected])
    np.testing.assert_array_equal(conditional.latent_mean,direct.latent_mean[selected])
    np.testing.assert_array_equal(conditional.group_index,direct.group_index[selected])
    assert conditional.observation_rows==tuple(range(72))
    np.testing.assert_allclose(conditional.latent_mean[:,0],.0003/variance*(.0001-libs.references[0].predicted_residual_m[selected,0]))
    assert len(np.unique(selected))<len(selected)  # repeated particles retain aligned z


def test_after_resampling_extra_view_uses_same_z_and_joint_over_base_ratio():
    gp=model();result=posterior.condition_rotation_reference(libraries(offset=True),observed(.0001),gp)
    base=result['empirical_posterior']['base_gp_conditional']
    view=CensoredRanges(np.ones((2048,1),bool),np.zeros((2048,1)))
    obs=CensoredRanges(np.ones(1,bool),np.array([-.00015]))
    joint,ratio=gp.condition_extra_view(base,view,obs,rows=np.array([72]))
    full_prediction=CensoredRanges(np.concatenate([base.predictions.visible,view.visible],1),
        np.concatenate([base.predictions.residual_m,view.residual_m],1))
    full_obs=CensoredRanges(np.concatenate([base.observation.visible,obs.visible]),
        np.concatenate([base.observation.residual_m,obs.residual_m]))
    direct=gp.condition(full_prediction,full_obs,rows=np.arange(73))
    np.testing.assert_allclose(joint.latent_mean,direct.latent_mean,atol=0,rtol=0)
    np.testing.assert_allclose(ratio,direct.log_likelihood-base.log_likelihood,atol=0,rtol=0)
    independent=gp.condition(view,obs,rows=np.array([72]))
    assert np.max(np.abs(ratio-independent.log_likelihood))>.01


def test_corrected_member_allowlist_prior_mass_and_inversions_retained_without_filtering():
    data,p=arrays(inverted=True);accessed=[]
    class Archive:
        def __getitem__(self,key):
            accessed.append(key)
            if key in ('weights','log_likelihood'):raise AssertionError('Old case weights/likelihood forbidden')
            return data[key]
    values=posterior.validate_corrected_arrays(Archive(),p,count=3,prefix=1)
    assert set(accessed)==set(data) and len(values[0])==6
    assert values[4][-1]==-.1 and values[5][-1]==2
    assert not values[0].flags.writeable
    data['prior_mass_1024']=data['prior_mass_1024'].copy();data['prior_mass_1024'][0]+=.01
    with pytest.raises(ValueError,match='Exact nested prior mass'):
        posterior.validate_corrected_arrays(data,p,count=3,prefix=1)


def metadata_fixture(root):
    def write(name,value):
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value));return dict(path=name,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    source=write('source.json',{'synthetic':True});sources={source['path']:source['sha256']}
    prior=write('prior.json',dict(schema='point-pin-normalized-prior-pushforward-v0.1',boundary_index=0,
        conditioned_on_BC=True,probability_renormalized=False,positive_area_dropped=False,total_probability_exact='1',
        masks=[dict(node_id=i,probability=1./55) for i in range(55)]))
    hardware=write('hardware.json',{});manifest=write('public.json',{});queries=write('queries.json',{})
    cache=write('cache.json',dict(completed=True,source_sha256=sources,
        categories=[dict(node_id=i) for i in range(55)],public_hardware=hardware,distribution=prior))
    plan=write('plan.json',dict(schema='point-rotation-observation-independent-prior-library-v0.2',
        seeds=list(posterior.SEEDS),counts_per_mask=[256,1024],mask_count=55,particles_per_seed=55*1024,
        total_particles=3*55*1024,allowed_old_archive_members=['geometry_ids','physical_coordinates'],
        observations_or_cohort_truth_used=False,likelihood_or_posterior_computed=False,source_sha256=sources,
        output='library',cache=cache,public_manifest=manifest,public_hardware=hardware,public_queries=queries,support_prior=prior))
    directory=root/'library';directory.mkdir()
    records=[]
    for seed in posterior.SEEDS:
        path=directory/f'corrected-prior-{seed}.npz';path.write_bytes(b'NEVER OPEN THIS INVALID ARCHIVE')
        records.append(dict(seed=seed,library=dict(path=str(path.relative_to(root)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
            particles=55*1024,categories_dropped=[]))
    write('library/execution.json',dict(terminal=True,completed=True))
    summary=write('library/summary.json',dict(status='COMPLETED_ALL3_CORRECTED_OBSERVATION_INDEPENDENT_PRIOR_LIBRARIES',
        plan=plan,records=records,source_sha256=sources,old_archive_members_read=['geometry_ids','physical_coordinates'],
        old_ray_or_likelihood_or_posterior_members_read=False,cohort_or_truth_used=False,posterior_computed=False))
    return summary


def test_all3_archive_hashes_checked_before_first_archive_member_is_opened(tmp_path):
    summary=metadata_fixture(tmp_path)
    (tmp_path/f'library/corrected-prior-{posterior.SEEDS[-1]}.npz').write_bytes(b'CHANGED')
    # First two files are deliberately invalid NPZs. Reaching their members
    # would fail with a ZIP error instead of the required third-file hash error.
    with pytest.raises(ValueError,match='Frozen library artifact changed'):
        posterior.load_rotation_libraries(summary['path'],summary['sha256'],root=tmp_path)


def test_nonterminal_corrected_library_is_rejected_before_archive_load(tmp_path):
    summary=metadata_fixture(tmp_path)
    (tmp_path/'library/execution.json').write_text(json.dumps(dict(terminal=False,completed=False)))
    with pytest.raises(ValueError,match='not terminal complete'):
        posterior.load_rotation_libraries(summary['path'],summary['sha256'],root=tmp_path)


def test_one72ray_observation_and_all3_ordered_seeds_required():
    libs=libraries()
    with pytest.raises(ValueError,match='acquired72ray'):
        posterior.condition_rotation_reference(libs,CensoredRanges(np.ones(1,bool),np.zeros(1)),model())
    incomplete=posterior.CorrectedLibraries(libs.references[:2],libs.probabilities,libs.counts_per_mask,{}, {})
    with pytest.raises(ValueError,match='All3 fixed ordered seeds'):
        posterior.condition_rotation_reference(incomplete,observed(),model())


def test_public_hardware_identity_accepts_same_checked_bytes_at_different_paths(tmp_path):
    value=json.dumps({'rest_points_m':[[0.,0.,0.]],'handles':[{'mass_kg':.01}]}).encode()
    records=[]
    for name in ('cache-public-H0.json','cohort-public-H0.json'):
        (tmp_path/name).write_bytes(value)
        records.append(dict(path=name,sha256=hashlib.sha256(value).hexdigest()))
    assert records[0]!=records[1]
    assert posterior._same_public_hardware(tmp_path,*records) is True


def test_public_hardware_identity_rejects_different_valid_bytes_and_corruption(tmp_path):
    records=[]
    for name,mass in [('cache-H0.json',.01),('cohort-H0.json',.02)]:
        value=json.dumps({'handles':[{'mass_kg':mass}]}).encode()
        (tmp_path/name).write_bytes(value)
        records.append(dict(path=name,sha256=hashlib.sha256(value).hexdigest()))
    assert posterior._same_public_hardware(tmp_path,*records) is False
    # Identical claimed hashes never allow unchecked bytes or a copied record
    # to rescue a physically different hardware file.
    forged={**records[1],'sha256':records[0]['sha256']}
    with pytest.raises(ValueError,match='Frozen library artifact changed'):
        posterior._same_public_hardware(tmp_path,records[0],forged)
