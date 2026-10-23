"""Only supplied manufactured CDF tuples and temporary metadata files."""
import json
import numpy as np
import pytest
from experiments import diagnose_resolved_CDF_GL128_v06 as m


def test_trace_preserves_raw_tuple_identity_and_classifies_central_boundary_ulp():
 row=dict(cdf=np.nextafter(1.,np.inf),lower=1.-1e-13,upper=1.);trace=dict(last_queries=[],query_count=0,offending_queries=[]);query=m.trace_query(lambda x:row,trace)
 assert query(.1)is row and trace['offending_queries'][0]['raw']==row and trace['query_count']==1
 assert trace['offending_queries'][0]['conditions']==dict(finite=True,bounds_in_unit_interval=True,central_above_lower=True,central_below_upper=False)
 assert row['cdf']>1. and row['upper']==1.


@pytest.mark.parametrize('row,condition',[(dict(cdf=.5,lower=.6,upper=.7),'central_above_lower'),(dict(cdf=.5,lower=.7,upper=.6),'bounds_in_unit_interval'),(dict(cdf=float('nan'),lower=0.,upper=1.),'finite')])
def test_distinguishes_other_invalid_conditions_without_projection(row,condition):
 trace=dict(last_queries=[],query_count=0,offending_queries=[]);q=m.trace_query(lambda x:row,trace);assert q(.2)is row
 assert trace['offending_queries'][0]['conditions'][condition]is False
 json.dumps(trace,allow_nan=False)


def test_exact_v01_guard_failure_is_logged_not_repaired_or_retried(monkeypatch):
 monkeypatch.setattr(m,'_global_target_bounds',lambda*a:(0.,1.));row=dict(cdf=np.nextafter(1.,np.inf),lower=np.nextafter(1.,np.inf),upper=1.);monkeypatch.setattr(m.quantile,'polynomial_support_cdf_bounds',lambda*a,**k:row);trace=dict(last_queries=[],query_count=0,offending_queries=[])
 with pytest.raises(FloatingPointError,match='Finite ordered numerical CDF enclosure required'):m.trace_coordinate(type('Post',(),{'coordinate_cdf':lambda *a,**k:{},'Z':1.,'Z_error_estimate':0.,'categories':['a']})(),{'a':np.array([0.,1.,0.])},.5,trace)
 assert trace['query_count']==1 and len(trace['offending_queries'])==1 and row['cdf']>1.


def test_prepare_is_metadata_only_and_binds_exact_failed_linear_prefix(tmp_path,monkeypatch):
 def put(n,v):
  p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v));return m.run.io.record(tmp_path,p)
 put('source.py',{});monkeypatch.setattr(m,'sources',lambda:['source.py']);numeric=put('numerics.json',dict(failures=[dict(method='GL128',stage='CANONICAL_QUANTILES',error="FloatingPointError('Finite ordered numerical CDF enclosure required')")]));files={'numerics.json':numeric};r=put('failed.json',{});p=dict(output='old',inputs=dict(jobs=[dict(id='one__prefix',kind='prefix')]));put('old/execution.json',dict(records=[dict(job_id='one__prefix',completed=True,reaped=True,result=r)]));monkeypatch.setattr(m.run,'validate',lambda*a,**k:p);monkeypatch.setattr(m.run,'output_result',lambda*a:dict(models={'linear_rest_tangent':dict(numerical_controls_passed=False,files=files)}));monkeypatch.setattr(np,'load',lambda*a,**k:(_ for _ in()).throw(AssertionError('actual arrays during prepare')))
 put('runs/gaussian-campaign-v01/resolved-support-repair-v03/objects/one__prefix__linear_rest_tangent/numerics.json',dict(failures=[dict(method='GL128',stage='CANONICAL_QUANTILES')]))
 digest=m.prepare('p.json','out',resolved_protocol='resolved',resolved_protocol_sha256='a'*64,failed_job_id='one__prefix',root=tmp_path);recipe=json.loads((tmp_path/'p.json').read_text());assert recipe['only_method']=='GL128'and recipe['only_model']=='linear_rest_tangent'and recipe['only_v02_guard_calls']and not(tmp_path/'out').exists()
 assert m.run.io.sha(tmp_path/'p.json')==digest
 with pytest.raises(FileExistsError):m.prepare('p.json','out',resolved_protocol='resolved',resolved_protocol_sha256='a'*64,failed_job_id='one__prefix',root=tmp_path)


def test_real_GL128_constructor_startup_uses_legendre_API():
 from experiments.campaign_c38_exact_posterior_memo import BoundsMemoLikelihood
 model=BoundsMemoLikelihood(np.zeros((3,1)),np.zeros(1),np.eye(1),young_bounds=(75000.,150000.),young_reference_Pa=75000.,root_digits=50,root_log_energy_tolerance=1e-12,log_drops=(1.,4.,16.,64.))
 post=m.prefix.PolynomialPosterior({'a':model},method='legendre',order=128)
 assert post.numerics['method']=='legendre' and post.numerics['order']==128 and abs(post.cdf(100000.)-np.log(4/3)/np.log(2))<1e-12
 import inspect
 assert "models,method='legendre',order=128" in inspect.getsource(m.execute)


def test_real_category_accounting_exact_full_probability_and_partial_mass():
 from experiments.campaign_c38_exact_posterior_memo import BoundsMemoLikelihood
 def model():return BoundsMemoLikelihood(np.zeros((3,1)),np.zeros(1),np.eye(1),young_bounds=(75000.,150000.),young_reference_Pa=75000.,root_digits=50,root_log_energy_tolerance=1e-12,log_drops=(1.,4.,16.,64.))
 post=m.prefix.PolynomialPosterior({'a':model(),'b':model()},method='legendre',order=128)
 targets={c:np.array([0.,1.,0.])for c in post.categories}
 full=m.category_accounting(post,targets,2.)
 assert full['exact_ratio_float']==1. and not full['exact_ratio_above_one']
 assert all(r['active_is_full_domain']for r in full['categories'])
 partial=m.category_accounting(post,targets,.75)
 assert 0<partial['exact_ratio_float']<1 and all(not r['active_is_full_domain']for r in partial['categories'])
