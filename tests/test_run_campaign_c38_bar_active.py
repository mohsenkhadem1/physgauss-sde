import hashlib
import json
import time
import numpy as np
import pytest
from experiments import run_campaign_c38_bar_active as study
from experiments.campaign_c38_bar_posterior import BarPosteriorBatch


def test_synthetic_continuous_summary_independent_2d_and_cloud():
    # Unit-control datum, not any of the frozen study's future random objects.
    t=time.monotonic();p=BarPosteriorBatch([[.005]],[1.],[0],rtol=1e-8,conditional_order=64);report=study.summarize(p)
    answer=study.independent(p,0,report)
    assert answer['passed'] and answer['maximum_marginal_CDF_error']<5e-4
    assert (report['std']>0).all() and np.isfinite(report['field_std']).all()
    assert report['quantiles']['E'][0,0]<5000<report['quantiles']['E'][0,2]
    assert time.monotonic()-t<30


def test_prepare_metadata_only_pins_and_scalar_json(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'ROOT',tmp_path);monkeypatch.setattr(study,'SOURCES',['source.py']);(tmp_path/'source.py').write_text('# synthetic source')
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):monkeypatch.setenv(name,'2')
    p=tmp_path/'protocol.json';digest=study.prepare(p,tmp_path/'out')
    assert not (tmp_path/'out').exists()
    result=study.validate(p,digest);assert result['observations_generated'] is False
    assert 'NOT polynomial stable-NH' in result['constitutive_law']
    (tmp_path/'source.py').write_text('# changed')
    with pytest.raises(ValueError,match='source'):study.validate(p,digest)
    study.write(tmp_path/'numpy.json',dict(f=np.float64(.1),b=np.bool_(False),a=np.array([1,2])))
    assert json.loads((tmp_path/'numpy.json').read_text())=={'f':.1,'b':False,'a':[1,2]}


def test_exact_binomial_edges_and_reset_metadata():
    assert study.ci([True]*64)['exact_binomial_95percent_interval'][1]==1
    assert study.ci([False]*64)['exact_binomial_95percent_interval'][0]==0
    actions=study.actions();assert len(actions)==3
    assert len(set(a['target_history_sha256'] for a in actions))==1
    assert all(a['reset_to_common_rest'] for a in actions)
