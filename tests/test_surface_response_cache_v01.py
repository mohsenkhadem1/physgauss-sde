"""Tiny synthetic exact-cache equivalence, admission and corruption controls."""
import hashlib
import json
import numpy as np
import pytest
from experiments.campaign_surface_p2_inference_v01 import LabelledModal,IntegrationPlan,infer
from experiments.campaign_surface_response_cache_v01 import build_cache,CachedLabelledModal,model_fingerprint
from experiments.campaign_surface_inverse_reference_v01 import ModalStaticReference


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path):
    modal=ModalStaticReference(np.diag([1.,3.]),np.diag([.1,.7]),
        np.array([[.01,.02],[.03,.04]]),np.array([[1.,2.],[3.,4.],[5.,6.]]))
    model=LabelledModal(modal,['A','B'])
    receipt=tmp_path/'admission.json'
    receipt.write_text(json.dumps(dict(cache_admission=True,scope='synthetic_controls',
        model_fingerprint=model_fingerprint(model),protocol_SHA256='a'*64)))
    plan=IntegrationPlan((9,17,33),(8,16),13)
    path=tmp_path/'cache';manifest=build_cache(model,path,admission_receipt=receipt,admission_SHA256=sha(receipt),plan=plan)
    cached=CachedLabelledModal(model,path,manifest_SHA256=sha(path/'manifest.json'))
    return model,cached,path,receipt,manifest


def test_dyadic_gauss_offgrid_and_duplicate_data_equivalence(tmp_path):
    model,cached,path,receipt,manifest=fixture(tmp_path)
    chunks=[]
    for n in (9,17,33):
        axis=np.linspace(0,1,n);chunks.append(np.stack(np.meshgrid(axis,axis,indexing='ij'),-1).reshape(-1,2))
    for n in (8,16):
        axis=(np.polynomial.legendre.leggauss(n)[0]+1)/2
        chunks.append(np.stack(np.meshgrid(axis,axis,indexing='ij'),-1).reshape(-1,2))
    # One ULP change must take exact modal fallback, never parameter snapping.
    chunks.append(np.array([[np.nextafter(.5,1),.5],[.123456789,.731234567],[0.,1.]]))
    for unit in chunks:
        assert np.allclose(cached.select(['B','A'])(unit),model.select(['B','A'])(unit),rtol=1e-14,atol=1e-18)
    obs=[('A',np.array([2,4,6])*1e-6),('A',np.array([1,3,5])*1e-6),('B',np.zeros(3))]
    assert np.allclose(cached.loglikelihood(obs)(chunks[-1]),model.loglikelihood(obs)(chunks[-1]),rtol=1e-14,atol=1e-12)
    # A delegated selection spy proves one-ULP/offgrid uses the supplied
    # exact adapter (including a future reviewed GPU adapter), not table snap.
    calls=[]
    original=cached._fallback_model.select
    def spy(labels):
        exact=original(labels)
        def evaluate(unit):
            calls.append(unit.copy());return exact(unit)
        return evaluate
    cached._fallback_model.select=spy
    off=chunks[-1][:2]
    assert np.allclose(cached.select(['A'])(off),model.select(['A'])(off),rtol=1e-14,atol=1e-18)
    assert len(calls)==2  # cached fallback and direct comparison evaluation
    calls.clear();cached.select(['A'])(np.array([[.5,.5]]))
    assert calls==[]
    assert manifest['projected_table_bytes']==(33**2+8**2+16**2)*2*3*8
    assert manifest['interpolation'] is False


def test_admission_mismatch_and_large_synthetic_refusal(tmp_path):
    model,cached,path,receipt,manifest=fixture(tmp_path)
    with pytest.raises(ValueError,match='receipt SHA'):
        build_cache(model,tmp_path/'bad',admission_receipt=receipt,admission_SHA256='0'*64)
    with pytest.raises(ValueError,match='actual study'):
        build_cache(model,tmp_path/'large',admission_receipt=receipt,admission_SHA256=sha(receipt))
    model.modal.coefficients=model.modal.coefficients.copy()*2
    with pytest.raises(ValueError,match='fingerprint'):
        build_cache(model,tmp_path/'wrong',admission_receipt=receipt,admission_SHA256=sha(receipt),plan=IntegrationPlan((9,17),(8,16)))


def test_corrupt_table_and_incomplete_manifest_refusal(tmp_path):
    model,cached,path,receipt,manifest=fixture(tmp_path)
    file=path/manifest['tables'][0]['path']
    with file.open('r+b') as handle:handle.seek(-8,2);handle.write(b'00000000')
    with pytest.raises(ValueError,match='hash mismatch'):
        CachedLabelledModal(model,path,manifest_SHA256=sha(path/'manifest.json'))
    manifest['status']='INCOMPLETE';(path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='Complete'):
        CachedLabelledModal(model,path,manifest_SHA256=sha(path/'manifest.json'))


def test_toy_cached_and_direct_integrated_posterior_agreement(tmp_path):
    model,cached,path,receipt,manifest=fixture(tmp_path)
    observations=[('A',np.array([4.,9.,14.])*1e-6),('B',np.array([5.,11.,17.])*1e-6)]
    plan=IntegrationPlan((9,17,33),(8,16),13)
    direct=infer(model,observations,['B'],plan=plan)
    saved=infer(cached,observations,['B'],plan=plan)
    assert saved['qualified']==direct['qualified']
    for key in ('predictive_mean','predictive_SD'):
        assert np.allclose(saved[key],direct[key],rtol=1e-13,atol=1e-18)
    for key in ('unit_mean','unit_quantiles','quadrature_probability'):
        assert np.allclose(saved['local_final_simpson'][key],direct['local_final_simpson'][key],rtol=1e-12,atol=1e-14)


def test_resource_preflight_refuses_before_table_allocation(tmp_path,monkeypatch):
    import experiments.campaign_surface_response_cache_v01 as cache
    from types import SimpleNamespace
    model,cached,path,receipt,manifest=fixture(tmp_path)
    plan=IntegrationPlan((9,17),(8,16),13)
    monkeypatch.setattr(cache,'_rss',lambda:cache.RSS_LIMIT_BYTES)
    with pytest.raises(MemoryError):
        build_cache(model,tmp_path/'RSS-refused',admission_receipt=receipt,admission_SHA256=sha(receipt),plan=plan)
    assert not list((tmp_path/'RSS-refused').glob('*.npy'))
    with pytest.raises(MemoryError):cached.select(['A'])(np.array([[.5,.5]]))
    monkeypatch.setattr(cache,'_rss',lambda:1024**2)
    monkeypatch.setattr(cache.shutil,'disk_usage',lambda _:SimpleNamespace(free=cache.DISK_RESERVE_BYTES))
    with pytest.raises(OSError,match='disk reserve'):
        build_cache(model,tmp_path/'disk-refused',admission_receipt=receipt,admission_SHA256=sha(receipt),plan=plan)
    assert not list((tmp_path/'disk-refused').glob('*.npy'))


def test_reopen_uses_stored_axes_not_regenerated_library_nodes(tmp_path,monkeypatch):
    import experiments.campaign_surface_response_cache_v01 as cache
    model,cached,path,receipt,manifest=fixture(tmp_path)
    axis=(np.polynomial.legendre.leggauss(16)[0]+1)/2
    unit=np.array([[axis[3],axis[5]]])
    expected=model.select(['B'])(unit)
    def wrong_generation(n):raise AssertionError('Reopen must not regenerate Gauss nodes')
    monkeypatch.setattr(cache.np.polynomial.legendre,'leggauss',wrong_generation)
    reopened=CachedLabelledModal(model,path,manifest_SHA256=sha(path/'manifest.json'))
    assert np.allclose(reopened.select(['B'])(unit),expected,rtol=1e-14,atol=1e-18)
    axispath=path/manifest['tables'][1]['axis_path']
    np.save(axispath,np.linspace(0,1,8))
    with pytest.raises(ValueError,match='axis path/hash'):
        CachedLabelledModal(model,path,manifest_SHA256=sha(path/'manifest.json'))
