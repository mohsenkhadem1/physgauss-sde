"""Manufactured metadata, fake entropy and fake noise ONLY; no actual draws."""
import copy,json,hashlib,os
from pathlib import Path
import numpy as np
import pytest
from experiments import run_campaign_c38_controlled_observations as m
from test_campaign_c38_controlled_observations import FakeFactory,fake_bound,design


def record(root,p):return dict(path=str(p.relative_to(root)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def setup(tmp,monkeypatch):
    original=m.ROOT
    for name in m.sources()+[m.obs.DESIGN_PATH]:
        p=tmp/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((original/name).read_bytes())
    a=tmp/'candidate.npz';a.write_bytes(b'fake candidate bytes; prepare must never extract')
    d=tmp/'direct.npz';d.write_bytes(b'fake direct bytes; loaders replaced during pure run controls')
    validation=tmp/'validation.json';validation.write_text(json.dumps(dict(completed=True,validation=dict(finite_checks_passed=True,tolerance_m=1e-6),archives=dict(TABLE_NODE=record(tmp,a),WITHHELD_INTERPOLATION_VALIDATION=record(tmp,d)))))
    controls=tmp/'controls.json';controls.write_text(json.dumps(dict(completed=True,all_numerical_controls_passed=True,design_sha256=m.obs.DESIGN_SHA256,point_table_validation_sha256=record(tmp,validation)['sha256'],candidate_archive_sha256=record(tmp,a)['sha256'])))
    return validation,controls


def test_public_codec_roundtrip_strict_scalars_arrays_and_no_private_members(tmp_path):
    payload=m.obs.public_prefix_payload('a'*32,np.zeros((9,3)),design=design());encoded=m.encode_public(payload,design=design());path=tmp_path/'prefix.npz';m.exclusive_npz(path,encoded)
    decoded=m.load_public_prefix(tmp_path,record(tmp_path,path),design=design())
    assert decoded['is_RGBD_perception'] is False and type(decoded['object_id']) is str and decoded['known_density_kg_m3']==625.
    assert set(decoded)==m.obs.PUBLIC_KEYS
    bad=copy.deepcopy(encoded);bad['is_RGBD_perception']=np.array(0,dtype=np.int64)
    with pytest.raises(ValueError,match='Scalar boolean'):m.decode_public_members(bad,design=design())
    bad=copy.deepcopy(encoded);bad['object_id']=np.array(['a'*32])
    with pytest.raises(ValueError,match='Scalar Unicode'):m.decode_public_members(bad,design=design())
    bad=copy.deepcopy(encoded);bad['observed_positions_m']=bad['observed_positions_m'].astype(np.float32)
    with pytest.raises(ValueError,match='dtype'):m.decode_public_members(bad,design=design())
    bad=copy.deepcopy(encoded);bad['young_Pa']=np.array(90000.)
    with pytest.raises(ValueError,match='allowlist'):m.decode_public_members(bad,design=design())
    with pytest.raises(FileExistsError):m.exclusive_npz(path,encoded)


def test_preflight_and_prepare_do_not_mint_ids_or_draw_noise_and_write_only_tempROOT(tmp_path,monkeypatch):
    validation,controls=setup(tmp_path,monkeypatch)
    monkeypatch.setattr(m.obs,'mint_opaque_ids',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('IDs during prepare')))
    monkeypatch.setattr(m.obs,'measurement_error',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('Noise during prepare')))
    monkeypatch.setattr(np,'load',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('Arrays during prepare')))
    writes=[];original=Path.open
    def guard(self,mode='r',*a,**kw):
        if any(c in mode for c in 'wax+'):
            assert self.resolve().is_relative_to(tmp_path);writes.append(self)
        return original(self,mode,*a,**kw)
    monkeypatch.setattr(Path,'open',guard)
    digest=m.prepare('opdev/protocol.json','output',point_validation=validation.name,posterior_controls=controls.name,root=tmp_path)
    p=m.validate('opdev/protocol.json',digest,root=tmp_path)
    assert len(writes)==1 and not (tmp_path/'output').exists() and p['counts']['public_prefixes']==64
    with pytest.raises(ValueError,match='Fresh exclusive'):m.prepare('opdev/protocol.json','output',point_validation=validation.name,posterior_controls=controls.name,root=tmp_path)


def test_inexact_control_archive_binding_rejected_before_entropy_or_arrays(tmp_path,monkeypatch):
    validation,controls=setup(tmp_path,monkeypatch);c=json.loads(controls.read_text());c['candidate_archive_sha256']='0'*64;controls.write_text(json.dumps(c))
    monkeypatch.setattr(np,'load',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('Arrays before control checks')))
    with pytest.raises(ValueError,match='posterior controls'):m.prepare('p.json','out',point_validation=validation.name,posterior_controls=controls.name,root=tmp_path)
    assert not (tmp_path/'p.json').exists() and not (tmp_path/'out').exists()


def fake_runtime(tmp,monkeypatch,*,fail_after=None):
    validation,controls=setup(tmp,monkeypatch)
    digest=m.prepare('opdev/protocol.json','output',point_validation=validation.name,posterior_controls=controls.name,root=tmp)
    monkeypatch.setattr(m,'guard',lambda root:None)
    monkeypatch.setattr(m.obs,'load_qualified_private_responses',lambda *a,private_configuration_index,**kw:fake_bound(private_configuration_index))
    ids=[format(i+1,'032x') for i in range(64)];minted=[]
    def mint(n):minted.append(n);return ids.copy()
    monkeypatch.setattr(m.obs,'mint_opaque_ids',mint)
    original=m.obs.private_repeat;calls=[]
    def repeat(c,r,token,bound,*,design):
        mapping=tmp/'output/PRIVATE/identity-map.json';assert mapping.exists()
        saved=json.loads(mapping.read_text());assert len(saved['records'])==64
        calls.append((c,r,token))
        if fail_after is not None and len(calls)>fail_after:raise RuntimeError('manufactured failure after saved IDs')
        return original(c,r,token,bound,design=design,rng_factory=FakeFactory())
    monkeypatch.setattr(m.obs,'private_repeat',repeat)
    return digest,minted,calls


def test_complete64_fake_run_manifests_keep_four_private_groups_and_public_only_prefixes(tmp_path,monkeypatch):
    digest,minted,calls=fake_runtime(tmp_path,monkeypatch);s=m.run('opdev/protocol.json',digest,root=tmp_path)
    assert s['completed'] and len(calls)==64 and minted==[64] and s['private_configuration_completed_counts']=={str(i):16 for i in range(4)}
    public_dir=tmp_path/'output/PUBLIC';manifest=json.loads((public_dir/'manifest.json').read_text());assert manifest['completed'] and len(manifest['records'])==64
    assert manifest['physical_configurations']==4 and not manifest['independent_population_objects']
    assert not {'private_records','identity_mapping','source_records','posterior_controls','young_Pa','support_category'}&set(manifest)
    for r in manifest['records']:
        assert set(r)=={'object_id','prefix'}
        p=m.load_public_prefix(public_dir,r['prefix'],design=design());assert set(p)==m.obs.PUBLIC_KEYS
    private=next((tmp_path/'output/PRIVATE').glob('*.npz'))
    with pytest.raises(ValueError,match='allowlist'):m.load_public_prefix(tmp_path,record(tmp_path,private),design=design())


def test_failed_partial_preserves_mapping_and_no_retry_or_identity_regeneration(tmp_path,monkeypatch):
    digest,minted,calls=fake_runtime(tmp_path,monkeypatch,fail_after=3)
    with pytest.raises(RuntimeError,match='manufactured failure'):m.run('opdev/protocol.json',digest,root=tmp_path)
    identity=tmp_path/'output/PRIVATE/identity-map.json';before=identity.read_bytes();state=json.loads((tmp_path/'output/PRIVATE/execution.json').read_text())
    assert not state['completed'] and state['identity_mapping_persisted_before_noise'] and len(state['records'])==3 and minted==[64]
    assert not (tmp_path/'output/PUBLIC/manifest.json').exists()
    with pytest.raises(FileExistsError,match='regeneration'):m.run('opdev/protocol.json',digest,root=tmp_path)
    assert identity.read_bytes()==before and minted==[64]
