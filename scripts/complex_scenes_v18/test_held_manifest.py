import json
import pytest
from scripts.complex_scenes_v18.held_manifest import create,sha
from scripts.complex_scenes_v18.held_protocol import draft


def test_private_worlds_are_fixed_across_public_queries_and_cannot_be_redrawn(tmp_path):
    source=tmp_path/'source.py';source.write_text('toy test inventory')
    artifact=tmp_path/'weights';artifact.write_bytes(b'toy weights')
    freeze=dict(status='frozen',held_data_generated=False,protocol=draft(),source_sha256={'source.py':sha(source)},artifact_sha256={'weights':sha(artifact)})
    path=tmp_path/'freeze.json';path.write_text(json.dumps(freeze));out=tmp_path/'campaign'
    create(path,out,123,tmp_path)
    private=json.loads((out/'private-worlds.json').read_text());public=json.loads((out/'public-manifest.json').read_text())
    assert len(private['worlds'])==len(public['worlds'])==24
    assert sum(len(w['queries'])for w in public['worlds'])==48
    assert 'log_young'not in (out/'public-manifest.json').read_text()
    assert len({w['world_material_commitment']for w in public['worlds']})==24
    with pytest.raises(ValueError,match='Preserve'):create(path,out,124,tmp_path)
    source.write_text('changed')
    with pytest.raises(ValueError,match='changed'):create(path,tmp_path/'other',123,tmp_path)


def test_draft_cannot_generate_held_worlds(tmp_path):
    p=tmp_path/'draft.json';p.write_text(json.dumps(dict(status='draft_not_frozen')))
    with pytest.raises(ValueError,match='frozen'):create(p,tmp_path/'out',1,tmp_path)
    assert not (tmp_path/'out').exists()


def test_reference_worker_uses_committed_world_and_declared_refinement(tmp_path,monkeypatch):
    from scripts.complex_scenes_v18 import held_reference
    source=tmp_path/'source.py';source.write_text('toy reference source')
    artifact=tmp_path/'weights';artifact.write_bytes(b'toy')
    freeze=dict(status='frozen',held_data_generated=False,protocol=draft(),source_sha256={'source.py':sha(source)},artifact_sha256={'weights':sha(artifact)})
    path=tmp_path/'freeze.json';path.write_text(json.dumps(freeze))
    campaign=tmp_path/'runs/complex-scenes-v18/toy-test';create(path,campaign,123,tmp_path)
    monkeypatch.setattr(held_reference,'ROOT',tmp_path)
    captured=[]
    def fake_run(*args,**kwargs):captured.append((args,kwargs));return dict(completed=True)
    monkeypatch.setattr(held_reference,'run',fake_run)
    held_reference.execute(campaign,path,'deformable_pair-world00','1',refine=True)
    args,kwargs=captured[-1];assert args[1]==6 and kwargs['surface_samples']==0 and kwargs['split']=='held'
    assert len(kwargs['world'].materials)==2 and kwargs['query'].velocity_scale==1.2
    with pytest.raises(ValueError,match='census'):held_reference.execute(campaign,path,'deformable_pair-world01','1',refine=True)
    assert len(captured)==1
