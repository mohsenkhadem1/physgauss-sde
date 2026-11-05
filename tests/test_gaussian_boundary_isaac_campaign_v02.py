import json
import numpy as np
import pytest
import torch
from scripts.gaussian_boundary_flow_v02 import isaac_bridge_campaign as campaign


def test_standardization_is_reproducible_finite_and_noncollapsed():
    transform=campaign.PhysicalParameterTransform()
    p=transform.unconstrained_to_physical(torch.zeros(4,25))
    a=campaign.fit_logit_standardization(p);b=campaign.fit_logit_standardization(p)
    assert all(torch.equal(x,y) for x,y in zip(a,b))
    assert torch.isfinite(a[0]).all() and (a[1]>=.1).all()
    z=torch.randn(2,25)
    assert torch.allclose(((z*a[1]+a[0])-a[0])/a[1],z,atol=1e-6)


def test_cache_test_role_rejected_before_shard_open(tmp_path,monkeypatch):
    contract=tmp_path/'contract.json';contract.write_text(json.dumps(dict(sources={})))
    rows=[dict(role='train',cache='does-not-exist') for _ in range(195)]+[dict(role='test',cache='private')]
    (tmp_path/'manifest.json').write_text(json.dumps(dict(contract_sha256=campaign.sha(contract),cases=rows)))
    with pytest.raises(ValueError,match='train/validation roster'):campaign.load_cache(tmp_path)


def test_frozen_recipe_rejects_source_mismatch_before_artifact_access(tmp_path):
    (tmp_path/'recipe.json').write_text(json.dumps(dict(source_sha256={})))
    with pytest.raises(ValueError,match='source binding'):campaign.validate_recipe(tmp_path)


def test_outside_support_original_labels_not_silently_clipped():
    t=campaign.PhysicalParameterTransform();p=t.lower[None].clone()
    with pytest.raises(ValueError,match='outside support'):campaign.fit_logit_standardization(p)


def test_duplicate_roster_rejected_before_shards(tmp_path):
    catalog=campaign.ROOT/'runs/gaussian-final-passive-catalog.json'
    cases=json.loads(catalog.read_text())['cases']
    rows=[dict(key=r['key'],role=r['split'],group=r['group_id'],recipe_sha256=r['recipe_sha256'],dataset=r['key'].rsplit('/',1)[0]) for r in cases]
    train_ids=[i for i,r in enumerate(rows) if r['role']=='train'];rows[train_ids[1]]=dict(rows[train_ids[0]])
    contract=tmp_path/'contract.json';contract.write_text(json.dumps(dict(sources={},catalog_sha256=campaign.sha(catalog))))
    (tmp_path/'manifest.json').write_text(json.dumps(dict(contract_sha256=campaign.sha(contract),cases=rows)))
    with pytest.raises(ValueError,match='Duplicate'):campaign.load_cache(tmp_path)


def test_nonfinite_teacher_validation_fails_explicitly(monkeypatch):
    model=campaign.IsaacConditionalFlowAdapter(width=32)
    p=model.transform.unconstrained_to_physical(torch.zeros(25)).numpy()
    row=dict(features=np.zeros((5,119),np.float32),physical=p,boundary=0,force=0,rest=np.zeros((7,3),np.float32))
    monkeypatch.setattr(campaign,'predict_particles',lambda *a,**k:torch.full((2,3,7,3),float('nan')))
    with pytest.raises(ValueError,match='Nonfinite validation teacher'):
        campaign.validation_score(model,[row],None,'cpu',dict(samples=2,flow_steps=2,seed=2,times_s=[2,2.5,3]))
