import json
import pytest
from dataclasses import asdict
from scripts.complex_scenes_v18.world_queries import Query
from scripts.complex_scenes_v18.held_manifest import sha
from scripts.complex_scenes_v18.held_prediction import context


def test_prediction_reads_public_inputs_only_and_rejects_wrong_world(tmp_path):
    campaign=tmp_path/'campaign';campaign.mkdir();prefix=tmp_path/'prefix';prefix.mkdir();fitdir=tmp_path/'fit';fitdir.mkdir()
    calibration=asdict(Query('clamp_release',0));query=asdict(Query('clamp_release',1,fixture_shift=.1))
    public=dict(split='held',freeze_sha256='frozen',worlds=[dict(world_id='w1',world_material_commitment='hidden-commitment',calibration=calibration,queries=[query,calibration])])
    (campaign/'public-manifest.json').write_text(json.dumps(public));(campaign/'manifest-seal.json').write_text(json.dumps(dict(public_manifest_sha256=sha(campaign/'public-manifest.json'),freeze_sha256='frozen')))
    metadata=dict(world_id='w1',world_material_commitment='hidden-commitment',public_query=calibration,split='held')
    (prefix/'metadata.json').write_text(json.dumps(metadata));(prefix/'observations.npz').write_bytes(b'toy observations')
    fit=dict(calibration_prefix=str(prefix),completed=True,forecast_targets_read=False,evaluation_split='held',prefix_metadata_sha256=sha(prefix/'metadata.json'),observations_sha256=sha(prefix/'observations.npz'),public_query=calibration)
    (fitdir/'fit-seal.json').write_text(json.dumps(fit))
    _,q=context(campaign,fitdir,'w1',0);assert q.fixture_shift==.1
    assert not (campaign/'private-worlds.json').exists()
    metadata['world_id']='wrong';(prefix/'metadata.json').write_text(json.dumps(metadata));fit['prefix_metadata_sha256']=sha(prefix/'metadata.json');(fitdir/'fit-seal.json').write_text(json.dumps(fit))
    with pytest.raises(ValueError,match='different world'):context(campaign,fitdir,'w1',0)
