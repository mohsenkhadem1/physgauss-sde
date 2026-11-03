import json
import pytest
from scripts.complex_scenes_v18.held_manifest import sha
from scripts.complex_scenes_v18.score_held import census,score
from scripts.complex_scenes_v18.world_queries import Query
from dataclasses import asdict


def test_pending_census_prevents_any_reference_access(tmp_path):
    query=asdict(Query('clamp_release',0))
    public=dict(split='held',freeze_sha256='frozen',worlds=[dict(world_id='world',family='clamp_release',world_material_commitment='commitment',queries=[query,query])])
    p=tmp_path/'public-manifest.json';p.write_text(json.dumps(public))
    (tmp_path/'manifest-seal.json').write_text(json.dumps(dict(public_manifest_sha256=sha(p),freeze_sha256='frozen',total_queries=2)))
    rows=census(tmp_path);assert len(rows)==10
    assert sum(r['state']=='unsupported'for r in rows)==2
    assert sum(r['state']=='pending'for r in rows)==8
    # No reference directory exists: scoring must refuse before opening any.
    with pytest.raises(ValueError,match='Pending predictions'):score(tmp_path)
