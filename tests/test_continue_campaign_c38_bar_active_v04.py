import hashlib
import json
import pytest
from experiments import continue_campaign_c38_bar_active_v04 as run


def test_reuse_hash_closure_rejects_changed_choice_or_outcome(tmp_path,monkeypatch):
    monkeypatch.setattr(run,'ROOT',tmp_path)
    paths=['choice.json','outcomes.npz','old-summary.json']
    for p in paths:(tmp_path/p).write_bytes(b'unit-control-bytes')
    records=[dict(path=p,sha256=hashlib.sha256((tmp_path/p).read_bytes()).hexdigest()) for p in paths]
    run.check_reused_records(dict(reused_records=records))
    (tmp_path/'choice.json').write_text('{"reranked":true}')
    with pytest.raises(ValueError,match='Changed'):run.check_reused_records(dict(reused_records=records))
    with pytest.raises(ValueError,match='Duplicate'):run.check_reused_records(dict(reused_records=records+records))


def test_reuse_cannot_escape_root_even_with_valid_content_digest(tmp_path,monkeypatch):
    root=tmp_path/'root';root.mkdir();outside=tmp_path/'outside.json';outside.write_text('{"unit":"outside"}')
    monkeypatch.setattr(run,'ROOT',root)
    digest=hashlib.sha256(outside.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='Changed'):
        run.check_reused_records(dict(reused_records=[dict(path='../outside.json',sha256=digest)]))
