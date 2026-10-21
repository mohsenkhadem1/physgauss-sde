import json
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments'))
from campaign_control_data import require_role, load_control_snapshot


def config():
    return json.loads((Path(__file__).resolve().parents[1]/'configs/gaussian_campaign_v01.json').read_text())


def test_fitting_and_checkpoint_selection_have_distinct_roles():
    c = config()
    assert require_role(c, 't0016', 'fit')[0] == 'train'
    assert require_role(c, 'v0000', 'select')[0] == 'validation'
    for eid, purpose in [('v0000', 'fit'), ('t0016', 'select'), ('v0008', 'select'),
                         ('t0112', 'fit'), ('t0012', 'fit'), ('h0000', 'fit'), ('z0000', 'select')]:
        with pytest.raises(ValueError, match='authorized'):
            require_role(c, eid, purpose)


def test_role_rejection_precedes_any_private_input_access():
    c = config()
    c['dataset'] = '/a/path/that/does/not/exist'
    with pytest.raises(ValueError, match='authorized'):
        load_control_snapshot(c, ['t0016', 'v0008'], 'fit')
    with pytest.raises(ValueError, match='Distinct'):
        load_control_snapshot(c, ['t0016', 't0016'], 'fit')
