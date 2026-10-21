import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments'))
from isaac_rich_control_worker import rich_menu


def test_rich_curriculum_covers_angles_and_amplitudes_without_evaluation_leak():
    cfg = json.loads((Path(__file__).resolve().parents[1]/'configs/gaussian_rich_controls_v01.json').read_text())
    menu = rich_menu(cfg, [[0., 0, .25], [.1, 0, .25]], np.eye(3), np.eye(4))
    assert len(menu['candidates']) == 13
    additions = menu['candidates'][4:]
    assert [a['action_id'] for a in additions] == cfg['new_fitting_ids']
    for index, action in enumerate(additions[:8]):
        assert action['force_type_index'] == 0
        np.testing.assert_allclose(np.linalg.norm(action['peak_force_N']), cfg['resultant_force_norms_N'][index % 2])
        assert np.count_nonzero(action['peak_force_N']) == 2
        assert not np.allclose(action['peak_force_N'], menu['evaluation_query']['peak_force_N'])
    assert 'evaluation' not in menu['fitting_ids']
    assert 'reset_replay' not in menu['fitting_ids']
    np.testing.assert_array_equal(additions[-1]['peak_force_N'], [0, 0, 0])
