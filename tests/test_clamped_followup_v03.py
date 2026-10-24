import numpy as np
from experiments.run_clamped_deformation_v03 import DESIGN,NEW_UNITS


def test_new_settings_prospective_stratification_and_uniform_force_reduction():
    assert DESIGN['force_scale']==.8 and DESIGN['design_tuned_on_v02']
    assert len(DESIGN['objects'])==13 and len(DESIGN['units'])==13
    assert len(set(DESIGN['objects']))==13
    for dimension in range(2):assert np.array_equal(np.sort(np.floor(NEW_UNITS[:,dimension]*8).astype(int)),np.arange(8))
    assert np.all((NEW_UNITS>0)&(NEW_UNITS<1))
    assert DESIGN['all_cases_required']and DESIGN['no_case_discard']and DESIGN['no_retry']
