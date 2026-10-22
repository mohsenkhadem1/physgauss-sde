"""One precise historical-negative classifier; never suppress other failures.

The original source remains byte-exact for frozen/unexecuted provenance. Only
its known J-report assertion can become the dedicated expected-failure type.
"""
import hashlib
from pathlib import Path
import numpy as np
import pytest
_HISTORICAL_FILE=Path(__file__).parent/'test_campaign_p2_warp_reference_force.py'
_HISTORICAL_SHA='47e666ccd63baacab1193ab3f70e2e75311870e7734a8bf5108b45d84320a514'
_HISTORICAL_FUNCTION='test_combined_manufactured_CPU_parity_fullM_R_and_all141_guards'
_HISTORICAL_NODE='tests/test_campaign_p2_warp_reference_force.py::'+_HISTORICAL_FUNCTION
_REASON=('Preserved historical independent-J-report minimum capped at1 during expansion. '
 'Exact failure receipt: runs/operator-dev/p2-warp-reference-force-v01/SUPERSEDED_UNEXECUTED-CPU-v01.json; '
 'active corrected test: tests/test_campaign_p2_warp_reference_force_v02.py. '
 'Only source-hash/line/signature-matched J failure is expected; E/g or other failures stay red.')

class HistoricalJReferenceFailure(AssertionError):pass

def _known_J_values(values):
    try:
        actual=values['a'];energy=values['E'];gradient=values['g'];J=values['J']
        return bool(np.isfinite([actual['energy'],energy,actual['minimum_J'],J]).all()
                    and np.isfinite(actual['gradient']).all() and np.isfinite(gradient).all()
                    and abs(actual['energy']-energy)<1e-12
                    and np.linalg.norm(actual['gradient']-gradient)<1e-10
                    and J==1. and actual['minimum_J']>1.
                    and abs(actual['minimum_J']-J)>=2e-14)
    except (KeyError,TypeError,ValueError):return False

def pytest_collection_modifyitems(items):
    for item in items:
        if item.nodeid==_HISTORICAL_NODE:
            if hashlib.sha256(_HISTORICAL_FILE.read_bytes()).hexdigest()!=_HISTORICAL_SHA:
                raise pytest.UsageError('Historical source changed: refusing to mark expected failure.')
            item.add_marker(pytest.mark.xfail(strict=True,raises=HistoricalJReferenceFailure,reason=_REASON))

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    outcome=yield
    if item.nodeid!=_HISTORICAL_NODE or not outcome.excinfo:return
    typ,value,tb=outcome.excinfo
    if typ is not AssertionError:return
    if hashlib.sha256(_HISTORICAL_FILE.read_bytes()).hexdigest()!=_HISTORICAL_SHA:return
    current=tb
    while current:
        frame=current.tb_frame
        if Path(frame.f_code.co_filename).resolve()==_HISTORICAL_FILE.resolve() and frame.f_code.co_name==_HISTORICAL_FUNCTION and current.tb_lineno==24:
            if _known_J_values(frame.f_locals):
                outcome.force_exception(HistoricalJReferenceFailure(_REASON).with_traceback(tb))
            return
        current=current.tb_next
