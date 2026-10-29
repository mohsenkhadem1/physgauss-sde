import numpy as np
from scripts.position_ood.energy_recovery import EnergyRecoveryFixture
from scripts.phystwin.consistent_fixture import ConsistentFixture
from scripts.position_ood.fixture import controls

def test_energy_warm_start_requires_original_newton_certificate(monkeypatch):
    model=EnergyRecoveryFixture(2,20000.,5000.,8000.,'crossed',device='cpu',field_amplitude_J=.08)
    model.set_controls(controls('stretch'))
    original=ConsistentFixture.solve
    reference=original(model,force_tolerance_N=1e-9)['displacement_local_m']
    calls=[]
    def fail_first(self,*args,**kwargs):
        calls.append(1)
        if len(calls)==1:raise RuntimeError('forced Newton recovery diagnostic')
        return original(self,*args,**kwargs)
    monkeypatch.setattr(ConsistentFixture,'solve',fail_first)
    state=model.solve(force_tolerance_N=1e-9)
    assert len(calls)==2 and state['energy_warm_evaluations']>0
    assert np.linalg.norm(state['residual'])<=1e-9
    np.testing.assert_allclose(state['displacement_local_m'],reference,atol=1e-8,rtol=1e-6)
