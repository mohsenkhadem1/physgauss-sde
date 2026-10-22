import numpy as np
from conftest import _known_J_values

def test_only_known_J_signature_matches_not_energy_force_other_regressions():
 a=dict(energy=1.,gradient=np.zeros((2,3)),minimum_J=1.0185126);values=dict(a=a,E=1.,g=np.zeros((2,3)),J=1.)
 assert _known_J_values(values)
 assert not _known_J_values({**values,'E':1.1})
 assert not _known_J_values({**values,'g':np.ones((2,3))})
 assert not _known_J_values({**values,'J':.9})
 assert not _known_J_values({**values,'a':{**a,'minimum_J':.9}})
 assert not _known_J_values({**values,'a':{**a,'gradient':np.full((2,3),np.nan)}})
