import numpy as np
from scripts.complex_scenes_v18.scenarios import smooth


def test_release_endpoints_roundoff_and_derivative():
    t=0.
    for _ in range(6401):
        f,df=smooth(t,.09,.03)
        assert 0<=f<=1 and df>=0
        t+=25e-6
    for t in [.095,.10,.11,.115]:
        f,df=smooth(t,.09,.03);h=1e-7
        assert np.isclose(df,(smooth(t+h,.09,.03)[0]-smooth(t-h,.09,.03)[0])/(2*h),rtol=1e-7)
