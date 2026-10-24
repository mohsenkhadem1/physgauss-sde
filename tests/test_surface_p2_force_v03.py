import hashlib
from pathlib import Path
import numpy as np
from experiments import campaign_surface_p2_force_v03 as f
from experiments import campaign_surface_p2_static_v02 as old


def test_external_force_only_version_preserves_generic_physics_bytes():
    assert f.NORMAL_FORCE_N==.08 and f.AXIAL_FORCE_N_PER_FACE==.18
    assert hashlib.sha256(Path(old.__file__).read_bytes()).hexdigest()=='5d7a118cdb52d1ae165478877cf64df45d2ce11dadbbf2bec13cdeb064c810f8'
    for n in(2,4,8):
        oldloads=old.qualification_loads(n);new=f.qualification_loads(n)
        assert new['paired_axial'].tobytes()==oldloads['paired_axial'].tobytes()
        np.testing.assert_allclose(new['normal'],.8*oldloads['normal'],rtol=2e-15,atol=2e-18)
        X=old.geometry(n)[0]
        for name,F in new.items():
            np.testing.assert_allclose(F.sum(0),[0,0,-.08]if name=='normal'else[.36,0,0],atol=2e-13,rtol=0)
            np.testing.assert_allclose(np.cross(X-[.18,.04,.04],F).sum(0),0,atol=2e-14)


def test_catalog_byte_matches_central_and_disjoint_labels():
    for n in(2,4,8):
        catalog=f.load_catalog(n);central=f.qualification_loads(n)
        assert len(catalog)==10
        for name in central:assert catalog[name+'_x180'].tobytes()==central[name].tobytes()
    assert set(f.CENTERS)=={.09,.135,.18,.225,.27}
