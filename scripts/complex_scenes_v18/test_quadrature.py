import numpy as np
from scripts.complex_scenes_v18.geometry import SceneBody
from scripts.compositional_neural_mechanics.mechanics import GeometrySpec
from scripts.compositional_dynamics_v17.dynamics import consistent_mass


def test_lobatto_keeps_affine_mass_and_uniform_strain_energy():
    models=[SceneBody(GeometrySpec(dimensions=(.12,.06,.06)),n=1,support_mode='free',bulk_quadrature=q,gravity=(0,0,0))for q in ['gauss3','lobatto4','lobatto5']]
    F=np.array([[.8,.1,.02],[0,1.1,.03],[0,0,1.15]])
    M=consistent_mass(models[0]).toarray();energies=[]
    for m in models:
        np.testing.assert_allclose(consistent_mass(m).toarray(),M,atol=1e-17,rtol=1e-11)
        energies.append(m.evaluate(m.rest@(F-np.eye(3)).T)['elastic_J'])
        np.testing.assert_allclose(m.total_mass,.27,atol=1e-14)
    np.testing.assert_allclose(energies,energies[0],atol=1e-12,rtol=1e-11)
