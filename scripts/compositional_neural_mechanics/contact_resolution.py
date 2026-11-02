"""Resolution-only extension; original contact/core sources remain unchanged."""
import numpy as np
from scripts.compositional_neural_mechanics.contact_mechanics import ContactQ2Mechanics


class ResolvedContactQ2(ContactQ2Mechanics):
    def __init__(self,*args,surface_order=16,**kwargs):
        if surface_order not in (16,32,64):raise ValueError('Prospective16/32/64 orders only')
        super().__init__(*args,surface_order=16,**kwargs)
        if surface_order!=16:self.contact_patches=self._surface_quadrature(surface_order)
        self.surface_order=surface_order


def sampled_contact_areas(model,u):
    result=[]
    for c in model.colliders:
        area=0.;total=0.
        for p in model.contact_patches:
            x=np.einsum('qa,eai->qei',p['N'],(model.rest+u)[p['ids']]);gap=c.signed_distance(model.engine.tensor(x)).detach().cpu().numpy()
            area+=float(np.sum(p['weights']*(gap<0)));total+=float(p['weights'].sum())
        result.append(dict(name=c.name,active_reference_area_m2=area,total_reference_surface_area_m2=total))
    return result


def exact_flat_sphere_patch(radius=.031,center_distance=.03,penalty=1e7):
    if not 0<center_distance<radius:raise ValueError('Partial spherical penetration required')
    delta=radius-center_distance
    return dict(force_normal_N=-np.pi*penalty*center_distance*delta**2,area_m2=np.pi*(radius**2-center_distance**2),
                energy_J=np.pi*penalty*(radius*delta**3/3-delta**4/4),max_penetration_m=delta)
