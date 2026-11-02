"""Compose frozen shared bulk/bilateral laws with known normal-contact geometry."""
import numpy as np
import torch
from torch import nn
from scripts.compositional_neural_mechanics.materials import teacher_energy
from scripts.compositional_neural_mechanics.scale_separated_material import ScaleSeparatedMaterialLaw
from scripts.compositional_neural_mechanics.interfaces import AnalyticInterfaceLaw,SharedInterfaceLaw
from scripts.compositional_neural_mechanics.contact_resolution import ResolvedContactQ2
from scripts.compositional_neural_mechanics.contact_mechanics import RigidPlane,RigidSphere
from scripts.compositional_neural_mechanics.conditional_inference_pilot import BULK,INTERFACE
FAMILIES={'mr_saturated':('mooney_rivlin',-1.,1.),'yeoh_polynomial':('yeoh',1.,0.)}


class TeacherBulk(nn.Module):
    def __init__(self,family):super().__init__();self.family=family
    def energy(self,F,z):return teacher_energy(F,z[...,:2],self.family)


def make_contact_model(family,method,n,order,device='cpu'):
    if family not in FAMILIES or method not in ('analytic_teacher','shared_laws'):raise ValueError('Declared development family/method required')
    law,flag,interfaceflag=FAMILIES[family]
    if method=='shared_laws':
        bulk=ScaleSeparatedMaterialLaw().double();bulk.load_state_dict(torch.load(BULK,map_location='cpu',weights_only=False)['state_dict'])
        interface=SharedInterfaceLaw().double();interface.load_state_dict(torch.load(INTERFACE,map_location='cpu',weights_only=False)['state_dict'])
    else:bulk=TeacherBulk(law);interface=AnalyticInterfaceLaw()
    bulk=bulk.to(device);interface=interface.to(device)
    for module in (bulk,interface):
        module.eval()
        for p in module.parameters():p.requires_grad_(False)
    def material(q):return q.new_tensor([0.,0.,flag]).expand(*q.shape[:-1],3)
    def attachment(q):return q.new_tensor([np.log(6000.),.5,interfaceflag]).expand(*q.shape[:-1],3)
    model=ResolvedContactQ2(n=n,surface_order=order,device=device,gravity=(0,0,-9.81),density=625.,bulk_energy=bulk.energy,
                            material_latent_field=material,interface_energy=interface.energy,interface_latent_fields=(attachment,attachment),
                            colliders=(RigidPlane(point=(0,0,-.04001)),RigidSphere(center=(.12,0,.09),radius=.045)))
    return model


def set_indenter(model,z):
    model.colliders=(RigidPlane(point=(0,0,-.04001)),RigidSphere(center=(.12,0,float(z)),radius=.045))
