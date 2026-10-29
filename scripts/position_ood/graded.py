"""Symmetric spatial refinement near fixture faces and patch edges."""
import numpy as np
from scripts.phystwin.tensor_fixture import TensorFixture
from scripts.position_ood.material_field import MaterialPositionFixture

class GradedGeometry:
    def surface(self,left,lower):
        # TensorFixture calls surface after topology/quadrature construction and
        # before creating its mechanical engine. Change only physical axes; Q2
        # topology and the constitutive/boundary laws remain identical.
        if not getattr(self,'_symmetric_grading',False):
            self._symmetric_grading=True
            self.axes[0]=.12*(1-np.cos(np.linspace(0,np.pi,self.cells[0]+1)))
            self.axes[1]=-.04*np.cos(np.linspace(0,np.pi,self.cells[1]+1))
            fine=[]
            for axis in self.axes:
                a=np.empty(2*len(axis)-1);a[::2]=axis;a[1::2]=(axis[:-1]+axis[1:])/2;fine.append(a)
            self.rest=np.stack(np.meshgrid(*fine,indexing='ij'),-1).reshape(-1,3)
            cell=np.stack(np.meshgrid(*[np.arange(k) for k in self.cells],indexing='ij'),-1).reshape(-1,3)
            self.widths=np.stack([np.diff(a)[cell[:,i]] for i,a in enumerate(self.axes)],axis=1);self.volumes=self.widths.prod(1)
        return super().surface(left,lower)

class GradedTensorFixture(GradedGeometry,TensorFixture):pass
class GradedMaterialFixture(GradedGeometry,MaterialPositionFixture):pass
