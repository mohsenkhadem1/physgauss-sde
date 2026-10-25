"""Tensor Chebyshev interpolation of a nonlinear FE response correction.

The exact linear response is supplied separately. This interpolates a physical
response and therefore requires independent FE and likelihood validation;
polynomial reproduction or posterior convergence alone is not admission.
"""
import hashlib
import numpy as np


def unit_axis(n):
    if type(n)is not int or n not in(3,5,9,17,33):
        raise ValueError('Declared nested Chebyshev counts3/5/9/17/33')
    fraction=np.arange(n,dtype=float)/(n-1)
    axis=np.sin(np.pi*fraction/2)**2
    axis[0]=0.;axis[-1]=1.;axis[(n-1)//2]=.5
    return axis


def barycentric_basis(query,axis):
    q=np.asarray(query,float);x=np.asarray(axis,float)
    if q.ndim!=1 or x.ndim!=1 or len(x)<3 or not np.isfinite(q).all():
        raise ValueError('Finite one-dimensional queries and declared axis')
    if not np.array_equal(x,unit_axis(len(x))):raise ValueError('Exact declared Chebyshev axis required')
    # Preserve the supplied floating-point parameter. No clipping or snapping.
    halo=64*np.finfo(float).eps
    if np.any(q < -halo)or np.any(q > 1+halo):raise ValueError('No extrapolation outside prior beyond roundoff halo')
    weight=(-1.)**np.arange(len(x));weight[[0,-1]]*=.5
    distance=q[:,None]-x
    hits=distance==0;exact=hits.any(axis=1)
    basis=np.zeros_like(distance)
    if np.any(exact):basis[exact]=hits[exact].astype(float)
    if np.any(~exact):
        raw=weight/distance[~exact]
        basis[~exact]=raw/raw.sum(axis=1)[:,None]
    if not np.isfinite(basis).all():raise FloatingPointError('Unresolved interpolation basis')
    return basis


class NonlinearResidualAtlas:
    """Immutable delta[unit-logE,unit-logK,label,channel] correction table."""
    def __init__(self,delta,labels,*,provenance_fingerprint):
        values=np.asarray(delta,dtype='<f8')
        labels=tuple(labels)
        if values.ndim!=4 or values.shape[0]!=values.shape[1]or values.shape[2]!=len(labels)or not values.shape[3]:
            raise ValueError('Square tensor grid with label/channel axes')
        self.axis=unit_axis(values.shape[0])
        if not labels or len(set(labels))!=len(labels)or not all(isinstance(v,str)and v for v in labels):raise ValueError('Distinct nonempty force labels')
        if not np.isfinite(values).all()or not isinstance(provenance_fingerprint,str)or not provenance_fingerprint:
            raise ValueError('Finite corrections and bound provenance required')
        self.delta=np.array(values,copy=True,order='C');self.delta.setflags(write=False)
        self.axis.setflags(write=False);self.labels=labels
        self.provenance_fingerprint=provenance_fingerprint
        h=hashlib.sha256();h.update(repr(values.shape).encode());h.update(self.delta.tobytes());h.update(repr(labels).encode());h.update(provenance_fingerprint.encode())
        self.fingerprint=h.hexdigest()

    def predict(self,unit,labels,*,chunk=4096):
        u=np.asarray(unit,float);selected=tuple(labels)
        if u.ndim!=2 or u.shape[1]!=2 or not np.isfinite(u).all()or not selected:
            raise ValueError('Finite unit[batch,2] and nonempty labels')
        if type(chunk)is not int or chunk<1:raise ValueError('Bounded positive query chunk')
        indices=[self.labels.index(v)for v in selected]
        result=np.empty((len(u),len(selected),self.delta.shape[3]))
        table=self.delta[:,:,indices,:]
        for first in range(0,len(u),chunk):
            q=u[first:first+chunk]
            a=barycentric_basis(q[:,0],self.axis);b=barycentric_basis(q[:,1],self.axis)
            result[first:first+len(q)]=np.einsum('bi,bj,ijlc->blc',a,b,table,optimize=True)
        if not np.isfinite(result).all():raise FloatingPointError('Nonfinite interpolated physical correction')
        return result

    def coarsen(self,n):
        axis=unit_axis(n)
        if n>len(self.axis)or (len(self.axis)-1)%(n-1):raise ValueError('Nested coarser grid required')
        step=(len(self.axis)-1)//(n-1)
        if not np.array_equal(self.axis[::step],axis):raise ValueError('Coarse coordinate bytes must match')
        return NonlinearResidualAtlas(self.delta[::step,::step],self.labels,provenance_fingerprint=self.fingerprint)
