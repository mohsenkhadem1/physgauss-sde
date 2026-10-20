"""Linear beam mechanics with material and boundary Gaussian kernels.

Reference-coordinate Gaussians parameterize fields; they are NOT probabilities
over position. Boundary kernels are integrated on the reduced beam boundary.
The exact posterior below enumerates boundary hypotheses and integrates loads.
"""
from dataclasses import dataclass, asdict
from itertools import product
import math
import numpy as np


@dataclass(frozen=True)
class Boundary:
    support_center: float
    support_width: float
    support_stiffness: float
    load_center: float
    load_width: float = 0.035


def candidates():
    return [Boundary(*v) for v in product(
        [0.075, 0.150, 0.225], [0.035, 0.060], [2000., 10000.], [0.85, 1.0])]


def shape(xi, h):
    """Cubic Hermite interpolation of displacement and rotation DOFs."""
    return np.stack([1-3*xi**2+2*xi**3, h*(xi-2*xi**2+xi**3),
                     3*xi**2-2*xi**3, h*(-xi**2+xi**3)], axis=-1)


def interpolate_matrix(x, elements):
    x = np.asarray(x)
    if np.any((x < 0) | (x > 1)):
        raise ValueError('Beam coordinates must be in [0,1] metres')
    e = np.minimum((x*elements).astype(int), elements-1)
    xi = x*elements-e
    out = np.zeros((len(x), 2*(elements+1)))
    for row, element in enumerate(e):
        out[row, 2*element:2*element+4] = shape(xi[row], 1/elements)
    return out


def gaussian_density(x, center, width):
    """Gaussian restricted and normalized on the physical interval [0,1]."""
    z = .5*(math.erf((1-center)/(width*math.sqrt(2)))
            - math.erf(-center/(width*math.sqrt(2))))
    return np.exp(-.5*((x-center)/width)**2)/(width*math.sqrt(2*math.pi)*z)


def material_rigidity(x, values=(.8, .8, .8)):
    """Partition-of-unity material Gaussians; EI is in N m^2."""
    weights = np.exp(-.5*((np.asarray(x)[..., None]-np.array([0., .5, 1.]))/.3)**2)
    weights /= weights.sum(axis=-1, keepdims=True)
    if min(values) <= 0:
        raise ValueError('Positive material bending rigidity required')
    return weights @ np.asarray(values)


def assemble(boundary, elements=40, rigidity=(.8, .8, .8)):
    """Galerkin beam: integral EI u''v'' + kappa w_s u v = integral F w_l v.

    Ends are naturally free (zero moment/shear). The Gaussian foundation is a
    compliant support with zero anchor displacement, not a hard clamp.
    """
    h = 1/elements
    z, weights = np.polynomial.legendre.leggauss(8)
    xi = (z+1)/2
    weights = weights*h/2
    n = shape(xi, h)
    curvature = np.stack([(-6+12*xi)/h**2, (-4+6*xi)/h,
                          (6-12*xi)/h**2, (-2+6*xi)/h], axis=-1)
    size = 2*(elements+1)
    elastic, support = np.zeros((size,size)), np.zeros((size,size))
    force = np.zeros(size)
    for e in range(elements):
        x = (e+xi)*h
        dof = slice(2*e, 2*e+4)
        elastic[dof,dof] += curvature.T @ ((weights*material_rigidity(x,rigidity))[:,None]*curvature)
        density = gaussian_density(x,boundary.support_center,boundary.support_width)
        support[dof,dof] += n.T @ ((weights*boundary.support_stiffness*density)[:,None]*n)
        force[dof] += n.T @ (weights*gaussian_density(x,boundary.load_center,boundary.load_width))
    return elastic, support, force


def response(boundary, x, elements=40, rigidity=(.8,.8,.8)):
    elastic, support, force = assemble(boundary,elements,rigidity)
    dofs = np.linalg.solve(elastic+support,force)
    return interpolate_matrix(x,elements) @ dofs


def camera(time):
    """Smooth moving orthographic camera; known pose and metric pixel scale."""
    yaw = -.45+.9*time/8
    pitch = .25+.2*np.sin(time/8*np.pi)
    normal = np.array([np.sin(yaw)*np.cos(pitch), np.sin(pitch), np.cos(yaw)*np.cos(pitch)])
    right = np.cross([0.,1.,0.],normal); right /= np.linalg.norm(right)
    up = np.cross(normal,right)
    return np.stack([right,-up])*400, normal


def force_covariance(times, std=.06, length=2.):
    """Known smooth load prior, conditioned on zero load in the static frame."""
    times = np.asarray(times)
    k = np.exp(-.5*((times[:,None]-times[None,:])/length)**2)
    k -= np.outer(np.exp(-.5*(times/length)**2),np.exp(-.5*(times/length)**2))
    return std**2*k


def observation_design(kind):
    if kind == 'sparse':
        frames, positions = [0,3,7], [.40,.55,.70]
    elif kind == 'richer':
        frames, positions = [0,1,3,5,7], np.linspace(.35,.75,9)
    else:
        raise ValueError(kind)
    return [(t,float(x),axis) for t in frames for x in positions for axis in [0,1]]


def observation_matrix(boundary, design, elements=40):
    spatial = response(boundary,np.array([v[1] for v in design]),elements)
    h = np.zeros((len(design),8))
    for row, (t, _, axis) in enumerate(design):
        h[row,t] = camera(t+1)[0][axis,1]*spatial[row]
    return h


class ExactBoundaryPosterior:
    """Finite Gaussian mixture. No sampling/convergence approximation."""
    def __init__(self, boundaries, design, sigma=.25, elements=40):
        self.boundaries, self.design, self.sigma = boundaries, design, sigma
        self.prior = force_covariance(np.arange(1,9))
        l = np.linalg.cholesky(self.prior)
        self.h, self.maps, self.cov, self.innovation_maps, self.logdet = [], [], [], [], []
        for boundary in boundaries:
            h = observation_matrix(boundary,design,elements)
            w = h @ l/sigma
            a = np.eye(8)+w.T@w
            chol = np.linalg.cholesky(a)
            aw = np.linalg.solve(a,w.T)
            self.h.append(h)
            self.maps.append(l @ aw/sigma)
            self.cov.append(l @ np.linalg.solve(a,l.T))
            self.innovation_maps.append(aw)
            self.logdet.append(2*np.log(np.diag(chol)).sum())
        self.h, self.maps, self.cov = map(np.asarray,(self.h,self.maps,self.cov))
        self.innovation_maps, self.logdet = map(np.asarray,(self.innovation_maps,self.logdet))
        self.whitened_h = self.h @ l /sigma

    def fit(self, y):
        y = np.atleast_2d(y)
        b = np.einsum('cmt,nm->nct',self.whitened_h,y/self.sigma)
        m = np.einsum('ctm,nm->nct',self.innovation_maps,y/self.sigma)
        logp = -.5*(np.sum((y/self.sigma)**2,axis=1)[:,None]
                       -np.sum(b*m,axis=2)+self.logdet[None,:]
                       +len(self.design)*np.log(2*np.pi*self.sigma**2))
        logp -= logp.max(axis=1,keepdims=True)
        weights = np.exp(logp); weights /= weights.sum(axis=1,keepdims=True)
        mean = np.einsum('ctm,nm->nct',self.maps,y)
        return weights, mean


_erf = np.frompyfunc(math.erf,1,1)


def normal_cdf(x):
    return .5*(1+_erf(np.asarray(x)/np.sqrt(2)).astype(float))


def mixture_cdf(value, weights, means, variances):
    return np.sum(weights*normal_cdf((np.asarray(value)[...,None]-means)
                                    /np.sqrt(np.maximum(variances,1e-30))),axis=-1)


def mixture_quantile(probability, weights, means, variances):
    sd = np.sqrt(np.maximum(variances,1e-30))
    low, high = (means-12*sd).min(axis=-1), (means+12*sd).max(axis=-1)
    for _ in range(52):
        mid = (low+high)/2
        below = mixture_cdf(mid,weights,means,variances) < probability
        low, high = np.where(below,mid,low), np.where(below,high,mid)
    return (low+high)/2


def wilson(successes, count):
    z = 1.959963984540054
    p = successes/count
    den = 1+z*z/count
    center = (p+z*z/(2*count))/den
    radius = z*np.sqrt(p*(1-p)/count+z*z/(4*count*count))/den
    return [float(center-radius),float(center+radius)]


def uncertainty_metrics(truth, weights, means, variances):
    pit = np.sort(mixture_cdf(truth,weights,means,variances))
    n = len(truth)
    ks = max(np.max(np.arange(1,n+1)/n-pit),np.max(pit-np.arange(n)/n))
    result = {'pit_ks':float(ks),'pit_ks_99pct_reference':float(1.63/np.sqrt(n))}
    for level in [.5,.8,.9,.95]:
        lo = mixture_quantile((1-level)/2,weights,means,variances)
        hi = mixture_quantile((1+level)/2,weights,means,variances)
        covered = (truth>=lo)&(truth<=hi)
        result[str(level)] = {'coverage':float(covered.mean()),
            'wilson_95pct':wilson(int(covered.sum()),n),
            'mean_width':float(np.mean(hi-lo))}
    return result


def simulation_audit(boundaries):
    checks=[]
    for b in [boundaries[0],boundaries[len(boundaries)//2],boundaries[-1]]:
        x = np.linspace(0,1,81)
        u40,u80,u160 = [response(b,x,n) for n in [40,80,160]]
        ke,ks,f = assemble(b,80)
        u = np.linalg.solve(ke+ks,f)
        translation = np.tile([1.,0.],81)
        rotation = np.empty(162); rotation[0::2]=np.linspace(0,1,81); rotation[1::2]=1
        reaction = -ks@u
        checks.append({'boundary':asdict(b),
            'relative_refinement_40_80':float(np.max(abs(u40-u80))/np.max(abs(u80))),
            'relative_refinement_80_160':float(np.max(abs(u80-u160))/np.max(abs(u160))),
            'unit_load_integral_error':float(abs(translation@f-1)),
            'force_balance_error':float(abs(translation@(f+reaction))),
            'moment_balance_error':float(abs(rotation@(f+reaction))),
            'relative_solve_residual':float(np.linalg.norm((ke+ks)@u-f)/np.linalg.norm(f)),
            'energy_work_error':float(abs(u@(ke+ks)@u-u@f))})
    return {'checks':checks,'pass':all(
        c['relative_refinement_40_80']<2e-3 and c['relative_refinement_80_160']<2e-4
        and max(c[k] for k in ['unit_load_integral_error','force_balance_error',
                               'moment_balance_error','relative_solve_residual','energy_work_error'])<1e-6
        for c in checks)}
