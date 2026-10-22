"""Tiny calls into the installed pinned Newton VBD tetrahedron kernel."""
import warp as wp
from newton._src.solvers.vbd.particle_vbd_kernels import evaluate_volumetric_neo_hookean_force_and_hessian

@wp.kernel
def tet_vertex_probe(positions:wp.array[wp.vec3],tets:wp.array2d[wp.int32],poses:wp.array[wp.mat33],mu:float,lam:float,force:wp.array2d[wp.vec3],hessian:wp.array2d[wp.mat33]):
 k=wp.tid();e=k//4;a=k%4
 f,H=evaluate_volumetric_neo_hookean_force_and_hessian(e,a,positions,positions,tets,poses[e],mu,lam,0.,1./240.)
 force[e,a]=f;hessian[e,a]=H
