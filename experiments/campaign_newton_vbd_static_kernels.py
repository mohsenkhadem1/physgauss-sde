"""Literal installed stableNH elastic-force audit; no solver changes."""
import warp as wp
from newton._src.solvers.vbd.particle_vbd_kernels import evaluate_volumetric_neo_hookean_force_and_hessian

@wp.kernel
def elastic_local(positions:wp.array[wp.vec3],tets:wp.array2d[wp.int32],poses:wp.array[wp.mat33],materials:wp.array2d[wp.float32],forces:wp.array2d[wp.vec3]):
 k=wp.tid();e=k//4;a=k%4
 f,H=evaluate_volumetric_neo_hookean_force_and_hessian(e,a,positions,positions,tets,poses[e],materials[e,0],materials[e,1],0.,1.)
 forces[e,a]=f
