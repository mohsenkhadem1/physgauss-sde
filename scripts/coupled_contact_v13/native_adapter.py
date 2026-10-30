"""PhysTwin's unchanged spring-force kernel with v13 shared contact/integration.

This is an adapted material-core baseline, NOT the authors' complete method:
we replace native training, graph construction, masses, collision and integration
with this study's shared interfaces. In particular, upstream PhysTwin DOES
support multi-object particle collision; it is not described as floor-only.
Only a scalar effective spring scale is calibrated from observed positions.
"""
import hashlib
import ast
import subprocess
import os
from pathlib import Path
import sys
import itertools
import numpy as np
from .physics import FEMSystem

ROOT=Path(__file__).resolve().parents[2]
UPSTREAM=ROOT/'runs/third_party/PhysTwin'
_KERNEL=None


def native_kernel():
    global _KERNEL
    if _KERNEL is None:
        os.environ.setdefault('PHYSTWIN_HEADLESS','1')
        sys.path.insert(0,str(UPSTREAM))
        from qqtt.model.diff_simulator.spring_mass_warp import eval_springs
        import warp as wp
        _KERNEL=(wp,eval_springs)
    return _KERNEL


def spring_graph(rest,tets,volumes):
    """Tetra-edge graph; one sixth of each incident tet volume per edge."""
    tributary={}
    for tet,volume in zip(tets,volumes):
        for a,b in itertools.combinations(tet,2):
            edge=tuple(sorted((int(a),int(b))))
            tributary[edge]=tributary.get(edge,0.)+float(volume)/6
    edges=np.array(sorted(tributary),dtype=np.int32)
    lengths=np.linalg.norm(rest[edges[:,1]]-rest[edges[:,0]],axis=1)
    volumes=np.array([tributary[tuple(edge)] for edge in edges])
    return edges,lengths,volumes


class NativeSpringSystem(FEMSystem):
    """Same SI lumped masses, gravity, contact, boundaries and integrator as FEM.

    material.young is a fitted *effective spring scale*, not inferred Young's
    modulus. Y_e = scale * tributary_volume / rest_length has force units,
    consistent with the upstream kernel's Y_e * engineering_strain law.
    Native dashpot is a scalar coefficient; we use the mean geometry-scaled
    viscosity coefficient. No per-spring trained parameters or native optimizer.
    """
    def __init__(self,*args,device='cpu',**kwargs):
        super().__init__(*args,**kwargs)
        self.edges,self.lengths,self.edge_volumes=spring_graph(self.rest,self.tets,self.volume)
        self.coefficients=self.material.young*self.edge_volumes/self.lengths
        self.dashpot=float(np.mean(self.material.viscosity*self.edge_volumes/self.lengths**2))
        self.device=device
        wp,self.kernel=native_kernel();self.wp=wp
        self.wp_edges=wp.array(self.edges,dtype=wp.vec2i,device=device)
        self.wp_lengths=wp.array(self.lengths.astype('float32'),dtype=float,device=device)
        self.wp_coeff=wp.array(np.log(self.coefficients).astype('float32'),dtype=float,device=device)
        self.wp_x=wp.zeros(len(self.rest),dtype=wp.vec3,device=device)
        self.wp_v=wp.zeros(len(self.rest),dtype=wp.vec3,device=device)
        self.wp_force=wp.zeros(len(self.rest),dtype=wp.vec3,device=device)
        self.wp_control=wp.zeros(0,dtype=wp.vec3,device=device)

    def internal_forces(self):
        J=np.linalg.det(self.deformation())
        if not np.isfinite(self.x).all() or J.min()<=.02:raise FloatingPointError('Invalid native spring mesh state')
        wp=self.wp
        self.wp_x.assign(np.asarray(self.x,dtype='float32'));self.wp_v.assign(np.asarray(self.v,dtype='float32'))
        self.wp_force.zero_()
        wp.launch(self.kernel,dim=len(self.edges),inputs=[self.wp_x,self.wp_v,self.wp_control,self.wp_control,
                  len(self.rest),self.wp_edges,self.wp_lengths,self.wp_coeff,self.dashpot,1e-12,1e12],
                  outputs=[self.wp_force],device=self.device)
        return self.wp_force.numpy().astype(float),J

    def elastic_energy(self,x=None):
        x=self.x if x is None else np.asarray(x)
        lengths=np.linalg.norm(x[self.edges[:,1]]-x[self.edges[:,0]],axis=1)
        return float(np.sum(.5*self.coefficients/self.lengths*(lengths-self.lengths)**2))

    def metadata(self):
        source=UPSTREAM/'qqtt/model/diff_simulator/spring_mass_warp.py'
        upstream_text=source.read_text()
        def extract(text):
            tree=ast.parse(text);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='eval_springs')
            return ast.dump(node,include_attributes=False)
        checkout=subprocess.check_output(['git','-C',str(UPSTREAM),'rev-parse','HEAD'],text=True).strip()
        committed=subprocess.check_output(['git','-C',str(UPSTREAM),'show','HEAD:qqtt/model/diff_simulator/spring_mass_warp.py'],text=True)
        identical=extract(committed)==extract(upstream_text)
        if not identical:raise RuntimeError('Native spring kernel differs from author checkout')
        return dict(label='PhysTwin native spring-force core + shared v13 contact/integrator',
                    author_repository='https://github.com/Jianghanxiao/PhysTwin',author_checkout=checkout,
                    kernel_ast_sha256=hashlib.sha256(extract(upstream_text).encode()).hexdigest(),
                    kernel_matches_author_checkout=identical,
                    local_source_amendment='Existing simulator constructor gravity_scale option; eval_springs unchanged; v13 bypasses native constructor',
                    full_author_method=False,unchanged_kernel='eval_springs',
                    upstream_source=str(source.relative_to(ROOT)),upstream_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    upstream_license='MIT, Copyright (c) 2025 Hanxiao Jiang',device=self.device,
                    graph='tetrahedral edges; no control attachment springs',
                    mass='identical SI lumped masses as v13 FEM, not upstream unit particle masses',
                    fitted_parameter='effective spring scale; not identified Young modulus',
                    spring_coefficient='effective_scale * tributary_tet_volume / rest_length (N)',
                    dashpot_coefficient=self.dashpot,
                    shared='initial geometry, node identities, gravity, boundary conditions, contact forces, friction, rigid-body response, integration, positive-J failure checks',
                    replaced='native graph, optimizer, per-spring inference, mass convention, collision and integration',
                    upstream_capability='native core separately supports multi-object particle impulses; not used by this controlled material-core comparison')
