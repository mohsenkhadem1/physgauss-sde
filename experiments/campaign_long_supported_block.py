"""SOURCE-only .48m supported block and matching-domain numerical preconditioner.

New domain, not an admitted continuation of .24m C38 fields. Fine constitutive
law/full mass are unchanged; supports and material patches use physical geometry.
"""
import numpy as np
import torch
from scipy import sparse
from experiments.boundary_block import block_mesh
from experiments.campaign_static_p2_block import enrich,shapes,EDGES,QuadraticBlock
from experiments.campaign_c38_supported_grading import symmetric_grade
from experiments.campaign_c38_static_family import boundary_faces6,array_record
from experiments.campaign_p2_exact_quadrature import exact_degree6_rule
from experiments.campaign_p2_keast_quadrature import positive_keast24
from experiments.campaign_newton_stable_material import StableP2Block
from experiments.campaign_p2_consistent_transient import scalar_mass
from experiments.campaign_p2_block_preconditioner import BlockJacobiP2
from experiments.campaign_p2_coarse_preconditioner import TwoLevelP2,compatible_prolongation
from experiments.campaign_p2_streaming import StreamingTwoLevelP2
from experiments.campaign_p2_torch import LinearFailure
SIZE=np.array([.48,.08,.08]);SUPPORTS=('both_ends','separated_bottom_patches')


def graded_axes(cells):
 if len(cells)!=3 or any(type(n) is not int or n<2 or n%2 for n in cells) or cells[0]%4:raise ValueError('Even cells, nx multiple4 required')
 anchors=(0.,.12,.36,.48);counts=(cells[0]//4,cells[0]//2,cells[0]//4);parts=[]
 for lo,hi,n in zip(anchors[:-1],anchors[1:],counts):
  a=lo+(hi-lo)*symmetric_grade(np.arange(n+1)/n);a[0],a[-1]=lo,hi;parts.append(a[:-1])
 return (np.r_[*parts,.48],)+tuple(-.04+.08*symmetric_grade(np.arange(n+1)/n) for n in cells[1:])


def compile_forces(model,world_surface_force_N):
 vector=np.asarray(world_surface_force_N,dtype=float)
 if vector.shape!=(3,) or not np.isfinite(vector).all():raise ValueError('Explicit finite public world force vector')
 T=model.elements;local=model.volumes[:,None]*np.array([-.05]*4+[.2]*6);integral=np.zeros(len(model.rest));np.add.at(integral,T,local);body=np.zeros_like(model.rest);body[:,2]=-9.81*625.*integral
 six=boundary_faces6(model.elements,model.faces);corners=model.rest[six[:,:3]];top=np.all(abs(corners[:,:,2]-.04)<1e-13,1);lo=corners[:,:,0].min(1);hi=corners[:,:,0].max(1)
 if np.any(top&(((lo<.12-1e-13)&(hi>.12+1e-13))|((lo<.36-1e-13)&(hi>.36+1e-13)))):raise ValueError('Material patch edges must conform')
 selected=top&(lo>=.12-1e-13)&(hi<=.36+1e-13);tri=corners[selected];areas=.5*np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1);area=float(areas.sum())
 if abs(area-.24*.08)>1e-13:raise ValueError('Physical full-width middle patch area')
 surface=np.zeros_like(body)
 for ids,A in zip(six[selected],areas):np.add.at(surface,ids[3:],A/(3*area)*vector)
 center=np.array([.24,0.,0.]);F=surface.sum(0);moment=np.cross(model.rest-center,surface).sum(0)
 if np.linalg.norm(F-vector)>1e-12 or np.linalg.norm(moment-np.cross([0,0,.04],vector))>1e-12 or abs(integral.sum()-np.prod(SIZE))>1e-16:raise ValueError('Original force/volume/wrench identities')
 return dict(total_N=body+surface,body_N=body,surface_N=surface,body_integral_weights_m3=integral,metadata=dict(force_vector_N=vector.tolist(),rest_surface_resultant_N=F.tolist(),rest_surface_moment_Nm=moment.tolist(),origin_m=center.tolist(),patch='z=.04,x in[.12,.36],full y width; world dead force/reference T6 weights',physical_mass_kg=float(625.*integral.sum()),signed_P2_shape_loads_not_particle_masses=True,no_E_or_support_input=True))


class LongSupportedGeometry:
 query=QuadraticBlock.query
 diagnostics=QuadraticBlock.diagnostics
 def __init__(self,cells=(4,2,2),*,support,young=300000.,rule='keast24',world_surface_force_N=(0.,0.,0.)):
  if support not in SUPPORTS or young!=300000. or rule not in ('keast24','exact80'):raise ValueError('New fixed E300k/two-support/domain-only scope')
  axes=graded_axes(cells);_,self.tets,self.faces=block_mesh(SIZE,tuple(cells),(SIZE[0]/2,0,0));X=np.stack(np.meshgrid(*axes,indexing='ij'),-1).reshape(-1,3);self.rest,self.elements=enrich(X,self.tets);self.reference_local=self.rest;corners=X[self.tets];Dm=np.swapaxes(corners[:,1:]-corners[:,:1],-1,-2);det=np.linalg.det(Dm)
  if (det<=0).any():raise ValueError('Positive affine rest elements')
  inv=np.linalg.inv(Dm);self.linear_gradients=np.concatenate([-inv.sum(1)[:,None],inv],1);self.volumes=det/6;self.young=young;self.poisson=.35;self.density=625.;self.minimum_J=.1;self.rotation=np.eye(3);self.translation=np.zeros(3);self.size_m=SIZE.copy();self.support_kind=support
  self.maximum_reference_tet_edge_m=float(max(np.linalg.norm(corners[:,i]-corners[:,j],axis=1).max() for i,j in EDGES));old,_=exact_degree6_rule();self.check_bary=np.concatenate([np.eye(4),[(np.eye(4)[i]+np.eye(4)[j])/2 for i,j in EDGES],old]);self.bary,self.weights=positive_keast24() if rule=='keast24' else exact_degree6_rule();self.N=shapes(self.bary);self.order=rule;self.estimated_peak_memory_bytes=None
  x,y,z=self.rest.T
  groups=np.stack([abs(x)<1e-13,abs(x-.48)<1e-13]) if support=='both_ends' else np.stack([(abs(z+.04)<1e-13)&(x<=.12+1e-13),(abs(z+.04)<1e-13)&(x>=.36-1e-13)])
  self.mask=np.repeat(groups.any(0)[:,None],3,1);self.free=np.flatnonzero(~self.mask.ravel());self.public_force_components=compile_forces(self,world_surface_force_N);self.load=self.public_force_components['total_N'];self.body_integral_weights=self.public_force_components['body_integral_weights_m3'];self.force_metadata=self.public_force_components['metadata'];self.mapping='NEWlongblock_piecewise_boundary_grade';self.geometry_metadata=dict(size_m=SIZE.tolist(),cells=list(cells),support=support,reference_geometry=array_record(self.rest),new_domain_requires_own_static_and_dynamic_qualification=True)
  six=boundary_faces6(self.elements,self.faces);tri=self.rest[six[:,:3]]
  if support=='both_ends':selected=np.all(abs(tri[:,:,0])<1e-13,1)|np.all(abs(tri[:,:,0]-.48)<1e-13,1);expected=2*.08*.08
  else:
   selected=np.all(abs(tri[:,:,2]+.04)<1e-13,1)&((tri[:,:,0].max(1)<=.12+1e-13)|(tri[:,:,0].min(1)>=.36-1e-13));expected=.24*.08
  area=.5*np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
  if not self.mask[six[selected]].all() or abs(area[selected].sum()-expected)>1e-13:raise ValueError('Exact full polynomial support trace')
  self.mask_trace=dict(passed=True,physical_area_m2=float(area[selected].sum()),all6_P2_shape_coefficients_fixed=True)
  self.grading_metadata=self.geometry_metadata


class LongDomainTwoLevelP2(TwoLevelP2):
 def __init__(self,model,device='cpu',maximum_cg_iterations=4000):
  if not np.array_equal(model.size_m,SIZE) or model.support_kind not in SUPPORTS:raise ValueError('Explicit NEWlong-domain geometry/support required')
  x,z=model.rest[:,0],model.rest[:,2]
  fixed=(abs(x)<1e-13)|(abs(x-.48)<1e-13) if model.support_kind=='both_ends' else (abs(z+.04)<1e-13)&((x<=.12+1e-13)|(x>=.36-1e-13))
  if not np.array_equal(model.mask,np.repeat(fixed[:,None],3,1)):raise ValueError('Exact declared long-domain support mask required')
  BlockJacobiP2.__init__(self,model,device,maximum_cg_iterations);coarse=LongSupportedGeometry((4,2,2),support=model.support_kind);d=(coarse.elements[:,:,None]*3+np.arange(3)).reshape(-1,30);coarse.rows=np.repeat(d,30,1).ravel().astype(np.int32);coarse.cols=np.tile(d,(1,30)).ravel().astype(np.int32)
  P=compatible_prolongation(coarse,model.rest);self.prolongation_csr=P;coo=P.tocoo();index=torch.tensor(np.vstack([coo.row,coo.col]),dtype=torch.int64,device=self.device);self.P=torch.sparse_coo_tensor(index,torch.tensor(coo.data,dtype=self.dtype,device=self.device),P.shape,check_invariants=True).coalesce();self.PT=self.P.transpose(0,1).coalesce();self.coarse_rest=torch.tensor(coarse.rest,dtype=self.dtype,device=self.device);self.coarse_free=torch.tensor(coarse.free,dtype=torch.int64,device=self.device);self.coarse_mask=torch.tensor(coarse.mask,dtype=torch.bool,device=self.device)
  K=StableP2Block.elastic(coarse,coarse.rest,True)[2][coarse.free][:,coarse.free].toarray();M=sparse.kron(scalar_mass(coarse),sparse.eye(3),format='csr')[coarse.free][:,coarse.free].toarray();defect=float(np.max(abs(K-K.T)))
  if defect>256*np.finfo(float).eps*max(1.,float(abs(K).max())):raise LinearFailure('Unmodified rest tangent symmetry guard')
  self.Kc=torch.tensor(.5*(K+K.T),dtype=self.dtype,device=self.device);self.Mc=torch.tensor(.5*(M+M.T),dtype=self.dtype,device=self.device);self.coarse_factor_cache={};self.two_level_records=[];self.coarse_symmetry_defect_N_m=defect


class StreamingLongSupportedP2(StreamingTwoLevelP2,LongDomainTwoLevelP2):
 """Exact fine operator; only the declared coarse PRECONDITIONER domain differs."""
 def __init__(self,model,device='cpu',maximum_cg_iterations=4000,quadrature_chunk_size=8,*,support_kind):
  if support_kind!=model.support_kind:raise ValueError('Explicit matching support declaration')
  super().__init__(model,device,maximum_cg_iterations,quadrature_chunk_size)


class SparseLongStatic(LongSupportedGeometry):
 """Independent tiny CPU control only; authored CSR, unchanged full stable law."""
 elastic=StableP2Block.elastic
 evaluate=QuadraticBlock.evaluate
 solve=QuadraticBlock.solve
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs);d=(self.elements[:,:,None]*3+np.arange(3)).reshape(-1,30);self.rows=np.repeat(d,30,1).ravel().astype(np.int32);self.cols=np.tile(d,(1,30)).ravel().astype(np.int32)
