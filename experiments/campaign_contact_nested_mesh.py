"""Sparse nested-P2 contact consistency controls; no dynamics or saved-file I/O."""
from types import SimpleNamespace
import numpy as np
from scipy import sparse
from experiments.boundary_block import block_mesh
from experiments.campaign_static_p2_block import enrich,shapes,EDGES
from experiments.campaign_p2_consistent_transient import scalar_mass
from experiments.campaign_p2_contact_root_fixed import integrate_face_fixed
from experiments.campaign_p2_contact_adaptive_energy_gradient import integrate_face_energy_gradient

SIZE=(.24,.08,.08)
ENERGY_GATE_J=1e-12
GRADIENT_GATE_N=1e-10


def geometry(cells):
 X,tets,faces=block_mesh(SIZE,cells,(.12,0.,0.));rest,elements=enrich(X,tets);rest=rest+np.array([0.,0.,.09]);corners=X[tets];volumes=np.linalg.det(np.swapaxes(corners[:,1:]-corners[:,:1],-1,-2))/6
 return SimpleNamespace(rest=rest,tets=tets,elements=elements,faces=faces,volumes=volumes,density=625.,cells=tuple(cells))


def lattice(m):
 """Actual connectivity on an exact integer lattice, in half-cell units."""
 count=int(m.tets.max())+1;corner=m.rest[:count];axes=[sorted(set(corner[:,j]))for j in range(3)];lookup=[{v:i for i,v in enumerate(a)}for a in axes]
 if tuple(len(a)-1 for a in axes)!=m.cells:raise ValueError('Actual corner lattice/cell mismatch')
 points=np.full((len(m.rest),3),-1,dtype=np.int64)
 for n,x in enumerate(corner):points[n]=[2*lookup[j][x[j]]for j in range(3)]
 for tet,e in zip(m.tets,m.elements):
  for k,(a,b)in enumerate(EDGES):
   p=(points[tet[a]]+points[tet[b]])//2;node=e[4+k]
   if points[node,0]>=0 and not np.array_equal(points[node],p):raise ValueError('Incompatible P2 midpoint connectivity')
   points[node]=p
 if np.any(points<0):raise ValueError('Unmapped P2 nodes')
 return points


def nested_transfer(coarse,fine):
 ratios=np.array(fine.cells)//np.array(coarse.cells)
 if not np.array_equal(ratios,ratios[:1].repeat(3))or ratios[0]!=2 or not np.array_equal(np.array(coarse.cells)*ratios,fine.cells):raise ValueError('Declared uniform factor-two lattice refinement required')
 C=lattice(coarse)*2;F=lattice(fine);corner=C[coarse.tets];D=np.swapaxes(corner[:,1:]-corner[:,:1],-1,-2);inverse=np.linalg.inv(D.astype(float))
 # Freudenthal tetrahedra have an exact dyadic inverse at refinement2.
 if not np.array_equal(inverse*4,np.rint(inverse*4))or not np.array_equal(inverse@D,np.broadcast_to(np.eye(3),D.shape)):raise ValueError('Exact dyadic parent-coordinate proof failed')
 def bary(points):
  tail=np.einsum('tij,tkj->tki',inverse,points[None]-corner[:,None,0]);return np.concatenate([1-tail.sum(-1,keepdims=True),tail],-1)
 parents=[]
 for tet in fine.tets:
  B=bary(F[tet]);valid=np.flatnonzero(np.all(B>=0,axis=(1,2)))
  if not len(valid):raise ValueError('Fine tetrahedron crosses coarse partition')
  parents.append(int(valid[0]))
 B=bary(F);inside=np.all(B>=0,axis=-1)
 if not inside.any(0).all():raise ValueError('Fine P2 node outside coarse partition')
 parent=inside.argmax(0);rows=np.arange(len(F));basis=shapes(B[parent,rows]);P=sparse.coo_matrix((basis.ravel(),(np.repeat(rows,10),coarse.elements[parent].ravel())),shape=(len(F),len(C))).tocsr();P.eliminate_zeros()
 error=float(np.max(abs(P@coarse.rest-fine.rest)));eps=np.finfo(float).eps
 # Ten weighted shape contributions plus physical grid-coordinate rounding.
 geom_bound=64*eps*max(1.,float(np.max(abs(coarse.rest))))
 if error>geom_bound:raise ValueError('Physical coordinate reconstruction exceeds independent FP64 budget')
 face_parents=[];covered=np.zeros(len(coarse.faces));fineX=F[fine.faces]
 for vertices in fineX:
  matches=[]
  for k,face in enumerate(coarse.faces):
   tri=C[face];normal=np.cross(tri[1]-tri[0],tri[2]-tri[0]);normal=normal.astype(np.int64)
   if np.any((vertices-tri[0])@normal):continue
   gram=np.column_stack([tri[1]-tri[0],tri[2]-tri[0]]);uv=np.linalg.lstsq(gram, (vertices-tri[0]).T,rcond=None)[0].T
   # Here material-face barycentrics must be exact dyadics; do not silently
   # round arbitrary geometry to claim containment.
   uv=np.rint(uv*4)/4
   if np.array_equal(uv@gram.T+tri[0],vertices)and np.all(uv>=0)and np.all(uv.sum(1)<=1):matches.append(k)
  if len(matches)!=1:raise ValueError('Fine boundary face lacks unique coarse face')
  k=matches[0];face_parents.append(k);a=.5*np.linalg.norm(np.cross(vertices[1]-vertices[0],vertices[2]-vertices[0]));covered[k]+=a
 areas=np.array([.5*np.linalg.norm(np.cross(C[f[1]]-C[f[0]],C[f[2]]-C[f[0]]))for f in coarse.faces])
 if not np.array_equal(covered,areas):raise ValueError('Coarse face partition areas not preserved')
 return P,dict(coarse_tets=len(coarse.tets),fine_tets=len(fine.tets),all_fine_tets_have_one_parent=True,exact_dyadic_material_partition=True,physical_coordinate_error_m=error,physical_coordinate_roundoff_budget_m=geom_bound,all_fine_boundary_faces_have_one_parent=True,coarse_face_area_partition_exact=True,P_nnz=P.nnz)


def polynomial_checks(coarse,fine,P):
 def basis(X):
  x,y,z=X.T;return np.column_stack([np.ones(len(X)),x,y,z,x*x,y*y,z*z,x*y,x*z,y*z])
 a,b=basis(coarse.rest),basis(fine.rest);error=np.max(abs(P@a-b),axis=0);budget=64*np.finfo(float).eps*np.maximum(1.,np.max(abs(b),axis=0))
 if np.any(error>budget):raise ValueError('P2 constant/affine/quadratic reproduction failed')
 return dict(maximum_error_by_polynomial=error.tolist(),independent_roundoff_budgets=budget.tolist(),passed=True)


def mass_check(coarse,fine,P):
 Mc,Mf=scalar_mass(coarse),scalar_mass(fine);restricted=(P.T@Mf@P).tocsr();error=(restricted-Mc).tocsr();norm=float(np.sqrt(error.multiply(error).sum()));absolute=(abs(P).T@abs(Mf)@abs(P));ops=2*int(np.diff(P.tocsc().indptr).max())*int(np.diff(Mf.indptr).max())+128;eps=np.finfo(float).eps;gamma=ops*eps/(1-ops*eps);bound=float(gamma*(absolute.sum()+abs(Mc).sum()))
 if norm>bound:raise ValueError('Signed full consistent-mass restriction failed')
 return dict(signed_entries_retained=True,no_dense_fine_mass=True,Mc_nnz=Mc.nnz,Mf_nnz=Mf.nnz,restriction_Frobenius_error_kg=norm,independent_operation_budget=ops,roundoff_bound_kg=bound,passed=True)


def surface_geometry(m):
 edge={tuple(sorted((int(t[a]),int(t[b])))):int(e[4+k])for t,e in zip(m.tets,m.elements)for k,(a,b)in enumerate(EDGES)}
 faces=np.array([list(f)+[edge[tuple(sorted((int(f[a]),int(f[b]))))]for a,b in ((0,1),(0,2),(1,2))]for f in m.faces]);X=m.rest[m.faces];areas=.5*np.linalg.norm(np.cross(X[:,1]-X[:,0],X[:,2]-X[:,0]),axis=1)
 return faces,areas


def contact(q,model,*,adaptive=False,area_scale=1.):
 faces,areas=surface_geometry(model);g=np.zeros_like(q);energy=0.;estimated_E=0.;estimated_g=0.;skipped=0
 for face,A in zip(faces,areas):
  z=np.ascontiguousarray(q[face,2]);bern=np.r_[z[:3],[2*z[3+k]-.5*(z[a]+z[b])for k,(a,b)in enumerate(((0,1),(0,2),(1,2)))]]
  if np.all(bern>=0):skipped+=1;continue
  r=integrate_face_energy_gradient(z,float(A*area_scale))if adaptive else integrate_face_fixed(z,float(A*area_scale),outer_order=128)
  energy+=r['energy_J'];np.add.at(g[:,2],face,r['gradient_N']);estimated_E+=r.get('estimated_energy_error_J',0.);estimated_g+=float(np.linalg.norm(r.get('estimated_gradient_errors_N',np.zeros(6))))
 return dict(energy=energy,gradient=g,adaptive_estimated_E_error_J=estimated_E,adaptive_sum_local_g_error_estimates_N=estimated_g,provably_open_faces_skipped=skipped)


def compare(coarse_result,fine_result,P):
 E=abs(coarse_result['energy']-fine_result['energy']);g=float(np.linalg.norm(coarse_result['gradient']-P.T@fine_result['gradient']))
 return dict(energy_difference_J=E,restricted_gradient_L2_difference_N=g,energy_gate_J=ENERGY_GATE_J,gradient_gate_N=GRADIENT_GATE_N,passed=E<=ENERGY_GATE_J and g<=GRADIENT_GATE_N)
