"""Prospective extended P2 reference v02; no automatic cases.

v01 remains immutable: its medium-grid and axial-strain negatives motivate
levels2/4/8 and axial force .18N per face. Normal .10N, material prior, geometry,
foundation, kernels and acceptance thresholds are unchanged. This is explicit
design tuning from development results, not a reclassification of v01.

Affine T10 reference geometry; consistent signed P2 surface Gram foundation and
load coefficients are never clipped or interpreted as particle masses. Reuses
reviewed P2 shape/NH kernels, retaining positive24/80 quadrature and node guards.
"""
from functools import lru_cache
import itertools
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from experiments import campaign_surface_traction_static_v01 as shared
from experiments.campaign_static_p2_block import EDGES,enrich,shapes,shape_gradients,QuadraticBlock
from experiments.campaign_p2_keast_quadrature import positive_keast24
from experiments.campaign_p2_exact_quadrature import exact_degree6_rule


SURFACE_QUADRATURE=16
GATES={k:v for k,v in shared.GATES.items()if k not in('support_12_vs_16_relative_Frobenius','support_8_vs_12_relative_Frobenius')}
GATES.update(support_surface_production_vs_refined_relative_Frobenius=1e-10,support_surface_production_order=16,support_surface_refined_order=20)


def triangle_rule(order):
    if type(order)is not int or order not in(4,8,12,16,20):raise ValueError('DeclaredpositiveDuffyorder')
    x,w=np.polynomial.legendre.leggauss(order);x=(x+1)/2;w=w/2;r,t=np.meshgrid(x,x,indexing='ij');wr,wt=np.meshgrid(w,w,indexing='ij')
    return np.column_stack([(1-r).ravel(),(r*(1-t)).ravel(),(r*t).ravel()]),(2*r*wr*wt).ravel()


def volume_rule(order):
    if order==24:return positive_keast24()
    if order==80:return exact_degree6_rule()
    raise ValueError('Positive declared24 or80point rule')


@lru_cache(maxsize=4)
def _topology(n):
    if type(n)is not int or n not in(1,2,4,8):raise ValueError('P2 base n1/2/4/8 only')
    X=np.array(list(itertools.product(np.linspace(0,.36,4*n+1),np.linspace(0,.08,n+1),np.linspace(0,.08,n+1))))
    node=lambda i,j,k:i*(n+1)**2+j*(n+1)+k;tets=[]
    for i,j,k in itertools.product(range(4*n),range(n),range(n)):
        for perm in itertools.permutations(range(3)):
            vertex=np.array([i,j,k]);path=[vertex.copy()]
            for axis in perm:vertex=vertex.copy();vertex[axis]+=1;path.append(vertex)
            ids=np.array([node(*v)for v in path])
            if np.linalg.det((X[ids[1:]]-X[ids[0]]).T)<0:ids[[1,2]]=ids[[2,1]]
            tets.append(ids)
    tets=np.array(tets);counts={}
    for tet in tets:
        for face in itertools.combinations(tet,3):
            key=tuple(sorted(face));counts[key]=counts.get(key,0)+1
    faces=np.array([f for f,count in counts.items()if count==1]);rest,elements=enrich(X,tets)
    edges=np.unique(np.sort(tets[:,np.asarray(EDGES)].reshape(-1,2),axis=1),axis=0);codes=edges[:,0]*len(X)+edges[:,1]
    faceedges=np.stack([faces[:,[0,1]],faces[:,[0,2]],faces[:,[1,2]]],axis=1);facecodes=faceedges[:,:,0]*len(X)+faceedges[:,:,1]
    faces6=np.concatenate([faces,len(X)+np.searchsorted(codes,facecodes)],axis=1)
    fixed=np.flatnonzero((rest[:,0]==0)|(rest[:,0]==.36));free=np.flatnonzero(np.repeat((rest[:,0]!=0)&(rest[:,0]!=.36),3))
    return dict(corner_X=X,corner_tets=tets,corner_faces=faces,X=rest,elements=elements,faces=faces6,edge_codes=codes,fixed=fixed,free=free)


def geometry(n):
    t=_topology(n);return tuple(t[k].copy()for k in('X','elements','faces','fixed','free'))


def triangle_shapes(b):
    return np.concatenate([b*(2*b-1),np.column_stack([4*b[:,0]*b[:,1],4*b[:,0]*b[:,2],4*b[:,1]*b[:,2]])],axis=1)


def consistent_support(X,faces,*,order=SURFACE_QUADRATURE):
    bary,qw=triangle_rule(order);N=triangle_shapes(bary);local=[];mass=0.;G=shared.gaussian_surface_normalization()
    for first in range(0,len(faces),128):
        vertices=X[faces[first:first+128,:3]];area=np.linalg.norm(np.cross(vertices[:,1]-vertices[:,0],vertices[:,2]-vertices[:,0]),axis=1)/2
        q=np.einsum('qa,fai->fqi',bary,vertices);g=np.exp(-.5*np.sum((q-shared.SUPPORT_CENTER)**2,axis=2)/shared.SUPPORT_WIDTH**2)
        local.append(np.einsum('f,q,fq,qa,qb->fab',area,qw,g,N,N)/G);mass+=float(np.einsum('f,q,fq->',area,qw,g))/G
    value=np.concatenate(local);rows=np.repeat(faces,6,axis=1).ravel();cols=np.tile(faces,(1,6)).ravel()
    W=sparse.coo_matrix((value.ravel(),(rows,cols)),shape=(len(X),len(X))).tocsr();W.sum_duplicates()
    return W,dict(order=order,exact_normalization_m2=G,integrated_normalized_mass=mass,normalization_relative_error=abs(mass-1),signed_entries_allowed=True,negative_entries=int(np.count_nonzero(W.data<0)))


def _clip(poly,xvalues,bound,keep_greater):
    answer=[]
    for a,b in zip(poly,np.roll(poly,-1,axis=0)):
        xa=float(a@xvalues);xb=float(b@xvalues);ia=xa>=bound-1e-15 if keep_greater else xa<=bound+1e-15;ib=xb>=bound-1e-15 if keep_greater else xb<=bound+1e-15
        if ia:answer.append(a)
        if ia!=ib:answer.append(a+(b-a)*(bound-xa)/(xb-xa))
    return np.asarray(answer)


def surface_traction(n,direction,total_force_N,*,face='top',center_x=.18,order=SURFACE_QUADRATURE):
    X,_,faces,_,_=geometry(n);direction=np.asarray(direction,float)
    if direction.shape!=(3,)or not np.isfinite(direction).all()or not np.isclose(np.linalg.norm(direction),1,atol=1e-13,rtol=0)or not np.isfinite(total_force_N)or total_force_N<0 or face not in('top','bottom')or not np.isfinite(center_x)or not .045<=center_x<=.315:raise ValueError('Finite declared patch, forceN, unitdirection')
    left,right=center_x-.045,center_x+.045;z=.08 if face=='top'else 0.;bary,qw=triangle_rule(order);weights=np.zeros(len(X));area_total=0.;subtriangles=0
    for ids in faces:
        vertices=X[ids[:3]]
        if not np.all(abs(vertices[:,2]-z)<1e-14):continue
        if vertices[:,0].max()<left or vertices[:,0].min()>right:continue
        poly=_clip(np.eye(3),vertices[:,0],left,True)
        if len(poly)<3:continue
        poly=_clip(poly,vertices[:,0],right,False)
        if len(poly)<3:continue
        for j in range(1,len(poly)-1):
            sub=np.array([poly[0],poly[j],poly[j+1]]);xyz=sub@vertices;area=np.linalg.norm(np.cross(xyz[1]-xyz[0],xyz[2]-xyz[0]))/2
            if area<1e-24:continue
            parent=bary@sub;q=parent@vertices;profile=np.sin(np.pi*(q[:,0]-left)/.09)**2;N=triangle_shapes(parent)
            local=area*np.einsum('q,q,qa->a',qw,profile,N)/.0036;np.add.at(weights,ids,local);area_total+=area;subtriangles+=1
    force=total_force_N*weights[:,None]*direction
    return force,dict(patch_area_m2=area_total,exact_profile_integral_m2=.0036,integrated_weight_mass=float(weights.sum()),weights=weights,subtriangles=subtriangles,center_x_m=center_x,face=face,order=order,signed_nodal_coefficients=True)


def qualification_loads(n,*,order=SURFACE_QUADRATURE):
    normal,_=surface_traction(n,[0,0,-1],.1,order=order);top,_=surface_traction(n,[1,0,0],.18,order=order);bottom,_=surface_traction(n,[1,0,0],.18,face='bottom',order=order)
    return dict(normal=normal,paired_axial=top+bottom)


def _bulk_assembly(rest,elements,corner_X,corner_tets,order):
    dm=(corner_X[corner_tets[:,1:]]-corner_X[corner_tets[:,:1]]).transpose(0,2,1);det=np.linalg.det(dm)
    if np.any(det<=0):raise ValueError('Positive affine geometry')
    volumes=det/6;g=np.linalg.inv(dm);linear_gradients=np.concatenate([-g.sum(1,keepdims=True),g],axis=1)
    nu=.35;mu=1/(2*(1+nu));lam=nu/((1+nu)*(1-2*nu));D=np.zeros((6,6));D[:3,:3]=lam;D[:3,:3]+=2*mu*np.eye(3);D[3:,3:]=mu*np.eye(3)
    local=np.zeros((len(elements),30,30));bary,qw=volume_rule(order)
    for b,w in zip(bary,qw):
        gradients=shape_gradients(b[None],linear_gradients);B=np.zeros((len(elements),6,30))
        for a in range(10):
            dx,dy,dz=gradients[:,a].T;k=3*a;B[:,0,k]=dx;B[:,1,k+1]=dy;B[:,2,k+2]=dz;B[:,3,k+1]=dz;B[:,3,k+2]=dy;B[:,4,k]=dz;B[:,4,k+2]=dx;B[:,5,k]=dy;B[:,5,k+1]=dx
        local+=(B.transpose(0,2,1)@D@B)*volumes[:,None,None]*w
    dofs=(elements[:,:,None]*3+np.arange(3)).reshape(-1,30);rows=np.repeat(dofs,30,axis=1).ravel();cols=np.tile(dofs,(1,30)).ravel()
    K=sparse.coo_matrix((local.ravel(),(rows,cols)),shape=(rest.size,rest.size)).tocsr();K.sum_duplicates()
    return K,linear_gradients,volumes,rows,cols


class SparseStaticForward:
    def __init__(self,n=1,*,quadrature_order=24):
        t=_topology(n);self.n=n;self.X=t['X'].copy();self.rest=self.X;self.elements=t['elements'].copy();self.tets=self.elements;self.tets10=self.elements;self.faces=t['faces'].copy();self.fixed=t['fixed'].copy();self.free=t['free'].copy();self.corner_X=t['corner_X'].copy();self.corner_tets=t['corner_tets'].copy();self.quadrature_order=quadrature_order
        self.Kunit,self.linear_gradients,self.volumes,self.rows,self.cols=_bulk_assembly(self.X,self.elements,self.corner_X,self.corner_tets,quadrature_order)
        self.W_scalar,self.support_quadrature=consistent_support(self.X,self.faces);self.W=sparse.kron(self.W_scalar,sparse.eye(3),format='csr');self.weights=np.asarray(self.W_scalar.sum(axis=1)).ravel()
        self.Kunit_free=self.Kunit[self.free][:,self.free].tocsc();self.W_free=self.W[self.free][:,self.free].tocsc()
    def staticforward(self,theta,load,*,return_diagnostics=False):
        theta=np.asarray(theta,float);load=np.asarray(load,float)
        if theta.ndim!=2 or theta.shape[1]!=2 or not len(theta)or not np.isfinite(theta).all()or np.any(theta<=0):raise ValueError('Physical[E,K]positive[B,2]')
        if load.ndim==2:load=load[None]
        if load.ndim!=3 or load.shape[1:]!=self.X.shape or not np.isfinite(load).all():raise ValueError('Knownforce[R,N,3]')
        rhs=load.reshape(len(load),-1).T;fields=[];diag=[]
        for E,K in theta:
            u=np.zeros_like(rhs);u[self.free]=splu(E*self.Kunit_free+K*self.W_free).solve(rhs[self.free]);r=(E*self.Kunit+K*self.W)@u-rhs;error=float(np.max(abs(r[self.free])))
            if not np.isfinite(u).all()or error>shared.GATES['free_residual_N']:raise FloatingPointError('Unmodified static residual failed')
            fields.append(u.T.reshape(len(load),len(self.X),3));diag.append(dict(maximum_free_residual_N=error,exact_clamp_zeros=bool(np.all(u.reshape(len(self.X),3,-1)[self.fixed]==0))))
        result=np.asarray(fields);return (result,diag)if return_diagnostics else result


P2StaticForward=SparseStaticForward


def _query(n,points):
    p=np.asarray(points,float);t=_topology(n);limit=np.array([.36,.08,.08]);tol=64*np.finfo(float).eps
    if p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all()or np.any(p < -tol)or np.any(p>limit+tol):raise ValueError('Physicalpoints insideblock')
    scale=limit/[4*n,n,n];x=np.clip(p,0,limit)/scale;cell=np.minimum(np.floor(x).astype(int),[4*n-1,n-1,n-1]);f=x-cell;perm=np.argsort(-f,axis=1,kind='stable');ordered=np.take_along_axis(f,perm,axis=1);bary=np.column_stack([1-ordered[:,0],ordered[:,0]-ordered[:,1],ordered[:,1]-ordered[:,2],ordered[:,2]])
    vertices=[cell.copy()];current=cell.copy()
    for k in range(3):current=current.copy();current[np.arange(len(p)),perm[:,k]]+=1;vertices.append(current)
    v=np.stack(vertices,axis=1);corners=v[:,:,0]*(n+1)**2+v[:,:,1]*(n+1)+v[:,:,2]
    edge=np.sort(corners[:,np.asarray(EDGES)],axis=2);codes=edge[:,:,0]*len(t['corner_X'])+edge[:,:,1];mid=len(t['corner_X'])+np.searchsorted(t['edge_codes'],codes)
    return np.concatenate([corners,mid],axis=1),bary


def field_queries(n,u,points):
    ids,bary=_query(n,points);u=np.asarray(u,float)
    if u.shape[-2:]!=_topology(n)['X'].shape or not np.isfinite(u).all():raise ValueError('FiniteP2field')
    return np.einsum('qa,...qac->...qc',shapes(bary),u[...,ids,:])


def point_query_matrix(forward,points):
    ids,bary=_query(forward.n,points);N=shapes(bary);q=len(ids);globaldof=(ids[:,:,None]*3+np.arange(3)).reshape(q,-1)
    row=np.broadcast_to(np.arange(3*q).reshape(q,1,3),(q,10,3)).ravel();col=globaldof.ravel();value=np.broadcast_to(N[:,:,None],(q,10,3)).ravel();inverse=np.full(forward.X.size,-1,int);inverse[forward.free]=np.arange(len(forward.free));freecol=inverse[col];keep=freecol>=0
    return sparse.coo_matrix((value[keep],(row[keep],freecol[keep])),shape=(3*q,len(forward.free))).tocsr()


def bulk_query_blocks(n=8,*,order=24,tet_chunk=128):
    t=_topology(n);X=t['corner_X'];tet=t['corner_tets'];bary,w=volume_rule(order);vol=np.linalg.det((X[tet[:,1:]]-X[tet[:,:1]]).transpose(0,2,1))/6
    for first in range(0,len(tet),tet_chunk):yield np.einsum('qa,tai->tqi',bary,X[tet[first:first+tet_chunk]]).reshape(-1,3),(vol[first:first+tet_chunk,None]*w[None]).ravel()


def strain_metrics(forward,u):
    u=np.asarray(u,float)
    if u.shape[-2:]!=forward.X.shape or not np.isfinite(u).all():raise ValueError('FiniteP2field')
    flat=u.reshape(-1,len(forward.X),3);b24,_=volume_rule(24);b80,_=volume_rule(80);checks=np.concatenate([np.eye(4),np.array([(np.eye(4)[i]+np.eye(4)[j])/2 for i,j in EDGES]),b24,b80]);minimum=np.full(len(flat),np.inf);maximum=np.zeros(len(flat))
    for b in checks:
        g=shape_gradients(b[None],forward.linear_gradients);G=np.einsum('beai,eaJ->beiJ',flat[:,forward.elements],g);eps=.5*(G+G.transpose(0,1,3,2));maximum=np.maximum(maximum,np.max(abs(np.linalg.eigvalsh(eps)),axis=(1,2)));minimum=np.minimum(minimum,np.linalg.det(np.eye(3)+G).min(1))
    return dict(minimum_J=minimum,maximum_absolute_symmetric_strain_eigenvalue=maximum,checked_nodes_and24and80=True,whole_element_J_certified=False)


class P2NonlinearReference(QuadraticBlock):
    @classmethod
    def from_forward(cls,forward,E,K,*,quadrature_order=24):
        if not np.isfinite([E,K]).all()or min(E,K)<=0:raise ValueError('PhysicalpositiveE,K')
        self=object.__new__(cls);self.forward=forward;self.rest=forward.X;self.elements=forward.elements;self.tets=forward.corner_tets;self.linear_gradients=forward.linear_gradients;self.volumes=forward.volumes;self.young=float(E);self.poisson=.35;self.minimum_J=.1;self.order=quadrature_order;self.bary,self.weights=volume_rule(quadrature_order);self.N=shapes(self.bary);b80,_=volume_rule(80);b24,_=volume_rule(24)
        self.check_bary=np.concatenate([np.eye(4),np.array([(np.eye(4)[i]+np.eye(4)[j])/2 for i,j in EDGES]),b24,b80]);self.rows=forward.rows;self.cols=forward.cols;self.mask=np.zeros_like(self.rest,dtype=bool);self.mask[forward.fixed]=True;self.free=forward.free;self.support_K=float(K);self.W=forward.W;self.load=np.zeros_like(self.rest)
        return self
    def evaluate(self,u,force=None,tangent=False):
        # Frozen parent Newton kernel calls evaluate(u, True) for a tangent.
        # Arrays in the second slot remain the explicit runner force API.
        if isinstance(force,(bool,np.bool_)):
            if tangent:raise ValueError('Ambiguous legacy tangent flag')
            tangent=bool(force);force=None
        force=self.load if force is None else np.asarray(force,float);u=np.asarray(u,float)
        if force.shape!=self.rest.shape or not np.isfinite(force).all():raise ValueError('Finiteknownforce')
        U,g,H,J=self.elastic(self.rest+u,tangent);Wu=(self.W@u.ravel()).reshape(u.shape);spring=.5*self.support_K*float(u.ravel()@Wu.ravel());work=float(np.sum(force*u));residual=g+self.support_K*Wu-force
        return dict(potential_J=U+spring-work,elastic_J=U,spring_J=spring,work_J=work,residual=residual,tangent=None if H is None else H+self.support_K*self.W,J=J,positions=self.rest+u)
    def solve(self,force,*,initial=None,maximum_iterations=40,force_tolerance_N=1e-10):
        self.load=np.asarray(force,float).copy();answer=super().solve(initial=initial,maximum_iterations=maximum_iterations,force_tolerance_N=force_tolerance_N)
        return answer['displacement_local_m'],dict(iterations=answer['iterations'],history=answer['history'],minimum_J=float(answer['J'].min()),maximum_free_residual_N=float(np.max(abs(answer['residual'].ravel()[self.free]))),sampled_J_only=True,search_J_guard=.1,accepted_physical_J_gate=shared.GATES['minimum_J'])


def support_quadrature_checks(n):
    X,_,faces,_,_=geometry(n);values={};meta={}
    for order in(8,12,16,20):values[order],meta[order]=consistent_support(X,faces,order=order)
    norm=lambda A:float(np.sqrt(np.sum(A.data**2)))
    return dict(production_order=SURFACE_QUADRATURE,normalization_relative_error=meta[SURFACE_QUADRATURE]['normalization_relative_error'],relative_16_vs_20=norm(values[16]-values[20])/norm(values[20]),relative_8_vs_12=norm(values[8]-values[12])/norm(values[12]),relative_12_vs_16=norm(values[12]-values[16])/norm(values[16]),signed_entries_allowed=True)


def design():
    return dict(schema='surface-p2-static-v02',base_n=[1,2,4,8],qualification_levels=[2,4,8],target_mesh_n=4,enriched_nodes=[81,425,2673,18785],free_DOF=[189,1125,7533,54621],elements_T10=[24,192,1536,12288],geometry='Affine reference T10; bothendfaces exactclamp; unchanged.36x.08x.08',loads='Smooth topnormal-.10N unchanged; pairedaxial+.18N perface total.36N; exactclippedpatchx[.135,.225],full y',foundation='Unchanged analyticnormalized surface Gaussian, consistent P2 Gram W; signed entries/no lumping/no clipping',bulk_rule='PositiveKeast24; linear stiffness degree2 exactlyintegrated; independent positive80 crosscheck. LogNH requires numerical24vs80 validation.',surface_rule='PositiveDuffy16 independentlychecked20 onoriginalorclippedtriangles; originalP2face shapes evaluated inparent barycentric coordinates',gates=GATES,design_tuned_on_v01=True,correction='v01 medium adequacy failed14/20 and fine soft axial strain exceeded.05. New n8 checks target n4; axial reduced25percent before new force outcomes. Old negatives and gates preserved.',scope='New higherorder static finite reference; no automatic admission from v01. All priorT4/P2v01 failures preserved. No dynamic, continuumstress, calibratedinference oractivepolicy claim.',no_automatic_execution=True)
