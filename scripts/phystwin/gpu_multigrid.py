"""FP64 GPU V-cycle preconditioner; CPU PyAMG constructs the hierarchy."""
import numpy as np


class GPUMultigrid:
    def __init__(self,A,positions,device):
        import torch,pyamg
        self.t=torch;self.device=device
        x=positions-positions.mean(0);B=np.zeros((len(x),3,6));B[:,:,:3]=np.eye(3)
        for j in range(3):B[:,:,3+j]=np.cross(np.eye(3)[j],x)
        B=B.reshape(-1,6);B/=np.linalg.norm(B,axis=0)
        hierarchy=pyamg.smoothed_aggregation_solver(A.tobsr(blocksize=(3,3)),B=B,symmetry='symmetric',max_coarse=100)
        self.levels=[]
        for i,l in enumerate(hierarchy.levels):
            a=self.sparse(l.A);d=self.tensor(l.A.diagonal());entry=dict(A=a,invD=1/d)
            if i==len(hierarchy.levels)-1:
                entry['inverse']=self.tensor(np.linalg.inv(l.A.toarray()))
            else:
                entry.update(P=self.sparse(l.P),R=self.sparse(l.P.T))
                # Deterministic power estimate of the diagonally scaled spectrum.
                invsqrt=torch.rsqrt(d);v=self.tensor(np.sin(np.arange(len(d))+1.));v/=torch.linalg.vector_norm(v)
                for _ in range(30):
                    w=invsqrt*(a@(invsqrt*v));v=w/torch.linalg.vector_norm(w)
                largest=float(torch.dot(v,invsqrt*(a@(invsqrt*v))))
                entry['omega']=1./largest
            self.levels.append(entry)
    def tensor(self,a):return self.t.as_tensor(np.asarray(a),dtype=self.t.float64,device=self.device)
    def sparse(self,A):
        A=A.tocsr();A.sum_duplicates();A.sort_indices()
        return self.t.sparse_csr_tensor(self.t.as_tensor(A.indptr.astype(np.int64),device=self.device),self.t.as_tensor(A.indices.astype(np.int64),device=self.device),self.tensor(A.data),size=A.shape,device=self.device,check_invariants=True)
    def cycle(self,b,i=0):
        l=self.levels[i]
        if 'inverse' in l:return l['inverse']@b
        A=l['A'];x=self.t.zeros_like(b);D=l['omega']*l['invD']
        for _ in range(2):x=x+D*(b-A@x)
        x=x+l['P']@self.cycle(l['R']@(b-A@x),i+1)
        for _ in range(2):x=x+D*(b-A@x)
        return x
    def solve(self,A,b,rtol=1e-9,atol=1e-13,maxiter=1000):
        t=self.t;a=self.sparse(A);rhs=self.tensor(b);x=t.zeros_like(rhs);r=rhs.clone();z=self.cycle(r);p=z.clone();rz=t.dot(r,z)
        tol=max(atol,rtol*float(t.linalg.vector_norm(rhs)))
        if float(t.linalg.vector_norm(r))<=tol:return x.cpu().numpy(),0
        for iteration in range(maxiter):
            ap=a@p;den=t.dot(p,ap)
            if float(den)<=0 or float(rz)<=0:raise RuntimeError('GPU PCG positive-definiteness check failed')
            alpha=rz/den;x+=alpha*p;r-=alpha*ap
            if float(t.linalg.vector_norm(r))<=tol:
                # Recompute the original residual to detect recursive drift.
                r=rhs-a@x
                if float(t.linalg.vector_norm(r))<=tol:return x.cpu().numpy(),iteration+1
                z=self.cycle(r);p=z.clone();rz=t.dot(r,z);continue
            z=self.cycle(r);new=t.dot(r,z);p=z+(new/rz)*p;rz=new
        raise RuntimeError('GPU PCG iteration cap')
