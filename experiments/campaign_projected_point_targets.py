"""Mixture target moments/risk with exact candidate-point GP projection.

No observations, labels, fit, physics or covariance nugget enters. The global
GP mean/factor vanish at EACH explicit candidate pin. Fantasy risk never
allocates a [fantasy,particle,target-coordinate] tensor. Coordinates are
displacements, with layout [time,node,xyz]; covariance excludes sensor noise.
"""
from __future__ import annotations
import torch


class ProjectedPointTargets:
    def __init__(self,displacements,discrepancy_mean,factor,candidate_pins,layout):
        X=torch.as_tensor(displacements);m=torch.as_tensor(discrepancy_mean,device=X.device);B=torch.as_tensor(factor,device=X.device)
        pins=torch.as_tensor(candidate_pins,device=X.device)
        if (len(layout)!=3 or any(type(n) is not int or n<=0 for n in layout) or layout[2]!=3
                or X.ndim!=2 or not len(X) or X.shape[1]!=layout[0]*layout[1]*3
                or m.shape!=(X.shape[1],) or B.ndim!=2 or B.shape[0]!=X.shape[1] or B.shape[1]<1
                or any(v.dtype!=torch.float64 or not bool(torch.isfinite(v).all()) for v in [X,m,B])
                or pins.dtype!=torch.int64 or pins.shape!=(len(X),) or bool((pins<0).any()) or bool((pins>=layout[1]).any())):
            raise ValueError('Finite float64 X[P,D],m[D],B[D,R],int64pins[P], explicit(T,N,3) required')
        self.P,self.D,self.R=len(X),X.shape[1],B.shape[1];self.layout=tuple(layout);self.B=B;self.pins=pins
        self.unique_pins,self.inverse=torch.unique(pins,sorted=True,return_inverse=True)
        rows=self._pin_rows(pins)
        if bool((X.gather(1,rows)!=0).any()):raise ValueError('Candidate target pin must already be exactlyzero; no drift repair')
        # Projection is a model constraint on discrepancy, not an X repair.
        base=(X+m).scatter(1,rows,torch.zeros_like(rows,dtype=X.dtype))
        self.anchor=base[0].clone();self.centered=base-self.anchor
        self.base_norm=self.centered.square().sum(1)/self.D
        pinB=B[rows]
        self.cross=(self.centered@B-torch.einsum('pk,pkr->pr',self.centered.gather(1,rows),pinB))/self.D
        self.cross_bound=(self.centered.abs()@B.abs()+torch.einsum('pk,pkr->pr',self.centered.gather(1,rows).abs(),pinB.abs()))/self.D
        uniqueB=B[self._pin_rows(self.unique_pins)]
        self.G_unique=(B.T@B-torch.einsum('qki,qkj->qij',uniqueB,uniqueB))/self.D
        # Bound roundoff in the DIFFERENCE of Grams too, including a factor
        # supported only at projected pin rows. Never alter the Gram or C.
        self.G_bound=(B.abs().T@B.abs()+torch.einsum('qki,qkj->qij',uniqueB.abs(),uniqueB.abs()))/self.D
        self.G=self.G_unique.index_select(0,self.inverse)

    def _pin_rows(self,pins):
        T,N,_=self.layout
        return ((torch.arange(T,device=pins.device)[None,:,None]*N+pins[:,None,None])*3+torch.arange(3,device=pins.device)[None,None]).flatten(1)

    def _float(self,value):
        a=torch.as_tensor(value,device=self.B.device)
        if a.dtype!=torch.float64 or not bool(torch.isfinite(a).all()):raise ValueError('Finite float64 statistics required')
        return a

    def _weights(self,value):
        w=self._float(value)
        if w.shape[-1:]!=(self.P,) or w.ndim not in (1,2) or bool((w<0).any()) or not bool((abs(w.sum(-1)-1)<=64*torch.finfo(w.dtype).eps).all()):raise ValueError('Normalized nonnegative particle weights required; never renormalized')
        return w

    def _covariance(self,value):
        C=self._float(value)
        if C.ndim!=3 or C.shape[1:]!=(self.R,self.R) or not len(C):raise ValueError('Covariancegroups[G,R,R] required')
        scale=C.abs().amax((-1,-2)).clamp_min(torch.finfo(C.dtype).tiny)
        eps=64*torch.finfo(C.dtype).eps
        if bool(((C-C.transpose(-1,-2)).abs().amax((-1,-2))>eps*scale).any()) or bool((torch.linalg.eigvalsh(C)[:,0]<-eps*scale).any()):raise ValueError('Symmetric PSD latentcovariance required; no covariance repair')
        return C

    def _index(self,value,groups):
        idx=torch.as_tensor(value,device=self.B.device)
        if idx.dtype!=torch.int64 or idx.shape!=(self.P,) or bool((idx<0).any()) or bool((idx>=groups).any()):raise ValueError('Explicit group_index[P] required')
        return idx

    @staticmethod
    def _nonnegative(value,absolute_bound):
        """Only scale-relative floating roundoff can be clamped; report count."""
        tolerance=128*torch.finfo(value.dtype).eps*absolute_bound.clamp_min(torch.finfo(value.dtype).tiny)
        if bool((value < -tolerance).any()):raise ValueError('Substantive negative variance/risk; no variance repair')
        count=int((value<0).sum())
        return torch.where(value<0,torch.zeros_like(value),value),count

    def _coefficients(self,w,a):
        weighted=w[:,:,None]*a
        per_node=torch.zeros((len(w),self.layout[1],self.R),dtype=self.B.dtype,device=self.B.device)
        per_node.index_add_(1,self.pins,weighted)
        return weighted.sum(1),per_node

    def _mean_chunk(self,w,total_a,node_a,start,end):
        B=self.B[start:end];nodes=(torch.arange(start,end,device=B.device)//3)%self.layout[1]
        return w@self.centered[:,start:end]+total_a@B.T-torch.einsum('fkr,kr->fk',node_a[:,nodes],B)

    def moments(self,weights,latent_means,covariance_groups,group_index,*,feature_chunk=512):
        """Return full-mixture mean/std, between-mean and conditional GP variance.

        Native LatentConditional fields: latent_mean, covariance_groups,
        group_index. Unused covariance groups are valid. No observation noise.
        Temporary particle-coordinate matrices cover only one feature chunk.
        """
        w=self._weights(weights);a=self._float(latent_means);C=self._covariance(covariance_groups);idx=self._index(group_index,len(C))
        if w.ndim!=1 or a.shape!=(self.P,self.R) or type(feature_chunk) is not int or feature_chunk<=0:raise ValueError('Single state weights[P],means[P,R],positive featurechunk required')
        means=[];between=[];within=[];clamps=0
        for start in range(0,self.D,feature_chunk):
            end=min(start+feature_chunk,self.D);B=self.B[start:end]
            nodes=(torch.arange(start,end,device=B.device)//3)%self.layout[1]
            mask=nodes[None]==self.pins[:,None]
            relative=self.centered[:,start:end]+(a@B.T).masked_fill(mask,0)
            mean=w@relative
            variance_between=torch.einsum('p,pk->k',w,(relative-mean).square())
            diagonal=torch.einsum('ki,gij,kj->gk',B,C,B)
            bound=torch.einsum('ki,gij,kj->gk',B.abs(),C.abs(),B.abs())
            diagonal,n=self._nonnegative(diagonal,bound);clamps+=n
            variance_within=w@(diagonal.index_select(0,idx).masked_fill(mask,0))
            means.append(self.anchor[start:end]+mean);between.append(variance_between);within.append(variance_within)
        mean=torch.cat(means);vb=torch.cat(between);vw=torch.cat(within);total=vb+vw
        return dict(mean_displacement=mean,std=torch.sqrt(total),total_variance=total,
            between_particle_variance=vb,within_gp_variance=vw,roundoff_clamps=clamps)

    def _trace(self,C,idx):
        table=torch.einsum('qij,gij->qg',self.G_unique,C)
        bound=torch.einsum('qij,gij->qg',self.G_bound,C.abs())
        table,n=self._nonnegative(table,bound)
        return table[self.inverse,idx],n

    def risk(self,weights,latent_means,covariance_groups=None,group_index=None,*,conditional_trace=None,feature_chunk=4096):
        """Normalized physical target MSE variance for F independent fantasies.

        C can be shared[G,R,R], stacked[F,G,R,R], or ragged list of[Gf,R,R].
        Group indices can be shared[P], stacked[F,P], or a list of[P].
        Alternatively conditional_trace[F,P] supplies Tr(G_pin*C), already
        normalized by D; it must be finite/nonnegative. No F*P*D allocations.
        """
        w=self._weights(weights);a=self._float(latent_means)
        if w.ndim!=2 or a.shape!=(len(w),self.P,self.R) or type(feature_chunk) is not int or feature_chunk<=0:raise ValueError('Fantasy weights[F,P]/means[F,P,R] required')
        F=len(w);clamps=0
        if conditional_trace is not None:
            if covariance_groups is not None:raise ValueError('Supply grouped covariance OR normalized conditionaltrace, not both')
            traces=self._float(conditional_trace)
            if traces.shape!=w.shape or bool((traces<0).any()):raise ValueError('Finite nonnegative normalized conditionaltrace[F,P] required')
        else:
            if covariance_groups is None or group_index is None:raise ValueError('Latent covariance/group information required')
            Cs=covariance_groups;indices=group_index
            ragged=isinstance(Cs,(list,tuple))
            if ragged:
                if len(Cs)!=F:raise ValueError('One ragged covariancebank per fantasy required')
            else:
                Cs=self._float(Cs)
                if Cs.ndim not in (3,4) or (Cs.ndim==4 and len(Cs)!=F):raise ValueError('Shared/stacked latent covariancegroups required')
            if isinstance(indices,(list,tuple)):
                if len(indices)!=F:raise ValueError('One groupindex vector per fantasy required')
                shared_index=False
            else:
                indices=torch.as_tensor(indices,device=self.B.device);shared_index=indices.ndim==1
                if not shared_index and indices.shape!=(F,self.P):raise ValueError('Shared or fantasy groupindices required')
            trace_rows=[]
            for f in range(F):
                C=self._covariance(Cs[f] if ragged or Cs.ndim==4 else Cs)
                idx=self._index(indices if shared_index else indices[f],len(C))
                trace,n=self._trace(C,idx);clamps+=n;trace_rows.append(trace)
            traces=torch.stack(trace_rows)
        norm=self.base_norm[None]+2*torch.einsum('fpr,pr->fp',a,self.cross)+torch.einsum('fpi,pij,fpj->fp',a,self.G,a)
        second=(w*norm).sum(1);total_a,node_a=self._coefficients(w,a)
        mean_norm=torch.zeros(F,dtype=self.B.dtype,device=self.B.device)
        for start in range(0,self.D,feature_chunk):
            mean=self._mean_chunk(w,total_a,node_a,start,min(start+feature_chunk,self.D))
            mean_norm+=mean.square().sum(1)/self.D
        norm_bound=self.base_norm[None]+2*torch.einsum('fpr,pr->fp',a.abs(),self.cross_bound)
        norm_bound+=torch.einsum('fpi,pij,fpj->fp',a.abs(),self.G_bound.index_select(0,self.inverse),a.abs())
        between,n=self._nonnegative(second-mean_norm,(w*norm_bound).sum(1)+mean_norm.abs());clamps+=n
        within=(w*traces).sum(1)
        return dict(risk=between+within,between_particle_risk=between,within_gp_risk=within,roundoff_clamps=clamps)
