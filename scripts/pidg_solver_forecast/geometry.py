"""Image-derived occupied-voxel volume and exact dense Gaussian binding.
No released point cloud or future image is consumed. Volume is an inferred proxy.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import label,binary_fill_holes,binary_dilation
from scripts.solver_driven_gaussians.dynamics import block_mesh

def volume(points,opacity,resolution=12):
 p=np.asarray(points);selected=p[np.asarray(opacity).ravel()>.1]
 if len(selected)<100:raise ValueError('Insufficient supported appearance points')
 lo,hi=np.quantile(selected,[.01,.99],axis=0);length=float(max(hi-lo));h=length/resolution;lo-=h;hi+=h;cells=np.maximum(2,np.ceil((hi-lo)/h).astype(int));dims=cells*h
 x,e=block_mesh(tuple(cells),tuple(dims));x+=lo+np.r_[0,dims[1:]/2]
 centers=lo+(np.stack(np.meshgrid(*[np.arange(n)+.5 for n in cells],indexing='ij'),-1).reshape(-1,3))*h
 dist,_=cKDTree(selected).query(centers);occ=(dist<h*1.2).reshape(tuple(cells));occ=binary_fill_holes(occ);labs,n=label(occ)
 counts=np.bincount(labs.ravel());admit=[i for i in range(1,n+1) if counts[i]>=max(4,.005*counts[1:].sum())];occ=np.isin(labs,admit)
 # Disconnected objects remain disconnected; do not bridge by a global convex hull.
 keep=np.repeat(occ.ravel(),6);e=e[keep];tetlabels=np.repeat(labs.ravel()[occ.ravel()],6);pairs=np.stack([np.repeat(tetlabels,4),e.ravel()],1);unique,inverse=np.unique(pairs,axis=0,return_inverse=True);nodes=x[unique[:,1]];e=inverse.reshape(-1,4)
 full_to_kept=np.full(len(keep),-1);full_to_kept[np.flatnonzero(keep)]=np.arange(keep.sum())
 import itertools
 perms=list(itertools.permutations(range(3)));lookup={p:i for i,p in enumerate(perms)}
 def bind(q):
  q=np.asarray(q);grid=(q-lo)/h;cell=np.clip(np.floor(grid).astype(int),0,cells-1);frac=grid-cell;perm=np.argsort(-np.clip(frac,0,1),axis=1,kind='stable');indices=np.ravel_multi_index(cell.T,tuple(cells))*6+np.array([lookup[tuple(v)] for v in perm]);tet=full_to_kept[indices];outside=tet<0
  if outside.any():
   tree=cKDTree(nodes[e].mean(1));_,near=tree.query(q[outside],k=min(16,len(e)));near=np.atleast_2d(near) if near.ndim==1 else near;best=np.zeros(len(near),int);score=np.full(len(near),np.inf)
   for j in range(near.shape[1]):
    verts=nodes[e[near[:,j]]];b=np.linalg.solve((verts[:,1:]-verts[:,:1]).transpose(0,2,1),(q[outside]-verts[:,0])[...,None])[...,0];N=np.c_[1-b.sum(1),b];s=np.maximum(-N,0).sum(1);take=s<score;best[take]=near[take,j];score[take]=s[take]
   tet[outside]=best
  verts=nodes[e[tet]];b=np.linalg.solve((verts[:,1:]-verts[:,:1]).transpose(0,2,1),(q-verts[:,0])[...,None])[...,0];N=np.c_[1-b.sum(1),b];return tet,N
 tet,N=bind(p);node_labels=labs.ravel()[np.flatnonzero(occ.ravel())];tetlabels=np.repeat(node_labels,6)
 return dict(nodes=nodes,elements=e,bind_tet=tet,bind_N=N,node_origin=lo,voxel_size=h,cells=cells,component_labels=tetlabels,components=len(admit),extrapolated_fraction=float(((N< -1e-7)|(N>1+1e-7)).any(1).mean()),length=length)

def separated_volume(points,opacity,features,count,resolution=8):
 """Fitting-frame-assisted instance count; spatial/colour clustering assigns splats.
 Count is declared from visible bodies in the first fitting RGB, not future labels.
 """
 from scipy.cluster.vq import kmeans2
 p=np.asarray(points);op=np.asarray(opacity).ravel();lo,hi=np.quantile(p[op>.1],[.01,.99],axis=0);length=float(max(hi-lo));color=np.clip(np.asarray(features)[:,0]*.28209479177387814+.5,0,1);color=color/np.maximum(color.sum(1,keepdims=True),.01);desc=np.c_[(p-(lo+hi)/2)/length,.6*color];active=np.flatnonzero(op>.1);rng=np.random.default_rng(7);sample=rng.choice(active,min(len(active),12000),replace=False);np.random.seed(7);centers,_=kmeans2(desc[sample],count,iter=60,minit='++');groups=cKDTree(centers).query(desc)[1];nodes=[];elements=[];labels=[];bind_tet=np.empty(len(p),int);bind_N=np.empty((len(p),4));node_offset=0;tet_offset=0;offset=0;details=[]
 for group in range(count):
  selected=np.flatnonzero(groups==group);v=volume(p[selected],op[selected],resolution=resolution);nodes.append(v['nodes']);elements.append(v['elements']+node_offset);labels.append(v['component_labels']+offset);bind_tet[selected]=v['bind_tet']+tet_offset;bind_N[selected]=v['bind_N'];details.append(dict(group=group,gaussians=len(selected),components=v['components'],nodes=len(v['nodes'])));node_offset+=len(v['nodes']);tet_offset+=len(v['elements']);offset+=int(v['component_labels'].max())
 return dict(nodes=np.concatenate(nodes),elements=np.concatenate(elements),component_labels=np.concatenate(labels),bind_tet=bind_tet,bind_N=bind_N,components=sum(d['components'] for d in details),extrapolated_fraction=float(((bind_N< -1e-7)|(bind_N>1+1e-7)).any(1).mean()),length=length,instance_groups=groups,instance_details=details)
