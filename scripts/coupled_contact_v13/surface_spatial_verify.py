"""Independent fixed-dt mesh consistency audit; no production mapper imported."""
import itertools,json
import numpy as np

def probes(rest):
    lo=rest.min(0);hi=rest.max(0)
    sparse=np.array(list(itertools.product(*[np.linspace(a,b,3) for a,b in zip(lo,hi)])))
    dense=[]
    for axis in range(3):
        tangent=[i for i in range(3) if i!=axis]
        count=np.ceil((hi-lo)[tangent]/.003).astype(int)
        for side in (lo[axis],hi[axis]):
            for i,j in itertools.product(range(count[0]),range(count[1])):
                p=lo.copy();p[axis]=side
                for k,index,n in zip(tangent,(i,j),count):p[k]=lo[k]+(index+.5)*(hi[k]-lo[k])/n
                dense.append(p)
    return sparse,np.array(dense)

def sample(rest,tets,frames,query):
    axes=[np.unique(rest[:,i]) for i in range(3)];shape=tuple(map(len,axes))
    grid=np.array(list(itertools.product(*axes)))
    if grid.shape!=rest.shape or not np.allclose(grid,rest,rtol=0,atol=1e-12):raise ValueError('Expected ordered Cartesian rest grid')
    if any(not np.allclose(np.diff(a),a[1]-a[0],rtol=1e-9,atol=1e-12) for a in axes):raise ValueError('Nonuniform grid')
    lo=rest.min(0);hi=rest.max(0)
    if np.any(query<lo-1e-12) or np.any(query>hi+1e-12):raise ValueError('Query outside rest grid')
    scaled=(query-lo)/(hi-lo)*(np.array(shape)-1);cell=np.minimum(np.floor(np.maximum(scaled,0)).astype(int),np.array(shape)-2)
    frac=np.clip(scaled-cell,0,1);order=np.argsort(-frac,axis=1,kind='stable')
    vertices=np.repeat(cell[:,None,:],4,axis=1)
    for step in range(3):
        vertices[:,step+1]=vertices[:,step]
        vertices[np.arange(len(query)),step+1,order[:,step]]+=1
    ids=np.ravel_multi_index(tuple(vertices.transpose(2,0,1)),shape)
    f=np.take_along_axis(frac,order,axis=1);weights=np.column_stack((1-f[:,0],f[:,0]-f[:,1],f[:,1]-f[:,2],f[:,2]))
    actual={tuple(sorted(t)) for t in tets}
    if any(tuple(sorted(t)) not in actual for t in ids):raise ValueError('Freudenthal tetrahedron absent from saved topology')
    if not np.allclose(np.einsum('qi,qic->qc',weights,rest[ids]),query,rtol=0,atol=1e-11):raise ValueError('Interpolation reconstruction failed')
    return np.einsum('qi,tqic->tqc',weights,frames[:,ids])

def metric(a,b):
    world=float(np.sqrt(np.mean((a[1:]-b[1:])**2)));residual=[]
    for x,y in zip(a[1:],b[1:]):
        x=x-x.mean(0);y=y-y.mean(0);u,_,vt=np.linalg.svd(x.T@y);fix=np.eye(3);fix[-1,-1]=np.linalg.det(u@vt)
        residual.append(np.mean((x@u@fix@vt-y)**2))
    return dict(world_rms_m=world,shape_rms_m=float(np.sqrt(np.mean(residual))))

def inspect(base,obj,scenario,p,minimum_j):
    reasons=[];levels=[];result=dict(qualified=False,reasons=reasons)
    try:
        for directory in ('reference','reference-spatial'):
            path=base/directory/obj/scenario;status=json.loads((path/'complete.json').read_text())
            if not status['success']:raise ValueError(directory+' generation failed')
            with np.load(path/'trajectory.npz') as z:a={k:z[k].copy() for k in z.files}
            if any(not np.isfinite(a[k]).all() for k in ('rest','frames','centers','times','reactions','velocities')):raise ValueError(directory+' nonfinite arrays')
            if len(a['times'])<2 or np.any(np.diff(a['times'])<=0):raise ValueError(directory+' invalid times')
            d=status['diagnostics']
            if not np.isfinite(d['min_j']) or d['min_j']<=.02 or minimum_j(a['frames'],a['rest'],a['tets'])<=.02:reasons.append(directory+' spatial J failed')
            if not np.isfinite(d['max_penetration_m']) or d['max_penetration_m']>p['gates']['penetration_m']:reasons.append(directory+' spatial penetration failed')
            levels.append((a,status))
        (a,sa),(b,sb)=levels
        if a['rest'].shape==b['rest'].shape and np.allclose(a['rest'],b['rest'],rtol=0,atol=1e-12):raise ValueError('Identical meshes cannot qualify spatial refinement')
        if a['times'].shape!=b['times'].shape or not np.allclose(a['times'],b['times'],rtol=0,atol=1e-10):raise ValueError('Spatial time grids differ')
        if not np.isfinite(sa['diagnostics']['dt']) or sa['diagnostics']['dt']<=0 or sa['diagnostics']['dt']!=sb['diagnostics']['dt']:raise ValueError('Spatial integration timesteps differ')
        if not np.allclose([a['rest'].min(0),a['rest'].max(0)],[b['rest'].min(0),b['rest'].max(0)],rtol=0,atol=1e-12):raise ValueError('Rest bounds differ')
        for label,q in zip(('sparse','dense'),probes(a['rest'])):
            score=metric(sample(a['rest'],a['tets'],a['frames'],q),sample(b['rest'],b['tets'],b['frames'],q));score['points']=len(q);result[label]=score
            if not np.isfinite(list(score.values())).all() or score['world_rms_m']>.001 or score['shape_rms_m']>.0005:reasons.append(label+' spatial difference exceeds limit')
        if a['centers'].shape!=b['centers'].shape:raise ValueError('Spatial body shapes differ')
        body=float(np.max(np.sqrt(np.mean((a['centers'][1:]-b['centers'][1:])**2,axis=(0,2)))))
        result['body_center_rms_m']=body
        if not np.isfinite(body) or body>.001:reasons.append('Spatial body difference exceeds limit')
    except (OSError,KeyError,ValueError,np.linalg.LinAlgError) as e:reasons.append('Spatial unavailable/invalid: '+str(e))
    result['qualified']=not reasons
    return result
