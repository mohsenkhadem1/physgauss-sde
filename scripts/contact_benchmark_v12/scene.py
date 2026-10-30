"""Initial-pose public-asset static colliders; no future states or force labels.

Heightfield is an explicitly approximate vertical-contact representation of the
actual mesh, not a signed-distance collider or coupled rigid-body solver.
"""
from pathlib import Path
import hashlib
import json
import urllib.request
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / 'runs/contact-benchmark-v12/scene-assets'
SOURCE = 'https://huggingface.co/datasets/NathanWu7/Isaaclab_Libero'
SUPPORTED = ('plate', 'basket', 'akita_black_bowl', 'glazed_rim_porcelain_ramekin')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def asset_mesh(name, download=False):
    """Return referenced mesh coordinates, preserving Isaac reference units."""
    from pxr import Usd, UsdGeom
    path = ASSETS / (name + '.usd')
    url = f'{SOURCE}/resolve/main/USD/{name}/{name}.usd'
    if not path.exists():
        if not download: raise FileNotFoundError(f'{path}; fetch {url}')
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    stage = Usd.Stage.Open(str(path)); cache = UsdGeom.XformCache()
    vertices, triangles, count = [], [], 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh): continue
        mesh = UsdGeom.Mesh(prim); p = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
        matrix = np.asarray(cache.GetLocalToWorldTransform(prim))
        p = (np.c_[p, np.ones(len(p))] @ matrix)[:, :3]
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()); start = 0
        for size in mesh.GetFaceVertexCountsAttr().Get():
            face = indices[start:start+size]; start += size
            for j in range(1, size-1): triangles.append([count+face[0], count+face[j], count+face[j+1]])
        vertices.append(p); count += len(p)
    return np.concatenate(vertices), np.asarray(triangles, dtype=int), dict(
        asset=name, source_url=url, local_path=str(path.relative_to(ROOT)), sha256=sha(path),
        stage_meters_per_unit=float(UsdGeom.GetStageMetersPerUnit(stage)),
        units_policy='Numeric USD reference coordinates, same as Isaac runtime; no metadata unit conversion',
        license_notice='Upstream asset terms apply; raw USD assets remain local; see upstream dataset card')


class StaticScene:
    def __init__(self, meshes=(), floor=0., spacing=.002):
        self.meshes=list(meshes); self.floor=float(floor); self.spacing=float(spacing)
        self.fields=[]
        for obj in self.meshes:
            v=np.asarray(obj['vertices']); f=np.asarray(obj['triangles']); low=v.min(0); high=v.max(0)
            origin=np.floor(low[:2]/spacing)*spacing
            shape=np.ceil((high[:2]-origin)/spacing).astype(int)+1
            h=np.full(tuple(shape)+(32,),-np.inf); normals=np.zeros(tuple(shape)+(32,3))
            for tri in v[f]:
                a,b,c=tri; den=(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
                if abs(den)<1e-14: continue
                lo=np.maximum(np.ceil((tri[:,:2].min(0)-origin)/spacing).astype(int),0)
                hi=np.minimum(np.floor((tri[:,:2].max(0)-origin)/spacing).astype(int),shape-1)
                if np.any(hi<lo): continue
                ii,jj=np.meshgrid(np.arange(lo[0],hi[0]+1),np.arange(lo[1],hi[1]+1),indexing='ij')
                x=origin[0]+ii*spacing-a[0]; y=origin[1]+jj*spacing-a[1]
                s=(x*(c[1]-a[1])-y*(c[0]-a[0]))/den
                t=((b[0]-a[0])*y-(b[1]-a[1])*x)/den
                z=a[2]+s*(b[2]-a[2])+t*(c[2]-a[2])
                valid=(s>=-1e-8)&(t>=-1e-8)&(s+t<=1+1e-8)
                normal=np.cross(b-a,c-a); normal/=np.linalg.norm(normal)
                if normal[2]<0: normal=-normal
                if normal[2] < .05: continue
                i,j,zz=ii[valid],jj[valid],z[valid]
                duplicate=np.any(np.abs(h[i,j]-zz[:,None])<1e-5,axis=1)
                i,j,zz=i[~duplicate],j[~duplicate],zz[~duplicate]
                layer=np.argmax(~np.isfinite(h[i,j]),axis=1)
                if np.any(np.all(np.isfinite(h[i,j]),axis=1)): raise ValueError('Too many vertical surface layers')
                h[i,j,layer]=zz; normals[i,j,layer]=normal
            self.fields.append(dict(origin=origin,height=h,normal=normals,bounds=[low.tolist(),high.tolist()]))

    def query_surface(self, points, max_height=None):
        """Vertical gap z-height, upward mesh normal, index (-1=fallback floor).

        Nearest 2mm raster cell; no extrapolation beyond each mesh footprint.
        With max_height, selects the highest intersection below that bound.
        Pass previous_z + tolerance for swept falling nodes to avoid jumping
        upward onto handles. Vertical walls and general undercuts are
        NOT represented physically. This API is for top-down landing only.
        """
        points=np.asarray(points,dtype=float); shape=points.shape[:-1]; p=points.reshape(-1,3)
        limit=np.full(len(p),np.inf) if max_height is None else np.broadcast_to(max_height,shape).reshape(-1)
        # The infinite floor remains a unilateral constraint even when an
        # inherited state penetrates it. Height bounds only select mesh layers.
        height=np.full(len(p),self.floor); normal=np.tile([0.,0.,1.],(len(p),1)); index=np.full(len(p),-1,int)
        for k,field in enumerate(self.fields):
            ij=np.rint((p[:,:2]-field['origin'])/self.spacing).astype(int); dims=field['height'].shape
            valid=(ij[:,0]>=0)&(ij[:,1]>=0)&(ij[:,0]<dims[0])&(ij[:,1]<dims[1]); ids=np.flatnonzero(valid)
            layers=field['height'][ij[ids,0],ij[ids,1]]
            eligible=np.where(layers<=limit[ids,None],layers,-np.inf)
            layer=np.argmax(eligible,axis=1); z=eligible[np.arange(len(ids)),layer]
            keep=z>height[ids]; ids=ids[keep];layer=layer[keep]
            height[ids]=field['height'][ij[ids,0],ij[ids,1],layer]; normal[ids]=field['normal'][ij[ids,0],ij[ids,1],layer];index[ids]=k
        return dict(gap=(p[:,2]-height).reshape(shape),height=height.reshape(shape),normal=normal.reshape(shape+(3,)),object_index=index.reshape(shape))

    def metadata(self):
        return dict(model='initial static mesh layered heightfields',spacing_m=self.spacing,floor_m=self.floor,
                    floor_provenance='initial object min-z fallback from v11; not measured table surface',
                    limitations=['No undercuts','No vertical side-wall collision','Static colliders; no two-way dynamics','Nearest raster query'],
                    objects=[dict(name=o['name'],pose=o['pose'],scale=o['scale'],bounds=f['bounds'],provenance=o['provenance']) for o,f in zip(self.meshes,self.fields)])


def load_scene(data, spacing=.002, download=False):
    """Read ONLY initial rigid poses and public config geometry for SoftVT."""
    if data['dataset']!='SoftVTBench': return StaticScene(floor=data['floor'],spacing=spacing)
    import h5py
    family='spatial' if data['name'].startswith('spatial') else 'object'
    config=ROOT/f'runs/third_party/SoftVTBench/configs/{family}_soft/libero_{family}.json'
    task=int(data['name'].split('-task')[1].split('-')[0]); episode=int(data['name'].split('-demo')[1])
    spec=next(t for t in json.loads(config.read_text())['tasks'] if int(t['task_id'])==task)
    meshes=[]
    with h5py.File(data['source']) as h:
        initial=h[f'data/demo_{episode}/initial_state/rigid_object']
        for name in initial:
            typ=name.rsplit('_',1)[0]
            if typ not in SUPPORTED: continue
            pose=initial[name+'/root_pose'][0].astype(float)
            if name not in spec['objects']: raise ValueError(f'No public object config for {name}')
            scale=np.asarray(spec['objects'][name]['scale'],dtype=float)
            v,f,provenance=asset_mesh(typ,download)
            rotation=Rotation.from_quat(pose[[4,5,6,3]]).as_matrix()
            world=(v*scale)@rotation.T+pose[:3]
            provenance.update(config_path=str(config.relative_to(ROOT)),config_sha256=sha(config),pose_source=f'initial_state/rigid_object/{name}/root_pose')
            meshes.append(dict(name=name,vertices=world,triangles=f,pose=pose.tolist(),scale=scale.tolist(),provenance=provenance))
    return StaticScene(meshes,floor=data['floor'],spacing=spacing)
