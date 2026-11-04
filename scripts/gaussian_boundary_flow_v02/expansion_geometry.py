"""Initial-only geometry preflight and immutable affine packages for eight cases.

Uses only existing stable predictor packages. Higher-resolution PhysTwin meshes
are rebuilt from their persisted full INITIAL support cloud and prescribed
controller trajectories; no raw pickle, future object targets or evaluator reads.
"""
from pathlib import Path
import json
import numpy as np
from scipy.spatial import cKDTree
from scipy.cluster.vq import kmeans2
from scripts.external_benchmark.data import fps
from scripts.gaussian_boundary_flow_v02.affine_observation import affine_observation_matrix
from scripts.gaussian_boundary_flow_v02.export_inputs import load_package,write_package,sha
from scripts.gaussian_boundary_flow_v02.object_backend_geometry import GeometryAwareGaussianBoundaryBackend
from scripts.gaussian_boundary_flow_v02.object_backend_wide import WideGaussianTableBoundaryBackend


def increase_phystwin_resolution(data,count):
    if not isinstance(count,int) or isinstance(count,bool) or count<4:raise ValueError('At least four initial mechanical nodes required')
    if data['dataset']!='PhysTwin':raise ValueError('Resolution reconstruction implemented for PhysTwin only')
    initial=np.asarray(data['support_geometry']);nodes=initial[fps(initial,count)]
    controls=np.asarray(data['controls']);groups=np.zeros(controls.shape[1],int)
    if data['name'].startswith('double') or '_double_' in data['name']:
        _,groups=kmeans2(controls[0],2,minit='++',seed=42)
    weights=[];targets=[]
    for label in np.unique(groups):
        c=controls[:,groups==label];dist,ids=cKDTree(c[0]).query(nodes,k=min(3,len(c[0])))
        if dist.ndim==1:dist,ids=dist[:,None],ids[:,None]
        proximity=1/np.maximum(dist,1e-6)**2;proximity/=proximity.sum(1,keepdims=True)
        weights.append(np.exp(-.5*(dist[:,0]/.015)**2))
        targets.append(nodes[None]+np.einsum('nk,tnkj->tnj',proximity,c[:,ids]-c[0,ids]))
    result=dict(data,nodes=nodes,attachment_weights=np.asarray(weights),controller_targets=np.asarray(targets).transpose(1,0,2,3))
    result['input_geometry_expansion']={'method':'FPS persisted full initial support cloud; reconstruct original controller-proximity rule','nodes':len(nodes),'future_object_access':False}
    return result


def export_and_preflight(source,destination,*,nodes=None):
    source=Path(source);data=load_package(source)
    if nodes is not None:data=increase_phystwin_resolution(data,nodes)
    matrix,diagnostic=affine_observation_matrix(data['nodes'],data['observed_initial'])
    data['observation_matrix']=matrix
    data['observation_map_revision']={'source_package':str(source.resolve()),'source_manifest_sha256':sha(source/'manifest.json'),'rule':'Initial-geometry affine-complete local map; no motion or future targets used','maximum_row_l1':diagnostic['maximum_row_l1'],'maximum_row_l2':diagnostic['maximum_row_l2'],'negative_weights_allowed':True}
    manifest=write_package(data,destination)
    geometry='filtered'if data['dataset']=='PhysTwin'else'convex'
    row={'case':data['name'],'package':str(destination),'nodes':len(data['nodes']),'geometry':geometry,'manifest_sha256':sha(Path(destination)/'manifest.json'),'maximum_row_l1':diagnostic['maximum_row_l1'],'maximum_affine_error_m':diagnostic['maximum_geometry_error'],'future_target_access':False}
    try:
        backend=(GeometryAwareGaussianBoundaryBackend if geometry=='filtered'else WideGaussianTableBoundaryBackend)(data,'cpu')
        m=backend.model
        if geometry=='convex':
            from scipy.sparse import coo_matrix
            from scipy.sparse.csgraph import connected_components
            tet=m.elements.numpy();edges=np.concatenate([tet[:,[0,j]]for j in(1,2,3)])
            components=int(connected_components(coo_matrix((np.ones(len(edges)),(edges[:,0],edges[:,1])),shape=(m.n,m.n)),directed=False)[0])
        else:components=backend.geometry_diagnostic['connected_components']
        row.update(eligible=components==1 and bool((m.mass>0).all()),elements=len(m.elements),connected_components=components,minimum_mass_fraction=float(m.mass.min()),volume_m3=float(m.vol.sum())*m.scale**3,mesh_filter=m.mesh_filter)
    except ValueError as error:row.update(eligible=False,error=str(error))
    return row


def main():
    base=Path('runs/gaussian-boundary-flow-v02-inputs');out=Path('runs/gaussian-boundary-flow-v02-expansion-affine-inputs');rows=[]
    for source in sorted(base.iterdir()):
        if not(source/'manifest.json').exists():continue
        row=export_and_preflight(source,out/source.name);rows.append(row);print(row,flush=True)
        if source.name=='double_lift_zebra' and not row['eligible']:
            for count in (120,192,256):
                row=export_and_preflight(source,out/f'{source.name}-nodes{count}',nodes=count);rows.append(row);print(row,flush=True)
                if row['eligible']:break
    report={'scope':'Eight historically exposed stable predictor packages; initial geometry and prescribed controller schedules only','rows':rows,'selection':'First connected positive-mass filtered resolution for PhysTwin; explicit convex64 for SoftVT','physics_qualified':False,'limitations':'Geometry construction admission only; no forward/mesh convergence, material inference or predictive test'}
    Path('reports/gaussian-boundary-flow-v02/expansion-geometry.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
