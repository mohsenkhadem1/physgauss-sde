"""Initial-only isolated sampled-vertex repair; existing FEM backend unchanged."""
import argparse,copy,json
from pathlib import Path
import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scripts.gaussian_boundary_flow_v02.export_inputs import load_package,write_package,sha
from scripts.gaussian_boundary_flow_v02.benchmark_inputs import predictor_digest,validate_predictor
from scripts.gaussian_boundary_flow_v02.affine_observation import affine_observation_matrix
from scripts.gaussian_boundary_flow_v02.object_backend_geometry import GeometryAwareGaussianBoundaryBackend
from scripts.external_benchmark.fem import ExternalFEM


def repair(data,elements,mass):
    elements=np.asarray(elements,int);mass=np.asarray(mass);n=len(data['nodes'])
    if mass.shape!=(n,)or not np.isfinite(mass).all()or np.any(mass<0):raise ValueError('Invalid mass vector')
    used=np.unique(elements);isolated=np.setdiff1d(np.arange(n),used);zero=np.flatnonzero(mass==0)
    if not len(zero)or not np.array_equal(isolated,zero):raise ValueError('Only exactly isolated zero-mass vertices may be removed')
    keep=np.flatnonzero(mass>0);remap=np.full(n,-1,int);remap[keep]=np.arange(len(keep));tet=remap[elements]
    edges=np.concatenate([tet[:,[0,j]]for j in (1,2,3)]);graph=coo_matrix((np.ones(len(edges)),(edges[:,0],edges[:,1])),shape=(len(keep),len(keep)))
    count,_=connected_components(graph,directed=False)
    if count!=1:raise ValueError('Positive-volume components are disconnected; cannot delete any component')
    result=copy.deepcopy(data);result['nodes']=data['nodes'][keep].copy();result['controller_targets']=data['controller_targets'][:,:,keep].copy();result['attachment_weights']=data['attachment_weights'][:,keep].copy()
    result['observation_matrix'],diag=affine_observation_matrix(result['nodes'],data['observed_initial'])
    result['isolated_geometry_repair']={'rule':'Remove only isolated zero-mass sampled vertices, retain every positive-volume component and every observation','source_predictor_digest':predictor_digest(data),'retained_node_ids':keep.tolist(),'removed_node_ids':zero.tolist(),'original_nodes':n,'retained_nodes':len(keep)}
    result['observation_map_revision']={'rule':'Initial-only affine-complete local map after isolated-vertex repair','maximum_row_l1':diag['maximum_row_l1'],'maximum_row_l2':diag['maximum_row_l2'],'negative_weights_allowed':True}
    for key in ('support_geometry','observed_initial','observed','observed_mask','times','controls','attachment_active'):
        if key in data and not np.array_equal(data[key],result[key],equal_nan=True):raise ValueError('Changed observation or support identity: '+key)
    return validate_predictor(result),tet,diag


def canonical(elements):return {tuple(sorted(x))for x in np.asarray(elements).tolist()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);args=p.parse_args();planpath=Path(args.plan);plan=json.loads(planpath.read_text());torch.set_num_threads(2)
    for source,digest in plan['source_sha256'].items():
        if sha(source)!=digest:raise ValueError('Changed repair source')
    rows=[]
    for spec in plan['cases']:
        data=load_package(spec['source']);row={'case':data['name'],'source':spec['source'],'source_manifest_sha256':sha(Path(spec['source'])/'manifest.json'),'eligible':False}
        if row['source_manifest_sha256']!=spec['manifest_sha256']:raise ValueError('Changed input package')
        try:
            old=ExternalFEM(data,device='cpu',field=False,contact=True,nonconvex=True,materials=1)
            repaired,tets,diag=repair(data,old.elements.numpy(),old.mass.numpy())
            removed=repaired['isolated_geometry_repair']['removed_node_ids']
            if removed!=spec['expected_isolated_ids']:raise ValueError('Isolated IDs differ from frozen repair plan')
            new=GeometryAwareGaussianBoundaryBackend(repaired,'cpu')
            if canonical(tets)!=canonical(new.model.elements.numpy()):raise ValueError('Unchanged backend remeshing changed positive-volume tetrahedra; explicit topology support would need a separate backend revision')
            output=Path(spec['output']);write_package(repaired,output);loaded=load_package(output)
            if predictor_digest(loaded)!=predictor_digest(repaired):raise ValueError('Stable reload identity mismatch')
            row.update(eligible=True,package=str(output),nodes=len(repaired['nodes']),geometry='filtered',removed_node_ids=removed,
                       selected_manifest_sha256=sha(output/'manifest.json'),selected_predictor_digest=predictor_digest(loaded),diagnostic=new.geometry_diagnostic,
                       positive_volume_tetrahedra_exactly_preserved=True,old_volume_m3=float(old.vol.sum())*old.scale**3,new_volume_m3=float(new.model.vol.sum())*new.model.scale**3,
                       affine_diagnostic={k:v for k,v in diag.items()if k!='rows'})
        except (ValueError,RuntimeError)as error:row['error']=str(error)
        rows.append(row);print(row['case'],row['eligible'],row.get('error',''),flush=True)
    Path(plan['result']).write_text(json.dumps({'plan_sha256':sha(planpath),'rows':rows,'target_access':False,'device':'cpu','backend_changed':False},indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
