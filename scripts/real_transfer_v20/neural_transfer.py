"""Frozen source FEM twin in a new interaction, with no target trajectory fit."""
import argparse,json,numpy as np,torch
from scipy.spatial import cKDTree
from scipy.cluster.vq import kmeans2
from .registration import initial_bundle
from .development_run import ROOT,BASE,write,sha
from scripts.real_transfer_v19.phystwin_run import load
from scripts.real_transfer_v19.neural_fem import NeuralFEM
from scripts.external_benchmark.fem import ExternalFEM
from scripts.external_benchmark.data import interpolate_matrix

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--target',required=True)
    p.add_argument('--method',choices=['neural','analytic'],default='neural');p.add_argument('--device',default='cuda:1')
    p.add_argument('--fit-root',default='runs/real-transfer-v19/fits/exploratory-01');p.add_argument('--label',default='original-frozen')
    args=p.parse_args();torch.set_num_threads(1)
    folder=ROOT/args.fit_root/args.source/args.method;plan=json.loads((folder/'plan.json').read_text())
    sealed=json.loads((folder/'complete.json').read_text())
    if sha(folder/'fit.json')!=sealed['fit_sha256']:raise ValueError('Source fit changed after sealing')
    n=plan['args']['nodes'];width=plan['args'].get('attachment_width_mm',15.)
    data,_=load(args.source,n,False)
    if width!=15.:data['attachment_weights']=data['attachment_weights']**((15./width)**2)
    m=NeuralFEM(data,args.device) if args.method=='neural' else ExternalFEM(data,args.device,field=False,nonconvex=True,materials=6)
    with np.load(folder/'best.npz') as f:old=f['theta']
    fit=json.loads((folder/'fit.json').read_text())
    if not np.array_equal(old,np.array(fit['best']['parameters'])):raise ValueError('Source checkpoint differs from sealed fit')
    source_floor_index=m.i_floor;source_material_index=m.i_material;source_field_index=m.i_field
    report=json.loads((ROOT/'reports/real-transfer-v20/initial-registration.json').read_text())
    row=next(r for r in report['pairs'] if r['source']==args.source and r['target']==args.target)
    R=np.array(row['rotation']);shift=np.array(row['translation_m'])
    _,controls,target_hash=initial_bundle(args.target)
    original_nodes=data['nodes'];new_nodes=original_nodes@R+shift
    origin=m.origin@R+shift
    groups=np.zeros(controls.shape[1],int)
    if args.target.startswith('double'):_,groups=kmeans2(controls[0],2,minit='++',seed=42)
    targets=[];weights=[];distances=[]
    for group in np.unique(groups):
        c=controls[:,groups==group];d,ids=cKDTree(c[0]).query(new_nodes,k=min(3,len(c[0])))
        w=np.exp(-.5*(d[:,0]/(width/1000))**2)
        proximity=1/np.maximum(d,1e-6)**2;proximity/=proximity.sum(1,keepdims=True)
        targets.append(new_nodes[None]+np.einsum('nk,tnkj->tnj',proximity,c[:,ids]-c[0,ids]))
        weights.append(w);distances.append(float(d[:,0].min()*1000))
    m.rest=m.rest@m.t(R);m.grad=m.grad@m.t(R)
    m.origin=origin;m.nodes=new_nodes
    m.weights=m.t(np.asarray(weights));m.weights/=m.weights.sum(1,keepdim=True).clamp_min(1e-9)
    m.targets=m.t((np.asarray(targets).transpose(1,0,2,3)-origin)/m.scale)
    m.active=m.t(np.ones((len(controls),len(weights))))
    floor=-10*m.scale if plan['args'].get('floor_mode')=='disabled' else 0.
    m.floor=(floor-origin[2])/m.scale
    m.i_floor=1+len(weights);m.i_material=m.i_floor+1;m.i_field=m.i_material+m.n_material
    if args.method=='neural':m.extra=m.i_field+1
    theta=np.r_[old[0],np.repeat(np.median(old[1:source_floor_index]),len(weights)),old[source_floor_index],
                old[source_material_index:source_field_index],old[source_field_index:]]
    theta=m.t(theta);prediction=[];diagnostics=[];x=None
    for frame in range(len(controls)):
        try:
            x,_,diagnostic=m.solve(theta,frame,x);points=x.detach().cpu().numpy()*m.scale+origin
        except (ValueError,RuntimeError) as error:
            points=np.full_like(new_nodes,np.nan);diagnostic=dict(failed=True,error=str(error))
        prediction.append(points);diagnostics.append(diagnostic)
        if frame%20==0:print('frame',frame,flush=True)
    prediction=np.array(prediction);A=interpolate_matrix(original_nodes,data['support_geometry'])
    initial=data['support_geometry']@R+shift
    dense=initial[None]+np.einsum('pn,tnc->tpc',A,prediction-new_nodes)
    out=BASE/'frozen-transfer'/args.label/f'{args.source}--{args.target}'/args.method
    out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/'prediction.npz',predicted=dense,predicted_nodes=prediction,nodes=new_nodes,theta=theta.detach().cpu().numpy())
    write(out/'complete.json',dict(source=args.source,target=args.target,method=args.method,registration=row,
        source_fit_sha256=sha(folder/'fit.json'),target_input_sha256=target_hash,
        prediction_sha256=sha(out/'prediction.npz'),diagnostics=diagnostics,
        closest_tool_to_node_mm=distances,target_post_initial_geometry_used=False,
        boundary_policy='New geometric attachments; median source attachment strength; no target fitting.',
        scope='Rigidly registered source shape and frozen material field; invalid registration is reported separately.'))
    print('complete',args.source,args.target,args.method,flush=True)

if __name__=='__main__':main()
