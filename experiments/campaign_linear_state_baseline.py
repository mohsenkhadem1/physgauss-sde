"""Fixed fit-only PCA/AR2-ridge forward baseline for four seen public hardware H.

No physics calls, future measured states, held-target fitting or unknown-H
nearest lookup. A residual-acceleration AR2 prior rolls from known rest.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
H_KEYS=('rest','handle_patches','handle_centers','handle_masses','handle_inertias','handle_com_offsets','handle_initial_rotations')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def hardware_fingerprint(data,index):
    digest=hashlib.sha256()
    for key in (*H_KEYS,'tets','faces'):
        a=np.ascontiguousarray(data[key] if key in ['tets','faces'] else data[key][index])
        digest.update(key.encode());digest.update(str(a.dtype).encode());digest.update(str(a.shape).encode());digest.update(a.tobytes())
    return digest.hexdigest()


def features(z1,z2,current,previous,support_current,support_previous):
    # Fixed units; no target-amplitude normalization or target-derived time scale.
    return np.concatenate([z1,z1-z2,current/.25,previous/.25,support_current/.01,support_previous/.01])


def lift(data,index):return data['support_offsets'][index,:,None,:]*data['fixture_mask'][index,None,:,None]


def fit_models(data,*,maximum_modes=8,ridge=1e-3):
    fit=np.flatnonzero(data['train_mask']);groups={}
    for i in fit:groups.setdefault(hardware_fingerprint(data,int(i)),[]).append(int(i))
    models={}
    for fingerprint,indices in sorted(groups.items()):
        # Read target q ONLY at fitting indices. Uncentered PCA preserves zero-rest.
        snapshots=np.concatenate([((data['q'][i]-data['rest'][i]-lift(data,i))/.24).reshape(61,-1) for i in indices])
        _,singular,basis=np.linalg.svd(snapshots,full_matrices=False)
        rank=min(maximum_modes,int(np.sum(singular>1e-10*max(float(singular[0]),1e-30))))
        modes=basis[:rank];X=[];Y=[]
        for i in indices:
            z=((data['q'][i]-data['rest'][i]-lift(data,i))/.24).reshape(61,-1)@modes.T
            for t in range(1,61):
                z1=z[t-1];z2=z[t-2] if t>=2 else np.zeros(rank)
                X.append(features(z1,z2,data['controls'][i,t],data['controls'][i,t-1],data['support_offsets'][i,t],data['support_offsets'][i,t-1]))
                Y.append(z[t]-2*z1+z2)
        X=np.array(X);Y=np.array(Y)
        W=np.linalg.solve(X.T@X+ridge*np.eye(X.shape[1]),X.T@Y)
        # Report, never retrospectively stabilize/clip learned dynamics.
        Wz=W[:rank].T;Wd=W[rank:2*rank].T
        transition=np.block([[2*np.eye(rank)+Wz+Wd,-np.eye(rank)-Wd],[np.eye(rank),np.zeros((rank,rank))]])
        controls=np.concatenate([data['controls'][i,1:,:6] for i in indices])
        _,cs,cv=np.linalg.svd(controls,full_matrices=False);crank=int(np.sum(cs>1e-8*max(float(cs[0]),1e-30)))
        models[fingerprint]={'basis':modes,'W':W,'fit_indices':indices,'rank':rank,'PCA_singular_values':singular[:maximum_modes].tolist(),
            'feature_matrix_rank':int(np.linalg.matrix_rank(X,tol=1e-10)),'feature_columns':int(X.shape[1]),
            'transition_spectral_radius':float(np.max(np.abs(np.linalg.eigvals(transition)))) if rank else 0.,
            'force_control_rank':crank,'force_control_basis':cv[:crank],
            'fit_handle1_force_excited':bool(np.any(controls[:,3:]!=0))}
    return models


def rollout(data,index,models):
    fingerprint=hardware_fingerprint(data,index)
    if fingerprint not in models:raise ValueError('Unknown complete public hardware H; no nearest-context fallback')
    model=models[fingerprint];basis=model['basis'];W=model['W'];rank=model['rank']
    z=np.zeros((61,rank));prediction=np.empty((61,*data['rest'][index].shape));known_lift=lift(data,index)
    prediction[0]=data['rest'][index]
    for t in range(1,61):
        z1=z[t-1];z2=z[t-2] if t>=2 else np.zeros(rank)
        x=features(z1,z2,data['controls'][index,t],data['controls'][index,t-1],data['support_offsets'][index,t],data['support_offsets'][index,t-1])
        z[t]=2*z1-z2+x@W
        prediction[t]=data['rest'][index]+known_lift[t]+.24*(z[t]@basis).reshape(-1,3)
        mask=data['fixture_mask'][index]
        prediction[t,mask]=data['rest'][index,mask]+data['support_offsets'][index,t]
    if not np.isfinite(prediction).all():raise FloatingPointError('Nonfinite fixed-budget baseline rollout; no stabilization retry')
    return prediction


def evaluate(data,prediction,models):
    rows=[];times=data['times'];tets=data['tets']
    rmse=lambda x:float(np.sqrt(np.mean(x*x)))
    for i in range(len(prediction)):
        model=models[hardware_fingerprint(data,i)];error=prediction[i]-data['q'][i];rest=data['rest'][i]
        source=np.swapaxes(rest[tets[:,1:]]-rest[tets[:,:1]],-1,-2)
        # Avoid advanced-index axis surprises: direct frame-indexed tetra vertices.
        vertices=prediction[i][:,tets];edges=np.swapaxes(vertices[:,:,1:]-vertices[:,:,:1],-1,-2)
        J=np.linalg.det(edges@np.linalg.inv(source)[None])
        controls=data['controls'][i,1:,:6];basis=model['force_control_basis']
        unexplained=controls-controls@basis.T@basis
        rows.append({'id':str(data['ids'][i]),'fit_pair':bool(data['train_mask'][i]),'position_rmse_m':rmse(error[times>0]),
            'future_after_1s_rmse_m':rmse(error[times>1]),'static_rmse_m':rmse((data['q'][i]-rest)[times>0]),
            'predicted_rms_motion_m':rmse((prediction[i]-rest)[times>0]),'actual_rms_motion_m':rmse((data['q'][i]-rest)[times>0]),
            'minimum_J':float(J.min()),'inverted_tetrahedron_frames':int((J<=0).sum()),
            'unexcited_known_force_fraction':float(np.linalg.norm(unexplained)/max(np.linalg.norm(controls),1e-30)),
            'fit_handle1_force_excited':model['fit_handle1_force_excited'],'held_handle1_force_nonzero':bool(np.any(controls[:,3:]!=0)),
            'force_control_rank':model['force_control_rank'],'input_measured_motion_frames':0,'input_physics':'seen-H fit dynamics; known support hypothesis/actuator history',
            'complete_public_H_fingerprint':hardware_fingerprint(data,i)})
    return rows


def run(config_path):
    config= json.loads(Path(config_path).read_text());source=Path(__file__)
    if config['source_sha256']!=sha(source) or config['maximum_modes']!=8 or config['ridge']!=1e-3:raise ValueError('Frozen one-budget baseline configuration required')
    dataset=ROOT/config['dataset'];output=ROOT/config['output']
    if output.exists():raise FileExistsError('Fresh baseline output required')
    if sha(dataset)!=config['dataset_sha256']:raise ValueError('Dataset changed')
    with np.load(dataset,allow_pickle=False) as raw:data={key:raw[key].copy() for key in raw.files}
    if int(data['train_mask'].sum())!=12 or len(data['rest'])!=16:raise ValueError('Exact fixed12/4 pilot required')
    output.mkdir(parents=True,exist_ok=False);write(output/'protocol.json',config);started=time.monotonic()
    models=fit_models(data,maximum_modes=8,ridge=1e-3)
    prediction=np.array([rollout(data,i,models) for i in range(16)])
    # First and only held-target metric evaluation follows the frozen fit/rollout.
    rows=evaluate(data,prediction,models)
    for mask,name in [(data['train_mask'],'fitting'),(~data['train_mask'],'held_cross_action')]:
        selected=[row for row,on in zip(rows,mask) if on]
        write(output/(name+'-metrics.json'),{'pairs':selected,'mean_rmse_m':float(np.mean([r['position_rmse_m'] for r in selected])),
            'mean_static_rmse_m':float(np.mean([r['static_rmse_m'] for r in selected])),
            'mean_future_after_1s_rmse_m':float(np.mean([r['future_after_1s_rmse_m'] for r in selected])),
            'sum_inverted_tetrahedron_frames':sum(r['inverted_tetrahedron_frames'] for r in selected),
            'fresh_layout_generalization_established':False,'production_dataset_admitted':False,'held_tuning_performed':False})
    np.savez_compressed(output/'predictions.npz',predicted_positions_m=prediction,times_s=data['times'],ids=data['ids'],train_mask=data['train_mask'])
    raw={}
    for index,(key,model) in enumerate(models.items()):raw['basis_'+str(index)]=model['basis'];raw['W_'+str(index)]=model['W']
    np.savez_compressed(output/'linear-models.npz',**raw)
    descriptions={key:{k:v for k,v in model.items() if k not in ['basis','W','force_control_basis']} for key,model in models.items()}
    write(output/'models.json',descriptions)
    result={'completed':True,'elapsed_s':time.monotonic()-started,'predictions_sha256':sha(output/'predictions.npz'),
        'config_sha256':sha(config_path),'source_sha256':sha(source),'four_seen_H_only':True,'held_tuning_performed':False,'physics_calls':0,'GPU_used':False}
    write(output/'completed.json',result);print(json.dumps(result));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);a=p.parse_args();run(a.config)
