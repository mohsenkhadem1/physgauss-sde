"""Fixed TRAIN-coverage and response-interpolation diagnosis on selection8 only.

Known physical parameters are supplied for this FORWARD diagnosis, not inferred.
Only original48 and expanded128 TRAIN candidate controls fit interpolation.
Selection predictions are durably saved before any selection trajectory opens.
No gate, calibration, zero-force, independent diagonal or TEST labels are used.
"""
from __future__ import annotations
import os
os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT', '1')
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

RIDGE = 1e-3
METHODS = ('nearest_raw', 'rbf_raw', 'nearest_fe', 'rbf_fe')
DESCRIPTOR = ('logE','logrho','nu','logdrag',
              'support0_x','support0_y','support0_z','support1_x','support1_y','support1_z',
              'public_handle0_x','public_handle0_y','public_handle0_z',
              'public_handle1_x','public_handle1_y','public_handle1_z',
              'moving_x','moving_y','moving_z')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def layout_id(record):
    return record['episode_id'].split('/')[0]


def stratum(record):
    action=record['episode_id'].split('/')[-1]
    if action not in ('poke_0','poke_1','poke_2','poke_3'):
        raise ValueError('Only four declared candidate controls are allowed')
    expected_family=2 if action=='poke_3' else 0
    if int(record['force'])!=expected_family:
        raise ValueError('Action and temporal family disagree')
    return int(record['bc']),action


def descriptor(record):
    p=np.asarray(record['physical'],dtype=float)
    if p.shape!=(25,) or not np.isfinite(p).all():raise ValueError('Expected finite known physical25')
    # Control loaders overwrite the load centers with PUBLIC handle centers.
    return p[np.r_[0:16,22:25]]


def force_over_E(record):
    p=np.asarray(record['physical'],dtype=float)
    amplitude=np.linalg.norm(p[16:19])
    if amplitude<=0:raise ValueError('Positive candidate force amplitude required')
    return amplitude/np.exp(p[0])


class ResponseInterpolator:
    """Small fixed-stratum interpolation with a TRAIN-only RBF mean fallback.

    Standardization, bandwidth and response mean use only fitting records.
    The RBF bandwidth is the median positive TRAIN pair distance; lambda=1e-3.
    Far-away kernels tend to zero, leaving the TRAIN response mean. There is
    no outcome-based tuning, extrapolation clipping, or access to query response
    labels. The nearest baseline deliberately reuses one TRAIN response.
    """
    def __init__(self, method):
        if method not in METHODS:raise ValueError('Unknown fixed interpolation method')
        self.method=method

    def fit(self, records):
        if not records or any(r['split']!='train' for r in records):
            raise ValueError('Only nonempty TRAIN records may fit interpolation')
        self.groups={}
        for key in sorted(set(stratum(r) for r in records)):
            rows=[r for r in records if stratum(r)==key]
            x=np.stack([descriptor(r) for r in rows]);center=x.mean(0);scale=x.std(0)
            scale=np.where(scale>1e-10,scale,1.)
            x=(x-center)/scale
            y=np.stack([np.asarray(r['q'],dtype=float)-r['rest'] for r in rows])
            if self.method.endswith('_fe') and key[0]!=3:
                y=y/np.array([force_over_E(r) for r in rows])[:,None,None,None]
            shape=y.shape[1:];y=y.reshape(len(rows),-1)
            distances=np.sqrt(np.maximum(np.square(x[:,None]-x[None]).mean(-1),0))
            positive=distances[np.triu_indices(len(rows),1)];positive=positive[positive>1e-12]
            length=float(np.median(positive)) if len(positive) else 1.
            kernel=np.exp(-.5*np.square(distances/length));mean=y.mean(0)
            coef=None
            if self.method.startswith('rbf'):
                coef=np.linalg.solve(kernel+RIDGE*np.eye(len(rows)),y-mean)
            self.groups[key]={'x':x,'center':center,'scale':scale,'length':length,
                'y':y,'mean':mean,'coef':coef,'shape':shape,'records':rows}
        return self

    def predict(self, record):
        key=stratum(record)
        if key not in self.groups:raise ValueError('No TRAIN support for requested BC/action stratum')
        g=self.groups[key];x=(descriptor(record)-g['center'])/g['scale']
        d=np.sqrt(np.square(g['x']-x).mean(-1))
        if self.method.startswith('nearest'):y=g['y'][int(np.argmin(d))]
        else:y=g['mean']+np.exp(-.5*np.square(d/g['length']))@g['coef']
        displacement=y.reshape(g['shape'])
        if self.method.endswith('_fe') and key[0]!=3:
            displacement=displacement*force_over_E(record)
        return np.asarray(record['rest'])+displacement

    def coverage(self, record):
        g=self.groups[stratum(record)];raw=descriptor(record)
        x=(raw-g['center'])/g['scale'];d=np.sqrt(np.square(g['x']-x).mean(-1))
        nearest=int(np.argmin(d));reference=np.asarray(record['rest'])
        lo,hi=reference.min(0),reference.max(0)
        def faces(r):
            centers=np.asarray(r['physical'])[4:10].reshape(2,3)
            distances=np.concatenate([np.abs(centers-lo),np.abs(centers-hi)],axis=1)
            return tuple(np.argmin(distances,axis=1).tolist())
        support_faces=faces(record)
        count=2 if int(record['bc'])==2 else 1
        return {'train_stratum_count':len(g['records']),
            'nearest_train_episode_id':g['records'][nearest]['episode_id'],
            'nearest_standardized_rms_distance':float(d[nearest]),
            'nearest_over_train_median_pair_distance':float(d[nearest]/g['length']),
            'outside_train_coordinate_range':((raw<np.min(np.stack([descriptor(r) for r in g['records']]),axis=0)-1e-9)|
                                             (raw>np.max(np.stack([descriptor(r) for r in g['records']]),axis=0)+1e-9)).tolist(),
            'support_face_signature':list(support_faces[:count]),
            'support_face_signature_seen':any(faces(r)[:count]==support_faces[:count] for r in g['records']),
            'rbf_bandwidth_standardized_rms_distance':g['length']}


def leave_one_layout_out(records, method):
    """All preprocessing refits after holding out every action of one layout."""
    predictions=np.empty_like(np.stack([r['q'] for r in records]),dtype=np.float32)
    for layout in sorted(set(layout_id(r) for r in records)):
        training=[r for r in records if layout_id(r)!=layout]
        model=ResponseInterpolator(method).fit(training)
        for i,r in enumerate(records):
            if layout_id(r)==layout:predictions[i]=model.predict(r)
    return predictions


def load_selection_metadata(config, episode_ids):
    """Known forward physics and public menus only; NO response trajectory IO."""
    from campaign_control_data import require_role
    from gaussian_operator import physical_vector, category_labels
    from diagnose_gaussian_forward import control_physical
    roles=[require_role(config,eid,'select') for eid in episode_ids]
    if len(set(episode_ids))!=len(episode_ids) or not episode_ids:
        raise ValueError('Distinct nonempty selection layouts required')
    dataset=Path(config['dataset']);cases={c['episode_id']:c for c in json.loads((dataset/'plan.json').read_text())['episodes']}
    rows=[]
    for eid,(_,pool) in zip(episode_ids,roles):
        case=cases[eid]
        if case['split']!='validation':raise ValueError('Selection metadata split differs')
        source=dataset/'episodes'/eid
        spec=json.loads((source/'truth/boundaries.json').read_text())
        menu=json.loads((pool/'public'/eid/'action-menu.json').read_text())
        with np.load(source/'reference.npz') as z:rest=z['rest_points_m'].astype(np.float32);tets=z['tetrahedra']
        physical=physical_vector(case,spec,rest);bc,_=category_labels(case)
        actions=menu['candidates']
        if [a['action_id'] for a in actions]!=['poke_0','poke_1','poke_2','poke_3']:
            raise ValueError('Selection menu differs from fixed four candidates')
        for action in actions:
            r={'episode_id':eid+'/'+action['action_id'],'group_id':case['group_id'],
                'role':'selection_control','split':'validation','bc':bc,'force':action['force_type_index'],
                'physical':control_physical(physical,action,menu['known_handle_centers_m']),
                'rest':rest,'tets':tets,'times':np.asarray(menu['sensor']['times_s'],dtype=np.float32),
                'path':str(pool/'private'/eid/action['action_id']),
                'recipe_sha256':case['recipe_sha256']}
            stratum(r)
            expected=np.eye(3)[min(int(action['action_id'][-1]),2)]*.02
            np.testing.assert_allclose(action['peak_force_N'],expected,atol=1e-12)
            np.testing.assert_allclose(r['times'],np.arange(31)/10,atol=1e-6)
            rows.append(r)
    return rows


def score(predictions, records):
    rows=[]
    for prediction,r in zip(predictions,records):
        target=np.asarray(r['q']);rest=np.asarray(r['rest']);times=r['times']
        rmse=lambda x:float(np.sqrt(np.mean(np.square(x))))
        actual=rmse(target-rest);error=rmse(prediction-target)
        rows.append({'episode_id':r['episode_id'],'group_id':r['group_id'],'bc':int(r['bc']),
            'action_id':stratum(r)[1],'coordinate_rmse_m':error,'static_coordinate_rmse_m':actual,
            'normalized_rmse_0p04mm_floor':error/max(actual,.00004),
            'predicted_rms_motion_m':rmse(prediction-rest),'actual_rms_motion_m':actual,
            'amplitude_ratio':rmse(prediction-rest)/max(actual,1e-12),
            'late_2to3_coordinate_rmse_m':rmse((prediction-target)[times>=2.-1e-6])})
    def aggregate(selected):
        return {key:float(np.mean([r[key] for r in selected])) for key in
                ['coordinate_rmse_m','static_coordinate_rmse_m','normalized_rmse_0p04mm_floor']}
    return {'aggregate':aggregate(rows),'by_bc':{str(bc):aggregate([r for r in rows if r['bc']==bc]) for bc in sorted(set(r['bc'] for r in rows))},
        'by_action':{a:aggregate([r for r in rows if r['action_id']==a]) for a in sorted(set(r['action_id'] for r in rows))},'episodes':rows}


def run(campaign, output):
    import torch
    from campaign_control_data import load_control_snapshot
    from train_gaussian_control_forward import collect_control_records
    from train_gaussian_inverse import load_frozen_operator
    config=json.loads(Path(campaign).read_text());output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    train_ids=config['new_train_control_ids'];selection_ids=config['new_validation_roles']['selection']
    if train_ids!=[f't{i:04d}' for i in range(16,48)] or selection_ids!=[f'v{i:04d}' for i in range(8)]:
        raise ValueError('Fixed full32 TRAIN / selection8 design changed')
    source_names=['campaign_forward_coverage.py','campaign_control_data.py','train_gaussian_control_forward.py',
                  'gaussian_probe_roles.py','gaussian_operator.py','diagnose_gaussian_forward.py','train_gaussian_inverse.py']
    sources={name:Path(__file__).with_name(name).read_bytes() for name in source_names}
    (output/'provenance').mkdir()
    for name,content in sources.items():(output/'provenance'/name).write_bytes(content)
    protocol={'campaign_sha256':digest(campaign),'methods':list(METHODS),'rbf_ridge':RIDGE,
        'bandwidth':'Median positive TRAIN pair distance within BC/action stratum',
        'descriptor':list(DESCRIPTOR),'standardization':'TRAIN stratum only; LOO refits after excluding whole layout',
        'stationary_scaling':'FE methods interpolate displacement*E/|F|, undo at query; moving BC remains raw',
        'moving_bc_index':3,'new_train_ids':train_ids,'selection_ids':selection_ids,
        'no_passive_targets':True,'no_gate_calibration_zero_diagonal_test_labels':True,
        'known_physics_forward_diagnosis_only':True,'source_sha256':{n:hashlib.sha256(b).hexdigest() for n,b in sources.items()},
        'qualification':'Fixed nonparametric diagnostic, not a new inverse model. F/E is an approximate scaling, not exact dynamic similarity. Public hardware uses handle centers only; omitted attachment patch details remain a representation confound.'}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    old=collect_control_records(Path(config['dataset']),Path(config['old_train_pool']),Path('configs/gaussian_probe_roles_v01.json'))
    expanded=load_control_snapshot(config,train_ids,'fit');selection=load_selection_metadata(config,selection_ids)
    if (len(old),len(expanded),len(selection))!=(48,128,32):raise ValueError('Fixed snapshot incomplete')
    train_sets={'old48':old,'all176':old+expanded};predictions={};loo={};coverage={}
    catalog={name:[{k:r.get(k) for k in ['episode_id','group_id','split','role','path','trajectory_sha256','recipe_sha256']}
                   for r in records] for name,records in train_sets.items()}
    catalog['selection_metadata_only']=[{k:r.get(k) for k in ['episode_id','group_id','split','role','path','recipe_sha256']} for r in selection]
    (output/'catalog.json').write_text(json.dumps(catalog,indent=2)+'\n')
    anchor=old[0]['rest']
    for r in old+expanded+selection:
        np.testing.assert_allclose(r['rest'],anchor,atol=0,rtol=0)
        np.testing.assert_allclose(r['times'],old[0]['times'],atol=1e-6)
    started=time.monotonic()
    for name,records in train_sets.items():
        coverage[name]=[]
        for method in METHODS:
            model=ResponseInterpolator(method).fit(records)
            predictions[name+'_'+method]=np.stack([model.predict(r) for r in selection]).astype(np.float32)
            loo[name+'_'+method]=leave_one_layout_out(records,method)
            if method==METHODS[0]:coverage[name]=[{'episode_id':r['episode_id'],**model.coverage(r)} for r in selection]
            print(json.dumps({'training_set':name,'method':method,'loo_complete':True,'selection_labels_opened':False}),flush=True)
    predictions['static']=np.stack([np.broadcast_to(r['rest'],(len(r['times']),*r['rest'].shape)) for r in selection])
    checkpoints={}
    for name,path in [('original_passive',config['original_passive_forward']),('original_controlled',config['original_control_forward'])]:
        model=load_frozen_operator(path,'cpu');checkpoints[name]={'path':path,'sha256':digest(path)}
        values=[]
        with torch.no_grad():
            for r in selection:
                values.append(model(torch.tensor(r['rest']),torch.tensor(r['times']),torch.tensor(r['physical'][None]),
                    torch.tensor([r['bc']]),torch.tensor([r['force']]))[0].cpu().numpy())
        predictions[name]=np.stack(values);del model
    # Durable prediction commitment BEFORE any selection trajectory is loaded.
    np.savez_compressed(output/'selection-predictions-before-labels.npz',episode_ids=np.array([r['episode_id'] for r in selection]),**predictions)
    np.savez_compressed(output/'train-loo-predictions.npz',**loo)
    commitment={'selection_predictions_sha256':digest(output/'selection-predictions-before-labels.npz'),
        'train_loo_predictions_sha256':digest(output/'train-loo-predictions.npz'),
        'selection_response_labels_opened':False,'checkpoints':checkpoints,'elapsed_s':time.monotonic()-started}
    (output/'prediction-commitment.json').write_text(json.dumps(commitment,indent=2)+'\n')
    actual=load_control_snapshot(config,selection_ids,'select')
    for meta,label in zip(selection,actual):
        assert meta['episode_id']==label['episode_id']
        np.testing.assert_array_equal(meta['physical'],label['physical'])
    results={'protocol':protocol,'prediction_commitment':commitment,'coverage':coverage,
        'train_layouts':{k:len(set(layout_id(r) for r in rows)) for k,rows in train_sets.items()},
        'training_leave_one_layout_out':{k:score(v,train_sets[k.split('_')[0]]) for k,v in loo.items()},
        'selection8':{k:score(v,actual) for k,v in predictions.items()},
        'labels_opened':'Selection8 candidate responses only, AFTER saving all predictions; no gate/calibration/zero/diagonal/TEST',
        'elapsed_s':time.monotonic()-started}
    for name,content in sources.items():
        if Path(__file__).with_name(name).read_bytes()!=content:raise RuntimeError('Diagnostic source changed')
    (output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps({'output':str(output),'selection8':{k:v['aggregate'] for k,v in results['selection8'].items()},'elapsed_s':results['elapsed_s']},indent=2))
    return results


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',default='configs/gaussian_campaign_v01.json');p.add_argument('--output',required=True)
    a=p.parse_args();run(a.campaign,a.output)
