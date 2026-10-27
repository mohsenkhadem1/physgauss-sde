"""v05 continuous posterior, physical uncertainty audit and frozen probing comparison."""
from pathlib import Path
import argparse,hashlib,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse.linalg import spsolve
from scripts.phystwin.qualified_reference import BASE,ROOT,BOUNDS,LAYOUTS,ACTIONS,protocol,write_json,sample81,nonrigid,observation_points
from scripts.phystwin.active_support_policy import Experiment,indices,score,calibration
from scripts.phystwin.uncertainty_posterior import discrepancy,marginal_quantiles
OUT=BASE/'analysis'


def bank_path(mesh,name):return (BASE/'fields' if mesh==4 else BASE/'banks'/f'n{mesh}')/(name+'.npz')


def interpolators(model):
    axes=[np.linspace(*b,9) for b in BOUNDS]
    result=[]
    for l in range(4):
        values=[]
        for i in range(9):
            for j in range(9):
                name=f'bank9-{l}-{i}-{j}';v=np.load(bank_path(model['mesh'],name))['fields']
                if model['coarse'] is not None:v=v+(v-np.load(bank_path(model['coarse'],name))['fields'])/((model['mesh']/model['coarse'])**3-1)
                values.append(v)
        result.append(RegularGridInterpolator(axes,np.stack(values).reshape(9,9,972),method='cubic',solver=spsolve))
    return result


def load_qualified(split,i):
    audit=json.loads((BASE/'qualification'/f'{split}-{i}.json').read_text());assert audit['qualified']
    return np.load(BASE/'fields'/f'{split}-{i}-qualified.npz')


def direct_development(i,n):
    paths=[BASE/'refinements'/f'development-{i}'/f'n{n}-q3-{action}.npz' for action in ACTIONS]
    if all(p.exists() for p in paths):return np.stack([sample81(np.load(p)['sampled_u']) for p in paths])
    return np.load(bank_path(n,f'development-{i}'))['fields']


def fit(spec,interp,model):
    pred=[];truth=[];checks=[]
    for i,r in enumerate(spec['development']):
        approximate=interp[LAYOUTS.index(r['layout'])](np.log([[r['E'],r['K']]]))[0]
        z=load_qualified('development',i);direct=direct_development(i,model['mesh'])
        if model['coarse'] is not None:direct=direct+(direct-direct_development(i,model['coarse']))/((model['mesh']/model['coarse'])**3-1)
        interpolation=float(np.sqrt(np.mean((approximate-direct.ravel())**2)))
        forward=float(np.sqrt(np.mean((approximate-z['fields'].ravel())**2)))
        best=z['nonrigid'][1:3];j=int(np.argmax(best[:,0]));deformation=best[j]
        checks.append(dict(object=i,interpolation_rms_m=interpolation,forward_rms_m=forward,forward_passed=forward<=spec['interpolation_gate_rms_m'],nonrigid_rms_m=float(deformation[0]),nonrigid_fraction=float(deformation[0]/deformation[1]),
            interpolation_passed=interpolation<=spec['interpolation_gate_rms_m'],deformation_passed=bool(deformation[0]>=spec['probe_nonrigid_rms_gate_m'] and deformation[0]/deformation[1]>=spec['probe_nonrigid_fraction_gate'])))
        pred.append(approximate);truth.append(z['fields'].ravel())
    diagnostic=dict(objects=checks,forward_passed=all(r['forward_passed'] for r in checks),interpolation_passed=all(r['interpolation_passed'] for r in checks),deformation_passed=all(r['deformation_passed'] for r in checks))
    return *discrepancy(np.array(truth)-np.array(pred)),diagnostic


def select_model(spec):
    path=OUT/'model-selection.json'
    if path.exists():return json.loads(path.read_text())['model']
    rule=json.loads((BASE/'surrogate-refinement-addendum.json').read_text());trials=[]
    for model in rule['candidates']:
        if not all(bank_path(model['mesh'],f'bank9-{l}-{i}-{j}').exists() for l in range(4) for i in range(9) for j in range(9)):
            write_json(OUT/'model-selection-progress.json',dict(trials=trials,required_bank_mesh=model['mesh']));raise RuntimeError(f"Generate n={model['mesh']} bank before model selection")
        interp=interpolators(model);_,_,diag=fit(spec,interp,model);trials.append(dict(model=model,diagnostic=diag))
        if diag['forward_passed'] and diag['interpolation_passed']:
            write_json(path,dict(model=model,trials=trials,rule=rule));return model
    write_json(OUT/'model-selection-progress.json',dict(trials=trials));raise RuntimeError('No candidate forward surrogate passes; no inference seal')


def observations(spec,split):
    rng=np.random.default_rng(dict(development=701,calibration=702,test=703)[split]);prefix,probes,target=indices()
    ys=[[],[],[]];truth=[]
    for i in range(len(spec[split])):
        f=load_qualified(split,i)['fields'].ravel()
        for _ in range(spec['noise_repeats']):
            noise=rng.normal(0,spec['noise_sigma_m'],972)
            for a,ix in enumerate([prefix]+[np.r_[prefix,q] for q in probes]):ys[a].append(f[ix]+noise[ix])
            truth.append(f[target])
    return [np.array(y) for y in ys],np.array(truth)


def predictions(ex,ys,size):
    pred=ex.predict(ys)
    for m,y,p in zip(ex.models,ys,pred):
        weights=m.weights(y).cpu().numpy().reshape(len(y),4,size,size)
        p['log_parameter_quantiles']=np.stack([marginal_quantiles(weights.sum((1,3)),np.linspace(*BOUNDS[0],size)),marginal_quantiles(weights.sum((1,2)),np.linspace(*BOUNDS[1],size))],axis=1)
        p['effective_nodes']=1/np.sum(weights.reshape(len(y),-1)**2,axis=1)
    return pred


def convergence(a,b,spec):
    row={key:max(float(np.max(abs(p[key]-q[key]))) for p,q in zip(a,b)) for key in ['mean','sd','log_parameter_quantiles']}
    row['passed']=max(row['mean'],row['sd'])<=spec['quadrature_gate_mean_sd_m'] and row['log_parameter_quantiles']<=spec['quadrature_gate_log_quantile']
    return row


def physical_score(pred,spec,split,choices):
    pars=np.repeat(np.log([[r['E'],r['K']] for r in spec[split]]),spec['noise_repeats'],axis=0)
    labels=np.repeat([LAYOUTS.index(r['layout']) for r in spec[split]],spec['noise_repeats']);out={}
    for name,choice in choices.items():
        if name=='random':continue
        col=np.arange(len(pars));q=np.stack([p['log_parameter_quantiles'] for p in pred])[choice,col]
        prob=np.stack([p['layout_probability'] for p in pred])[choice,col]
        covered=(pars>=q[:,:,0])&(pars<=q[:,:,2]);error=abs(q[:,:,1]-pars)
        out[name]=dict(E_90_coverage=float(covered[:,0].mean()),K_90_coverage=float(covered[:,1].mean()),
            E_log_median_MAE=float(error[:,0].mean()),K_log_median_MAE=float(error[:,1].mean()),
            E_mean_log_interval_width=float(np.mean(q[:,0,2]-q[:,0,0])),K_mean_log_interval_width=float(np.mean(q[:,1,2]-q[:,1,0])),
            layout_accuracy=float(np.mean(prob.argmax(1)==labels)),true_layout_mean_probability=float(prob[col,labels].mean()))
    if 'random' in choices:
        branches=physical_score(pred,spec,split,dict(left=np.ones(len(pars),dtype=int),right=np.full(len(pars),2)))
        out['random']={key:.5*(branches['left'][key]+branches['right'][key]) for key in branches['left']}
    return out


def deformation_errors(predicted,truth):
    rest=observation_points();a=predicted.reshape(-1,81,3)+rest;b=truth.reshape(-1,81,3)+rest
    a=a-a.mean(1,keepdims=True);b=b-b.mean(1,keepdims=True)
    v,_,w=np.linalg.svd(np.swapaxes(a,1,2)@b);D=np.broadcast_to(np.eye(3),(len(a),3,3)).copy();D[:,2,2]=np.linalg.det(v@w)
    aligned=np.mean((a@v@D@w-b)**2,axis=(1,2))
    def strain(u):
        u=u.reshape(-1,9,3,3,3);g=np.stack(np.gradient(u,.03,.04,.04,axis=(1,2,3),edge_order=2),axis=-1)
        F=g+np.eye(3);return .5*(np.swapaxes(F,-1,-2)@F-np.eye(3))
    strain_mse=np.mean((strain(predicted)-strain(truth))**2,axis=(1,2,3,4,5))
    return aligned,strain_mse


def deformation_scores(pred,truth,choices,repeats):
    errors=[deformation_errors(p['mean'],truth) for p in pred];out={};objects={}
    for name,choice in choices.items():
        values=[]
        for k in range(2):
            stacked=np.stack([e[k] for e in errors]);v=stacked[1:].mean(0) if name=='random' else stacked[choice,np.arange(len(truth))]
            values.append(v)
        out[name]=dict(aligned_rmse_m=float(np.sqrt(values[0].mean())),observed_grid_strain_rmse=float(np.sqrt(values[1].mean())))
        objects[name]=dict(aligned_mse=values[0].reshape(-1,repeats).mean(1).tolist(),strain_mse=values[1].reshape(-1,repeats).mean(1).tolist())
    return out,objects


def hashes(paths):return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def verify_seal(seal):
    for path,h in seal['hashes'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h,f'Sealed file changed: {path}'


def run(stage,device):
    spec=protocol();OUT.mkdir(parents=True,exist_ok=True);model=select_model(spec)
    if stage=='select':print('SELECTED',model,flush=True);return
    interp=interpolators(model);mu,C,diag=fit(spec,interp,model)
    write_json(OUT/'development-checks.json',diag);assert diag['interpolation_passed'] and diag['forward_passed'],'Forward surrogate gate failed'
    assert diag['deformation_passed'],'Development probes do not provide the required nonrigid deformation'
    if stage!='development':assert (OUT/'development-summary.json').exists(),'Finish development first'
    if stage=='test':
        seal=json.loads((OUT/'seal.json').read_text());assert seal['protocol']==spec;verify_seal(seal)
    split={'development':'development','calibrate':'calibration','test':'test'}[stage];ys,truth=observations(spec,split)
    previous=None;checks=[]
    extension=BASE/'posterior-refinement-extension.json'
    levels=spec['quadrature_sizes']+(json.loads(extension.read_text())['additional_sizes'] if extension.exists() else [])
    for size in levels:
        ex=Experiment(interp,size,mu,C,spec['noise_sigma_m'],device);pred=predictions(ex,ys,size)
        if previous is not None:
            c=dict(coarse=old_size,fine=size,**convergence(previous,pred,spec));checks.append(c);print('POSTERIOR',split,c,flush=True)
            if c['passed']:break
        previous=pred;old_size=size;del ex
    assert checks and checks[-1]['passed'],'Posterior moments/physical quantiles not converged'
    coarse=Experiment(interp,checks[-1]['coarse'],mu,C,spec['noise_sigma_m'],device)
    utility=[];adaptive=[]
    for i,y in enumerate(ys[0]):
        risks=[ex.choose(y,n) for n in spec['utility_samples'][:2]]
        samples=spec['utility_samples'][1]
        if risks[0].argmin()!=risks[1].argmin():
            samples=spec['utility_samples'][2];risks.append(ex.choose(y,samples))
        choice=int(risks[-1].argmin());stable=bool(risks[-2].argmin()==choice)
        other=coarse.choose(y,samples)
        utility.append(dict(risks=[r.tolist() for r in risks],samples=samples,stable=stable,posterior_grid_agreement=bool(other.argmin()==choice)))
        adaptive.append(choice+1)
        if i%8==0:print('POLICY',split,i,len(ys[0]),flush=True)
    del coarse
    count=len(truth);adaptive=np.array(adaptive);heuristic=np.where(pred[0]['layout_probability'][:,:2].sum(1)>=.5,2,1)
    if stage=='calibrate':
        scale,scores=calibration(pred,truth,spec['noise_repeats']);fixed=1+int(np.argmin([np.mean((p['mean']-truth)**2) for p in pred[1:]]))
    elif stage=='test':scale=seal['scale'];fixed=seal['fixed_action']
    else:scale=1.;fixed=1
    choices=dict(initial=np.zeros(count,dtype=int),fixed=np.full(count,fixed),random=None,adaptive=adaptive,fixture_side_heuristic=heuristic,left=np.ones(count,dtype=int),right=np.full(count,2))
    measures,object_mse=score(pred,truth,choices,scale,spec['noise_repeats'])
    deformation,deformation_objects=deformation_scores(pred,truth,choices,spec['noise_repeats'])
    for name,metrics in deformation.items():measures[name].update(metrics)
    summary=dict(split=split,quadrature=checks,grid_size=size,scale=scale,fixed_action=fixed,policies=measures,
        physical=physical_score(pred,spec,split,choices),utility_stable=sum(r['stable'] for r in utility),utility_trials=count,
        grid_action_agreement=sum(r['posterior_grid_agreement'] for r in utility),heuristic_agreement=float(np.mean(adaptive==heuristic)),
        adaptive_action_counts={str(i):int(np.sum(adaptive==i)) for i in [1,2]},deformation_gate_passed=diag['deformation_passed'])
    if stage=='calibrate':
        paths=list((ROOT/'scripts/phystwin').glob('qualified_*.py'))+[ROOT/'scripts/phystwin/active_support_policy.py',ROOT/'scripts/phystwin/uncertainty_posterior.py',ROOT/'scripts/phystwin/consistent_fixture.py',ROOT/'scripts/phystwin/gpu_multigrid.py',ROOT/'scripts/phystwin/qualify_fixture.py',ROOT/'scripts/phystwin/summarize_fixture.py',OUT/'model-selection.json']+list(BASE.glob('*.json'))
        paths += [ROOT/p for p in ['scripts/phystwin/active_support_reference.py','scripts/phystwin/tensor_fixture.py','scripts/phystwin/requirements-fixture.txt',
            'experiments/campaign_static_p2_block.py','experiments/campaign_static_uniaxial_bar.py',
            'experiments/campaign_static_supported_block.py','experiments/boundary_block.py',
            'experiments/diagnose_nonlinear_physical_probes.py']]
        paths+=list((BASE/'fields').glob('bank*.npz'))+list((BASE/'fields').glob('development*.npz'))+list((BASE/'fields').glob('calibration*.npz'))
        paths+=list((BASE/'banks').glob('**/*.npz'))
        sealed=dict(protocol=spec,scale=scale,fixed_action=fixed,calibration_scores=scores.tolist(),hashes=hashes(paths))
        path=OUT/'seal.json'
        if path.exists():assert json.loads(path.read_text())==sealed,'Seal cannot be overwritten'
        else:write_json(path,sealed)
    np.savez_compressed(OUT/f'{split}-predictions.npz',truth=truth,adaptive=adaptive,heuristic=heuristic,**{f'{i}_{k}':v for i,p in enumerate(pred) for k,v in p.items()})
    write_json(OUT/f'{split}-summary.json',summary);write_json(OUT/f'{split}-object-mse.json',object_mse);write_json(OUT/f'{split}-utility.json',utility)
    write_json(OUT/f'{split}-deformation-object-mse.json',deformation_objects)
    print('COMPLETE',split,json.dumps(summary),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['select','development','calibrate','test'],required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();run(a.stage,a.device)

if __name__=='__main__':main()
