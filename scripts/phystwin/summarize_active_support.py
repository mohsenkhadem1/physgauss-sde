"""Seal verification, paired object-level comparisons and numerical audit figures."""
from pathlib import Path
import hashlib
import json
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.phystwin.active_support_reference import BASE,ROOT,LAYOUTS,ActiveBlock,protocol


def main():
    spec=protocol();out=ROOT/'reports/phystwin-active-support-v03';out.mkdir(exist_ok=True)
    seal=json.loads((BASE/'analysis/seal.json').read_text())
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in seal['hashes'].items())
    for name in ['development-checks.json','development-summary.json','calibration-summary.json','seal.json','test-summary.json','test-utility.json','test-choice-refinement.json']:
        shutil.copyfile(BASE/'analysis'/name,out/name)
    shutil.copyfile(BASE/'protocol.json',out/'protocol.json')
    mse={k:np.array(v) for k,v in json.loads((BASE/'analysis/test-object-mse.json').read_text()).items()}
    rng=np.random.default_rng(801)
    # The frozen test has six independent parameter draws per fixture family.
    groups=[np.array([i for i,r in enumerate(spec['test']) if r['layout']==layout]) for layout in LAYOUTS]
    ix=np.concatenate([rng.choice(g,(10000,len(g)),replace=True) for g in groups],axis=1)
    comparisons={}
    for name in ['fixed','random','fixture_side_heuristic','left','right']:
        benefit=np.sqrt(mse[name][ix].mean(1))-np.sqrt(mse['adaptive'][ix].mean(1))
        comparisons[name]=dict(adaptive_rmse_improvement_m=float(np.sqrt(mse[name].mean())-np.sqrt(mse['adaptive'].mean())),
            paired_stratified_bootstrap95_m=np.quantile(benefit,[.025,.975]).tolist())
    mesh=[]
    for i in range(12):
        fields=[np.load(BASE/'fields'/f'development-{i}-n{n}.npz')['fields'] for n in [2,4,8]]
        coarse=float(np.sqrt(np.mean((fields[0]-fields[1])**2)));fine=float(np.sqrt(np.mean((fields[1]-fields[2])**2)))
        row=dict(object=i,coarse_n2_n4_rms_m=coarse,fine_n4_n8_rms_m=fine,decreasing=bool(fine<coarse))
        if i in [2,11]:
            f12=np.load(BASE/'fields'/f'development-{i}-n12.npz')['fields']
            row['n8_n12_rms_m']=float(np.sqrt(np.mean((fields[2]-f12)**2)))
        mesh.append(row)
    allchecks=np.concatenate([np.load(p)['checks'] for p in (BASE/'fields').glob('*.npz')])
    test=json.loads((out/'test-summary.json').read_text());audit=json.loads((out/'test-choice-refinement.json').read_text())
    success=dict(beats_calibration_selected_fixed=comparisons['fixed']['paired_stratified_bootstrap95_m'][0]>0,
        calibrated_adaptive_object_coverage_at_least90=test['policies']['adaptive']['calibrated_object_all_repeats_coverage']>=.9,
        test_utility_choice_refinement_passed=test['utility_stable']==test['utility_trials'],
        posterior_grid_action_agreement=audit['choice_agreement']==audit['trials'],
        beats_fixture_heuristic=comparisons['fixture_side_heuristic']['paired_stratified_bootstrap95_m'][0]>0)
    numerical=dict(maximum_residual_N=float(allchecks[:,0].max()),minimum_J=float(allchecks[:,1].min()),
        maximum_nodal_displacement_m=float(allchecks[:,4].max()),frozen_hashes_verified=len(seal['hashes']),
        numerical_scope='n8 finite-mesh targets, not continuum truth; n12 diagnostic at two development parameters',mesh=mesh)
    for name,obj in [('comparison.json',comparisons),('numerical-audit.json',numerical),('decision.json',success)]:
        (out/name).write_text(json.dumps(obj,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    policies=['initial','fixed','random','adaptive','fixture_side_heuristic']
    labels=['Initial','Best fixed\n(calibration)','Random','Expected-variance\npolicy','Fixture-side\nheuristic']
    fig,axes=plt.subplots(1,3,figsize=(14,4.6),layout='constrained')
    values=[test['policies'][k] for k in policies];colors=['#aaa','#c58040','#b1a36a','#246da8','#42916a']
    for ax,key,scale,title in zip(axes,['rmse_m','raw_coordinate_coverage','calibrated_mean_width_m'],[1000,100,1000],
                                  ['Held-field RMSE (mm)','Raw coordinate coverage (%)','Calibrated band width (mm)']):
        ax.bar(np.arange(5),[v[key]*scale for v in values],color=colors)
        ax.set_xticks(np.arange(5),labels,rotation=25,ha='right');ax.set_title(title);ax.grid(axis='y',alpha=.2)
    axes[1].axhline(90,ls='--',color='grey');axes[1].set_ylim(0,105)
    fig.suptitle('24 held-out objects across four fixture families; same observation budget per probe')
    fig.savefig(out/'policy-comparison.png',dpi=160)
    # Fixed illustrative selection: first test object in each family, first noise repeat.
    pred=np.load(BASE/'analysis/test-predictions.npz');rest=ActiveBlock(2,3000,400,'left_full').rest
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for l,ax in enumerate(axes.ravel()):
        obj=l;row=obj*4;choice=int(pred['adaptive'][row]);truth=pred['truth'][row].reshape(81,3)+rest
        mean=pred[f'{choice}_mean'][row].reshape(81,3)+rest
        ax.scatter(rest[:,0]*1000,rest[:,1]*1000,c='#ccc',s=9,label='Rest')
        ax.scatter(truth[:,0]*1000,truth[:,1]*1000,c='#3c8d45',s=12,label='n8 FEM reference')
        ax.scatter(mean[:,0]*1000,mean[:,1]*1000,c='#c47b35',s=17,marker='+',label='Posterior prediction')
        ax.set_aspect('equal');ax.set_xlabel('x (mm)');ax.set_ylabel('y (mm)')
        ax.set_title(f'{LAYOUTS[l]} | chosen push: {"left" if choice==1 else "right"}')
        ax.grid(alpha=.2)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside lower center',ncol=3)
    fig.suptitle('Held oblique response after probing; x–y projection, no deformation amplification')
    fig.savefig(out/'held-predictions.png',dpi=160)
    print(json.dumps(dict(comparisons=comparisons,decision=success,n12=[r for r in mesh if 'n8_n12_rms_m' in r]),indent=2))

if __name__=='__main__':main()
