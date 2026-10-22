"""Prospective small 3D log-NH E/K reference and reset-action study (CPU only).

Known spring location/law, idealized endpoint displacements, independent known
Gaussian sensor. Not image perception, unknown layouts, GPU stable-NH, learned
uncertainty or real-world calibration. No study observations before prepare.
"""
from __future__ import annotations
import argparse,hashlib,json,os,resource,signal,time,traceback
from pathlib import Path
import numpy as np
from scipy.integrate import cubature
from scipy.stats import beta
from experiments import campaign_c38_bar_reference as law
from experiments.campaign_c38_bar_posterior import BarPosteriorBatch
from experiments.campaign_c38_actions import select_reset_actions,freeze_choices
from experiments.campaign_c38_study import paired_object_interval

ROOT=Path(__file__).resolve().parents[1]
SOURCES=['experiments/run_campaign_c38_bar_active.py','experiments/campaign_c38_bar_reference.py','experiments/campaign_c38_bar_posterior.py','experiments/campaign_c38_actions.py','experiments/campaign_c38_study.py','experiments/campaign_point_active_views.py','experiments/censored_range_observation.py','experiments/campaign_nonlinear_bar_inverse.py','experiments/campaign_static_uniaxial_bar.py','experiments/campaign_static_p2_block.py','experiments/campaign_static_supported_block.py','experiments/campaign_sparse_nonlinear_gaussian.py','experiments/boundary_block.py','experiments/diagnose_nonlinear_physical_probes.py']
COUNT=64;INDEPENDENT=(0,9,18,27,36,45,54,63)
SEEDS={'prior':2026122300,'prefix_camera':2026122301,'prefix_noise':2026122302,'action_noise':2026122303,'fantasies':2026122304,'random_policy':2026122305,'bootstrap':2026122306}
CRITERIA={'relative_normalizer':5e-5,'marginal_CDF':5e-4,'E_quantile_Pa':1.,'K_quantile_N_m':.05,'held_quantile_m':1e-6,'field_mean_m':1e-7,'field_std_m':2e-7,'quadrature_cloud_normalizer':5e-5,'quadrature_cloud_field_moment_m':2e-6,'independent_relative_normalizer':5e-5,'independent_field_mean_m':2e-7}


def safe(x):
    if isinstance(x,dict):return {str(k):safe(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [safe(v) for v in x]
    if isinstance(x,np.ndarray):return safe(x.tolist())
    if isinstance(x,np.generic):return x.item()
    return x


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(safe(x),indent=2,allow_nan=False)+'\n')


def prepare(protocol,output):
    path=Path(protocol);out=Path(output).resolve()
    if path.exists() or out.exists():raise FileExistsError('Fresh protocol and output only')
    design=dict(schema='c38-bar-active-reference-v01',status='PROSPECTIVELY_FROZEN_BEFORE_NOISE',objects=COUNT,source_sha256={p:sha(ROOT/p) for p in SOURCES},output=str(out.relative_to(ROOT)),cpu_threads=2,wall_cap_seconds=300,observations_generated=False,
        constitutive_law='Compressible logarithmic Neo-Hookean: mu/2*(tr(F.T F)-3)-mu*log(J)+lambda/2*log(J)^2; NOT polynomial stable-NH GPU model',size_m=law.SIZE,poisson=.35,density_kg_m3=625.,static_density_does_not_enter_equilibrium=True,
        support='Known uniform normal left spring with total K; y=0/z=0 symmetry rollers; consistent P2 face Gram',prior='Independent uniform log E[4500,9000] Pa and log K[100,400] N/m',sensor=dict(kind='Idealized measured endpoint x displacement, NOT rendered camera depth',sigma_m=law.SIGMA_M,measurements_per_frame=1,frames_per_acquisition=1,independent_measurement_noise=True),
        prefix=dict(force_N=1.,camera='Prospectively random left/right endpoint, independent of physical latents'),actions=['left_endpoint_at_F1','right_endpoint_at_F1','F2_reset_probe_at_prefix_endpoint'],acquisition_duration_s=.2,one_action_per_policy=True,canonical_target=dict(force_N=1.5,kind='Static held-force INTERPOLATION from common rest, no temporal forecasting',mass_average='Exact 2x2x2 positive material volume quadrature'),
        policies=['fixed_left','fixed_right','random','opposite_endpoint','information_gain','predictive_risk'],choice_before_all_action_outcomes_and_target_labels=True,common_noise_per_object_action=True,physical_reset_assumed=True,acquisition_budget_scope='0.2s is idealized equal-cost accounting, NOT demonstrated dynamic settling or action feasibility',decision_scope='64-fantasy Monte Carlo score/paired-advantage SE reported; integration agreement does NOT prove selected-action ranking convergence',
        integration=dict(lower=dict(rtol=1e-6,atol=1e-12,conditional_order=32),higher=dict(rtol=1e-8,atol=1e-13,conditional_order=64),conditional='Truncated Gaussian probability-coordinate quadrature with exact full finite prior boundaries and 1/c prior',outer='Adaptive intrinsic log-E integration',quantile_bisections=23,independent='Direct adaptive 2D Gauss-Kronrod cubature, rtol1e-8/atol1e-13, all 4 posterior states on fixed 8 objects',independent_indices=INDEPENDENT,design_cloud=dict(outer_order=16,conditional_order=64),fantasies=64,criteria=CRITERIA),
        seeds=SEEDS,coverage='Prior-predictive Bayesian-model population check (64 fresh prior objects); exact binomial CIs, ranks and widths, no forced90percent pass threshold/no claim real-physics or perception calibration',raw_local=True,preserved_failures_no_retries=True)
    path.parent.mkdir(parents=True,exist_ok=True);write(path,design);return sha(path)


def validate(protocol,digest):
    if sha(protocol)!=digest:raise ValueError('Protocol digest changed')
    p=json.loads(Path(protocol).read_text())
    if p['schema']!='c38-bar-active-reference-v01' or p['objects']!=COUNT or p['seeds']!=SEEDS or p['integration']['criteria']!=CRITERIA or set(p['source_sha256'])!=set(SOURCES):raise ValueError('Exact frozen design required')
    for source,d in p['source_sha256'].items():
        if sha(ROOT/source)!=d:raise ValueError('Frozen source changed: '+source)
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
        if os.environ.get(name)!='2':raise ValueError('Two numerical threads explicitly required')
    return p


def summarize(p):
    mean,std=p.moments();field_mean,field_std=p.field_moments(law.material_quadrature());q={}
    for kind in ('E','K','right'):
        q[kind]=np.stack([p.quantile(kind,.05),p.quantile(kind,.5),p.quantile(kind,.95)],axis=1)
    q['left']=law.TARGET_FORCE/q['K'][:,::-1]
    return dict(mean=mean,std=std,quantiles=q,field_mean=field_mean,field_std=field_std,normalizer=p.Z,maximum_adaptive_error_estimate=max(p.errors),cubature_errors_not_rigorous_bounds=True)


def agreement(low,high,plow,phigh):
    cdf=0.
    for kind in ('E','K','right'):
        for j in range(3):cdf=max(cdf,float(np.max(abs(plow.cdf(kind,high['quantiles'][kind][:,j])-phigh.cdf(kind,high['quantiles'][kind][:,j])))))
    relative=np.expm1(np.log(low['normalizer'])-plow.shift-np.log(high['normalizer'])+phigh.shift)
    values=dict(relative_normalizer=float(np.max(abs(relative))),marginal_CDF=cdf,E_quantile_Pa=float(np.max(abs(low['quantiles']['E']-high['quantiles']['E']))),K_quantile_N_m=float(np.max(abs(low['quantiles']['K']-high['quantiles']['K']))),held_quantile_m=float(max(np.max(abs(low['quantiles'][k]-high['quantiles'][k])) for k in ('left','right'))),field_mean_m=float(np.max(abs(low['field_mean']-high['field_mean']))),field_std_m=float(np.max(abs(low['field_std']-high['field_std']))))
    return dict(passed=all(v<=CRITERIA[k] for k,v in values.items()),metrics=values)


def independent(p,index,summary):
    y=p.y[index];F=p.F;sides=p.sides;shift=p.shift[index];X=law.material_quadrature()
    def integrand(unit):
        theta=law.prior_physical(unit);end=law.endpoint_displacements(theta,F);prediction=end[:,np.arange(len(F)),sides];w=np.exp(shift-.5*np.sum(((prediction-y)/law.SIGMA_M)**2,axis=1));delta=(law.oracle_positions(theta,law.TARGET_FORCE,X)-X).reshape(len(theta),-1)/.02
        return np.column_stack([w,w[:,None]*delta])
    answer=cubature(integrand,np.zeros(2),np.ones(2),rule='gk21',rtol=1e-8,atol=1e-13,max_subdivisions=512)
    if answer.status!='converged':raise RuntimeError('Independent 2D adaptive integration did not converge')
    mass=answer.estimate[0];expected=p.Z[index]*p.std*np.sqrt(2*np.pi)/np.log(4.)
    field=X+.02*(answer.estimate[1:]/mass).reshape(X.shape)
    report=dict(relative_normalizer_error=float(abs(mass/expected-1)),field_mean_maximum_error_m=float(np.max(abs(field-summary['field_mean'][index]))),status=answer.status,subdivisions=answer.subdivisions,estimated_absolute_error=float(np.max(answer.error)))
    cdf_errors=[]
    for axis,kind in enumerate(('E','K')):
        for col in (0,2):
            physical=summary['quantiles'][kind][index,col];upper=np.ones(2);upper[axis]=(np.log(physical)-law.LOWER[axis])/(law.UPPER[axis]-law.LOWER[axis])
            a=cubature(lambda u:integrand(u)[:,0],np.zeros(2),upper,rule='gk21',rtol=1e-8,atol=1e-13,max_subdivisions=512)
            if a.status!='converged':raise RuntimeError('Independent 2D marginal CDF did not converge')
            expected_cdf=p.cdf(kind,np.full(p.n,physical))[index];cdf_errors.append(float(abs(a.estimate/mass-expected_cdf)))
    report['maximum_marginal_CDF_error']=max(cdf_errors);report['passed']=report['relative_normalizer_error']<=CRITERIA['independent_relative_normalizer'] and report['field_mean_maximum_error_m']<=CRITERIA['independent_field_mean_m'] and report['maximum_marginal_CDF_error']<=CRITERIA['marginal_CDF']
    return report


def actions():
    return [dict(id=name,kind='view' if j<2 else 'reset_probe',duration_s=.2,frame_count=1,sensor_dimension=1,reset_to_common_rest=True,common_state_sha256=hashlib.sha256(b'C38 bar rest').hexdigest(),target_history_sha256=hashlib.sha256(b'C38 same held1.5N from rest').hexdigest(),force_domain_sha256=hashlib.sha256(b'C38 allowedF1F2').hexdigest()) for j,name in enumerate(('left_F1','right_F1','prefix_endpoint_F2'))]


def ci(success):
    s=int(np.sum(success));n=len(success)
    return dict(successes=s,objects=n,fraction=s/n,exact_binomial_95percent_interval=[0. if s==0 else float(beta.ppf(.025,s,n-s+1)),1. if s==n else float(beta.ppf(.975,s+1,n-s))])


def run(protocol,digest):
    config=validate(protocol,digest);out=ROOT/config['output'];out.mkdir(parents=True,exist_ok=False);start=time.monotonic();state=dict(schema='c38-bar-active-result-v01',protocol={'path':str(Path(protocol).relative_to(ROOT)),'sha256':digest},pid=os.getpid(),completed=False,passed=False,stages=[],raw_local=True)
    def save():write(out/'summary.json',state)
    def timeout(*_):raise TimeoutError('Frozen global300second CPU2 cap reached; partial results preserved')
    old=signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,config['wall_cap_seconds'])
    try:
        # Generate fresh prior objects and early observations only. Action and
        # target values are not materialized before ALL choice records freeze.
        theta=law.prior_physical(np.random.default_rng(SEEDS['prior']).random((COUNT,2)));camera=np.random.default_rng(SEEDS['prefix_camera']).integers(0,2,COUNT);prefix_true=law.endpoint_displacements(theta,[1.])[:,0][np.arange(COUNT),camera];prefix=prefix_true+np.random.default_rng(SEEDS['prefix_noise']).normal(0,law.SIGMA_M,COUNT)
        np.savez(out/'private-objects-and-prefix.npz',theta=theta,prefix_camera=camera,prefix_observation_m=prefix)
        del theta,prefix_true
        groups=[];objects={};choices=[];references=[];X=law.material_quadrature()
        for side in (0,1):
            indices=np.flatnonzero(camera==side);observed=prefix[indices,None]
            low=BarPosteriorBatch(observed,[1.],[side],**config['integration']['lower']);high=BarPosteriorBatch(observed,[1.],[side],**config['integration']['higher']);sl=summarize(low);sh=summarize(high);check=agreement(sl,sh,low,high)
            references.append((indices,None,high))
            state['stages'].append(dict(kind='prefix',indices=indices.tolist(),agreement=check));save()
            print(json.dumps(dict(stage='prefix_reference',side=side,objects=len(indices),passed=check['passed'],elapsed_s=time.monotonic()-start)),flush=True)
            if not check['passed']:raise RuntimeError('Prefix continuous reference did not meet preregistered agreement')
            for local,i in enumerate(indices):
                i=int(i)
                if i in INDEPENDENT:
                    independent_check=independent(high,local,sh);state['stages'].append(dict(kind='prefix_independent',object_index=i,check=independent_check));save()
                    if not independent_check['passed']:raise RuntimeError('Independent prefix reference failed')
                physical,w,cloud=high.quadrature_cloud(local,outer_order=16,conditional_order=64);targets=(law.oracle_positions(physical,law.TARGET_FORCE,X)-X).reshape(len(w),-1);means=w@targets;std=np.sqrt(w@((targets-means)**2))
                cloud['maximum_field_mean_error_m']=float(np.max(abs(means.reshape(X.shape)+X-sh['field_mean'][local])));cloud['maximum_field_std_error_m']=float(np.max(abs(std.reshape(X.shape)-sh['field_std'][local])))
                if cloud['normalizer_relative_error']>CRITERIA['quadrature_cloud_normalizer'] or max(cloud['maximum_field_mean_error_m'],cloud['maximum_field_std_error_m'])>CRITERIA['quadrature_cloud_field_moment_m']:raise RuntimeError('Design quadrature cloud unresolved')
                endpoints=law.endpoint_displacements(physical,[1.,2.]);predictions=np.stack([endpoints[:,0,0],endpoints[:,0,1],endpoints[:,1,side]])[:,:,None]
                selection=select_reset_actions(targets,predictions,np.eye(1)*law.SIGMA_M**2,w,actions(),object_index=i,fixed_index=0,fantasies=64,fantasy_seed=SEEDS['fantasies']+i,random_seed=SEEDS['random_policy']);selection['policies']={'fixed_left':0,'fixed_right':1,'random':selection['policies']['random'],'opposite_endpoint':1-side,'information_gain':selection['policies']['information_gain'],'predictive_risk':selection['policies']['predictive_risk']}
                path=out/f'choice-{i:03d}.json';choice_sha=freeze_choices(path,selection,{str(ROOT/p):d for p,d in config['source_sha256'].items()});choices.append(dict(object_index=i,path=path.name,sha256=choice_sha))
                objects[i]={'prefix':{k:sh[k][local] for k in ('mean','std','field_mean','field_std')},'prefix_quantiles':{k:v[local] for k,v in sh['quantiles'].items()},'cloud':cloud,'selection':selection,'post':{}}
            groups.append((side,indices))
        # Immutable all-object choice barrier before reading latent truth again
        # and synthesizing ANY acquired action outcomes or canonical target.
        write(out/'choices-frozen.json',dict(completed=True,objects=COUNT,choices=sorted(choices,key=lambda x:x['object_index']),all_choices_frozen_before_action_outcomes=True));state['choices_freeze_sha256']=sha(out/'choices-frozen.json');save()
        private=np.load(out/'private-objects-and-prefix.npz');theta=private['theta'];outcomes=np.empty((COUNT,3))
        for i in range(COUNT):
            end=law.endpoint_displacements(theta[i:i+1],[1.,2.])[0];truth=np.array([end[0,0],end[0,1],end[1,camera[i]]]);noise=np.array([np.random.default_rng(np.random.SeedSequence([SEEDS['action_noise'],i,j])).normal(0,law.SIGMA_M) for j in range(3)]);outcomes[i]=truth+noise
        np.savez(out/'action-observations.npz',observations_m=outcomes);del theta,truth,private
        for side,indices in groups:
            for action in range(3):
                F=[1.,1. if action<2 else 2.];sides=[side,action if action<2 else side];obs=np.stack([prefix[indices],outcomes[indices,action]],axis=1)
                low=BarPosteriorBatch(obs,F,sides,**config['integration']['lower']);high=BarPosteriorBatch(obs,F,sides,**config['integration']['higher']);sl=summarize(low);sh=summarize(high);check=agreement(sl,sh,low,high)
                references.append((indices,action,high))
                state['stages'].append(dict(kind='post_action',prefix_side=side,action=action,indices=indices.tolist(),agreement=check));save()
                print(json.dumps(dict(stage='post_reference',side=side,action=action,objects=len(indices),passed=check['passed'],elapsed_s=time.monotonic()-start)),flush=True)
                if not check['passed']:raise RuntimeError('Post-action continuous reference failed')
                for local,i in enumerate(indices):
                    i=int(i)
                    if i in INDEPENDENT:
                        independent_check=independent(high,local,sh);state['stages'].append(dict(kind='post_independent',object_index=i,action=action,check=independent_check));save()
                        if not independent_check['passed']:raise RuntimeError('Independent post-action reference failed')
                    objects[i]['post'][action]={k:sh[k][local] for k in ('mean','std','field_mean','field_std')};objects[i]['post'][action]['quantiles']={k:v[local] for k,v in sh['quantiles'].items()}
        # All predictions/intervals are persisted before canonical target and
        # parameter labels enter evaluation; no test-based policy choice.
        write(out/'posterior-states-frozen.json',dict(completed=True,objects=COUNT,states=objects));state['posterior_states_freeze_sha256']=sha(out/'posterior-states-frozen.json');save()
        theta=np.load(out/'private-objects-and-prefix.npz')['theta'];target=law.oracle_positions(theta,law.TARGET_FORCE,X);held=law.endpoint_displacements(theta,[law.TARGET_FORCE])[:,0]
        for indices,action,p in references:
            ranks=np.stack([p.cdf('E',theta[indices,0]),p.cdf('K',theta[indices,1])],axis=1)
            for local,i in enumerate(indices):
                key='prefix' if action is None else action
                if action is None:objects[int(i)]['prefix']['marginal_truth_CDF']=ranks[local]
                else:objects[int(i)]['post'][key]['marginal_truth_CDF']=ranks[local]
        policies=['prefix_only']+config['policies'];report={};scores={}
        for policy in policies:
            mse=[];ranks=[];covered={k:[] for k in ('E','K','left','right')};width={k:[] for k in covered}
            for i in range(COUNT):
                o=objects[i]
                if policy=='prefix_only':q=o['prefix'];quant=o['prefix_quantiles']
                else:q=o['post'][o['selection']['policies'][policy]];quant=q['quantiles']
                mse.append(float(np.mean((q['field_mean']-target[i])**2)))
                ranks.append(q['marginal_truth_CDF'])
                for j,key in enumerate(('E','K','left','right')):
                    truth=theta[i,j] if j<2 else held[i,j-2];covered[key].append(bool(quant[key][0]<=truth<=quant[key][2]));width[key].append(float(quant[key][2]-quant[key][0]))
            ranks=np.asarray(ranks)
            scores[policy]=mse;report[policy]=dict(pooled_3D_coordinate_RMSE_m=float(np.sqrt(np.mean(mse))),mean_object_3D_coordinate_RMSE_m=float(np.mean(np.sqrt(mse))),coverage_90percent={k:ci(v) for k,v in covered.items()},mean_90percent_width={k:float(np.mean(v)) for k,v in width.items()},parameter_marginal_truth_CDF_10bin_histogram={key:np.histogram(ranks[:,j],np.linspace(0,1,11))[0].tolist() for j,key in enumerate(('E','K'))},parameter_marginal_truth_CDF_mean=dict(zip(('E','K'),ranks.mean(0).tolist())))
        comparisons={}
        for policy in ('information_gain','predictive_risk'):
            for other in ('fixed_left','fixed_right','random','opposite_endpoint'):
                comparisons[policy+'_minus_'+other]=paired_object_interval(scores[policy],scores[other],seed=SEEDS['bootstrap'])
        decision={}
        for kind,scorekey,sekey,minimize in (('predictive_risk','expected_target_variance_m2','risk_advantage_mc_se_m2',True),('information_gain','expected_information_gain_nats','information_advantage_mc_se_nats',False)):
            uncertain=0;maxse=0.
            for i in range(COUNT):
                selection=objects[i]['selection'];values=np.asarray(selection[scorekey]);order=np.argsort(values if minimize else -values);gap=abs(values[order[1]]-values[order[0]]);se=selection[sekey][order[1]];maxse=max(maxse,se);uncertain+=int(gap<=2*se)
            decision[kind]=dict(objects_with_top_two_gap_at_most_two_paired_MC_SE=uncertain,maximum_runnerup_advantage_MC_SE=maxse,fantasies=64,not_a_ranking_convergence_proof=True)
        state.update(completed=True,passed=True,objects=COUNT,independent_integration_checks=32,policies=report,paired_object_MSE_intervals=comparisons,decision_MC_diagnostics=decision,choices={k:[int(objects[i]['selection']['policies'][k]) for i in range(COUNT)] for k in config['policies']},claim='Numerical reference and prior-predictive model check for one known homogeneous 3D log-NH spring family; not unknown layouts, perception, population real-physics calibration or online probing',acquisition_budget_scope=config['acquisition_budget_scope'],all_old_results_unchanged=True)
    except BaseException as error:
        state.update(completed=False,passed=False,failure=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc(limit=5)))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old);state['elapsed_s']=time.monotonic()-start;state['peak_RSS_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;save()
    print(json.dumps(dict(completed=state['completed'],passed=state['passed'],objects=COUNT,elapsed_s=state['elapsed_s'],output=str(out)),allow_nan=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='mode',required=True);p=sub.add_parser('prepare');p.add_argument('--protocol',required=True);p.add_argument('--output',required=True);r=sub.add_parser('run');r.add_argument('--protocol',required=True);r.add_argument('--protocol-sha256',required=True);args=parser.parse_args()
    if args.mode=='prepare':print(prepare(args.protocol,args.output))
    else:run(args.protocol,args.protocol_sha256)
