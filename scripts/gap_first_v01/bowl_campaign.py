"""Finite-hypothesis nonlinear bowl identification and prospective risk stress.

Only simulated known metric surface tracks enter inference. The posterior is a
finite discrete distribution; no analytic load elimination applies to contact.
"""
import argparse,hashlib,itertools,json,time
from datetime import datetime
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from .bowl_physics import run,shape_mesh,BowlFEM
from .bowl_qualification import metrics
from .quantiles import gaussian_mixture_quantiles

ROOT=Path('runs/gap-first-v01/bowl')
REPORT=Path('reports/gap-first-v01/bowl')
YOUNG=np.array([2500.,4000.,6500.,10000.])
FRICTION=np.array([.1,.25,.45,.7])
THETA=np.array(list(itertools.product(YOUNG,FRICTION)))
ACTIONS=('calibration_push','calibration_tilt')
SIGMA=.0005
TOLERANCE=.003
DT=2.5e-5
REFERENCE_DT=1.25e-5
PENALTY=38400.
INFERENCE_N=6
REFERENCE_N=12
DEADLINE_UNIX=datetime.fromisoformat('2026-11-07T21:44:00+00:00').timestamp()
MARKERS=np.array([np.ravel_multi_index((i,j,3),(4,4,4)) for i in (2,3) for j in (1,2)])

def nearest_times(r,ts):
    return np.argmin(abs(r['times'][:,None]-np.asarray(ts)[None]),axis=0)

def common_nodes(n):
    a,_=shape_mesh(3);b,_=shape_mesh(n)
    d=np.linalg.norm(a[:,None]-b[None],axis=2);ids=d.argmin(1)
    if np.max(d.min(1))>1e-12:raise ValueError('Fine/coarse material-node correspondence failed')
    return ids

def observations(r,times,ids):
    # Orthographic overhead image axes x,y; metric correspondence coordinates.
    return r['frames'][nearest_times(r,times)][:,ids[MARKERS],:2].reshape(-1)

def target(r,ids):
    return r['frames'][nearest_times(r,[1.1,1.2,1.3])][:,ids].reshape(-1)

def generate(family='fem',n=3):
    verify_sources()
    cls=BowlFEM
    if family=='spring':
        from .bowl_native import BowlNative
        cls=BowlNative
    label=f'{family}-n{n}';directory=ROOT/label;directory.mkdir(parents=True,exist_ok=True)
    ids=common_nodes(n);rows=[]
    for h,(young,friction) in enumerate(THETA):
        paths=[];records=[]
        for interaction in (*ACTIONS,'heldout_push'):
            path=directory/f'h{h:02d}-{interaction}.npz';paths.append(str(path))
            if path.exists():
                r=dict(np.load(path))
                if str(r.get('protocol_sha256',''))!=protocol_hash():raise RuntimeError(f'Cached trajectory lacks matching frozen protocol: {path}')
            else:
                started=time.monotonic()
                try:
                    if time.time()>=DEADLINE_UNIX:raise RuntimeError('Campaign deadline reached before missing trajectory')
                    r=run(young=young*(6 if family=='spring' else 1),friction=friction,n=n,dt=DT if n<=INFERENCE_N else REFERENCE_DT,
                        duration=1.3,interaction=interaction,penalty=PENALTY,system_class=cls,deadline_unix=DEADLINE_UNIX)
                    r['elapsed_s']=time.monotonic()-started
                    r['protocol_sha256']=protocol_hash()
                    np.savez_compressed(path,**r)
                except Exception as e:
                    (directory/f'h{h:02d}-{interaction}-failure.json').write_text(json.dumps({'theta':[young,friction],'error':repr(e),'elapsed_s':time.monotonic()-started},indent=2))
                    continue
            records.append(r)
        if len(records)!=3:
            rows.append(None);continue
        row=dict(initial=observations(records[0],[.08,.12],ids),
                 action=np.array([observations(r,[.94,1.2],ids) for r in records[:2]]),
                 target=target(records[2],ids),reaction=records[2]['reaction_averages'][nearest_times(records[2],[1.1,1.2,1.3])].reshape(-1))
        rows.append(row)
        (directory/'progress.json').write_text(json.dumps({'completed_hypotheses':h+1,'total':len(THETA),'family':family,'n':n}))
        print(label,h,'ok',flush=True)
    if any(r is None for r in rows):
        (directory/'qualification-failure.json').write_text(json.dumps({'status':'No accuracy evaluation: at least one required hypothesis simulation failed','failed_hypotheses':[i for i,r in enumerate(rows) if r is None],'retained_all_attempts':True},indent=2))
        return None
    bank={key:np.array([r[key] for r in rows]) for key in rows[0]}
    bank.update(theta=THETA,markers=MARKERS,noise_sigma_m=SIGMA,common_node_ids=ids)
    np.savez_compressed(ROOT/f'{label}-bank.npz',**bank)
    return bank

def generate_gpu(n=3,interaction=None):
    verify_sources()
    from .bowl_torch import Batch
    directory=ROOT/f'fem-n{n}';directory.mkdir(parents=True,exist_ok=True)
    selected=(*ACTIONS,'heldout_push') if interaction is None else (interaction,)
    for selected_interaction in selected:
        interaction_name=selected_interaction
        if all((directory/f'h{h:02d}-{interaction_name}.npz').exists() for h in range(len(THETA))):continue
        started=time.monotonic();batch=Batch(THETA,n=n,penalty=PENALTY,device='cuda:0')
        try:result=batch.run(duration=1.3,dt=DT if n<=INFERENCE_N else REFERENCE_DT,interaction=interaction_name,
            progress_path=REPORT/f'n{n}-{interaction_name}-progress.json',checkpoint_path=ROOT/'checkpoints'/f'n{n}-{interaction_name}.npz',deadline_unix=DEADLINE_UNIX)
        except Exception as e:
            (directory/f'{interaction_name}-batch-failure.json').write_text(json.dumps({'error':repr(e),'fallback':'After all GPU workers finish, generate mode attempts all missing individual CPU hypotheses and retains every failure'},indent=2))
            return None
        s=batch.base
        for h,(young,friction) in enumerate(THETA):
            r={key:(value[h] if key in ('frames','velocities','reactions','reaction_averages','min_j','max_penetration_m','friction_work_j') else value) for key,value in result.items()}
            r.update(rest=s.rest,tets=s.tets,faces=s.surface_triangles,mass=s.mass,young=young,friction=friction,penalty=PENALTY,interaction=interaction_name,
                elapsed_s=(time.monotonic()-started)/len(THETA),engine='float64 batched GPU parity-qualified against NumPy',protocol_sha256=protocol_hash())
            np.savez_compressed(directory/f'h{h:02d}-{interaction_name}.npz',**r)
        print('GPU',n,interaction_name,'elapsed_s',time.monotonic()-started,flush=True)
    return generate('fem',n) if interaction is None else None

def update(w,y,means):
    ll=-.5*np.sum(((means-y)/SIGMA)**2,axis=1)
    with np.errstate(divide='ignore'):logw=np.log(w)+ll
    return np.exp(logw-logsumexp(logw))

def errors(q,qhat):
    return np.sqrt(np.mean(np.sum((q.reshape(len(q),-1,3)-qhat.reshape(-1,3))**2,axis=2),axis=1))

def risk(w,q):
    index=int(np.argmax(w));failure=errors(q,q[index])>TOLERANCE
    return float(w@failure),index

def choose(w,bank,initial_log_weights=None):
    # Common standard-normal draws across actions, fixed independently of truth.
    z=np.random.default_rng(110701).normal(size=(128,bank['action'].shape[-1]))
    q=bank['target'];failure=np.array([errors(q,row)>TOLERANCE for row in q])
    if initial_log_weights is None:
        with np.errstate(divide='ignore'):initial_log_weights=np.log(w)
    scores=[]
    for a in range(2):
        observations=bank['action'][:,a,None,:]+SIGMA*z[None,:,:]
        ll=-.5*np.sum(((observations[:,:,None,:]-bank['action'][None,None,:,a,:])/SIGMA)**2,axis=-1)
        logw=initial_log_weights[None,None,:]+ll
        wp=np.exp(logw-logsumexp(logw,axis=-1,keepdims=True))
        selected=np.argmax(wp,axis=-1)
        values=np.sum(wp*failure[selected],axis=-1)
        value=float(w@values.mean(1))
        scores.append(value)
    return int(np.argmin(scores)),scores

def score(w,bank,truth):
    probability,index=risk(w,bank['target']);q=bank['target'];qhat=q[index];mean=w@q
    err=float(errors(truth[None],qhat)[0]);meanerr=float(errors(truth[None],mean)[0])
    lo,hi=gaussian_mixture_quantiles(w,q,np.zeros_like(q),(.05,.95))
    return dict(map_index=index,predicted_failure_probability=probability,actual_failure=bool(err>TOLERANCE),
        map_rms_mm=1000*err,posterior_mean_rms_mm=1000*meanerr,
        brier=(probability-float(err>TOLERANCE))**2,coordinate_coverage90=float(np.mean((truth>=lo)&(truth<=hi))),
        coordinate_width90_mm=float(np.mean(hi-lo)*1000),effective_hypotheses=float(1/np.sum(w*w)),weights=w.tolist())

def verify_sources():
    protocol_path=REPORT/'protocol.json'
    if not protocol_path.exists():raise RuntimeError('Freeze protocol before generating test banks')
    frozen=json.loads(protocol_path.read_text())
    for path,expected in frozen['source_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:raise RuntimeError(f'Frozen source changed: {path}')

def protocol_hash():return hashlib.sha256((REPORT/'protocol.json').read_bytes()).hexdigest()

def seal_banks():
    verify_sources();path=REPORT/'bank-seal.json'
    if path.exists():raise RuntimeError('Bank seal already exists; do not overwrite')
    files=[ROOT/f'fem-n{n}-bank.npz' for n in (INFERENCE_N,REFERENCE_N)]
    path.write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))

def verify_banks():
    verify_sources()
    for path,expected in json.loads((REPORT/'bank-seal.json').read_text()).items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:raise RuntimeError(f'Frozen bank changed: {path}')

def model_banks():
    coarse=dict(np.load(ROOT/f'fem-n{INFERENCE_N}-bank.npz'))
    fine=dict(np.load(ROOT/f'fem-n{REFERENCE_N}-bank.npz'))
    mixture={key:np.concatenate([coarse[key],fine[key]],axis=0) for key in ('initial','action','target','reaction','theta')}
    # Explicit controlled comparator: both reference resolutions are candidate
    # forward models by construction. No claim of unknown-model coverage.
    return {'fem':coarse,'resolution-mixture':mixture}

def predict():
    verify_banks()
    if (REPORT/'prospective-predictions.json').exists():raise RuntimeError('Predictions already sealed; refuse overwrite')
    banks=model_banks()
    # Model prediction banks include every candidate's prospective response.
    # Only released observation arrays enter the selected-world data here.
    truths={}
    for label,n in [('matched',INFERENCE_N),('mesh-mismatch',REFERENCE_N)]:
        with np.load(ROOT/f'fem-n{n}-bank.npz') as archive:
            truths[label]={key:archive[key] for key in ('initial','action')}
    predictions=[]
    for cohort,truthbank in truths.items():
        for h in range(len(THETA)):
            for replicate in range(2):
                seed=1107100+(0 if cohort=='matched' else 1000)+10*h+replicate
                rng=np.random.default_rng(seed)
                y=truthbank['initial'][h]+SIGMA*rng.normal(size=truthbank['initial'][h].shape)
                ya=truthbank['action'][h]+SIGMA*rng.normal(size=truthbank['action'][h].shape)
                for family,bank in banks.items():
                    initial_log_weights=-np.log(len(bank['target']))-.5*np.sum(((bank['initial']-y)/SIGMA)**2,axis=1)
                    w=np.exp(initial_log_weights-logsumexp(initial_log_weights))
                    action,utilities=choose(w,bank,initial_log_weights)
                    if family=='fem':coarse_action=action
                    base=dict(cohort=cohort,hypothesis=h,replicate=replicate,seed=seed,model=family,
                              chosen_action=ACTIONS[action],pre_outcome_expected_risks=utilities)
                    stages=[('initial',None),('fixed_push',0),('fixed_tilt',1),('active',action)]
                    if family=='resolution-mixture':stages.append(('shared_coarse_active',coarse_action))
                    for stage,a in stages:
                        logw=initial_log_weights if a is None else initial_log_weights-.5*np.sum(((bank['action'][:,a]-ya[a])/SIGMA)**2,axis=1)
                        wp=np.exp(logw-logsumexp(logw))
                        probability,index=risk(wp,bank['target'])
                        record=dict(base,stage=stage,weights=wp.tolist(),predicted_failure_probability=probability,map_index=index)
                        force=bank['reaction'];force_lo,force_hi=gaussian_mixture_quantiles(wp,force,np.zeros_like(force),(.05,.95))
                        record.update(reaction_mean_N=(wp@force).tolist(),reaction_map_N=force[index].tolist(),reaction_lower90_N=force_lo.tolist(),reaction_upper90_N=force_hi.tolist())
                        record['initial_fit_residual_rms_mm']=float(np.sqrt(np.mean((bank['initial'][np.argmax(w)]-y)**2))*1000)
                        predictions.append(record)
    # Persist and hash every prediction before reading any held-out truth target.
    prospective=REPORT/'prospective-predictions.json'
    prospective.write_text(json.dumps(predictions,indent=2))
    (REPORT/'prediction-seal.json').write_text(json.dumps(dict(sha256=hashlib.sha256(prospective.read_bytes()).hexdigest(),created_unix=time.time(),protocol_sha256=hashlib.sha256((REPORT/'protocol.json').read_bytes()).hexdigest()),indent=2))

def score_saved():
    verify_banks()
    if (REPORT/'case-results.json').exists():raise RuntimeError('Outcomes already scored; refuse overwrite')
    prospective=REPORT/'prospective-predictions.json';seal=json.loads((REPORT/'prediction-seal.json').read_text())
    if hashlib.sha256(prospective.read_bytes()).hexdigest()!=seal['sha256']:raise RuntimeError('Sealed predictions changed')
    predictions=json.loads(prospective.read_text())
    banks=model_banks()
    truths={label:dict(np.load(ROOT/f'fem-n{n}-bank.npz')) for label,n in [('matched',INFERENCE_N),('mesh-mismatch',REFERENCE_N)]}
    results=[]
    for prediction in predictions:
        bank=banks[prediction['model']];truth=truths[prediction['cohort']]['target'][prediction['hypothesis']]
        outcome=dict(prediction,**score(np.asarray(prediction['weights']),bank,truth))
        force_truth=truths[prediction['cohort']]['reaction'][prediction['hypothesis']].reshape(3,2,3)
        for c,name in enumerate(('bowl','tool')):
            for estimator in ('mean','map'):
                difference=np.asarray(prediction[f'reaction_{estimator}_N']).reshape(3,2,3)[:,c]-force_truth[:,c]
                outcome[f'{name}_reaction_{estimator}_vector_rms_N']=float(np.sqrt(np.mean(np.sum(difference*difference,axis=-1))))
            lo=np.asarray(prediction['reaction_lower90_N']).reshape(3,2,3)[:,c]
            hi=np.asarray(prediction['reaction_upper90_N']).reshape(3,2,3)[:,c]
            outcome[f'{name}_reaction_coordinate_coverage90']=float(np.mean((force_truth[:,c]>=lo)&(force_truth[:,c]<=hi)))
            outcome[f'{name}_reaction_coordinate_width90_N']=float(np.mean(hi-lo))
        results.append(outcome)
    (REPORT/'case-results.json').write_text(json.dumps(results,indent=2))
    return results

def protocol():
    source=Path('scripts/gap_first_v01/bowl_campaign.py')
    record=dict(status='pre-outcome protocol',scope='Controlled finite-hypothesis nonlinear contact; supplied metric overhead surface tracks, not RGB reconstruction',
        hypotheses=THETA.tolist(),spring_label='Attempted unchanged PhysTwin spring-force core adapter failed development qualification; no accuracy ranking or claim about native full method',
        mesh=dict(inference_cells=INFERENCE_N,matched_truth_cells=INFERENCE_N,mismatch_truth_cells=REFERENCE_N,geometry='same piecewise-linear tapered ellipsoid; refinement preserves volume and boundary'),
        initial_times_s=[.08,.12],calibration_times_s=[.94,1.2],target_times_s=[1.1,1.2,1.3],
        marker_indices=MARKERS.tolist(),camera='orthographic overhead, image x/y metric coordinates; visibility audit required',
        noise_sigma_m=SIGMA,decision_tolerance_m=TOLERANCE,risk='posterior probability that the MAP-model 3D point RMS error exceeds tolerance',
        known_parameters=dict(density_kg_m3=1000,poisson_ratio=.3,viscosity_Pa_s=2,gravity_m_s2=[0,0,-9.81],bowl_radius_m=.09,bowl_center_m=[0,0,.09],pusher_radius_m=.018),
        target_metric='3D Euclidean point RMS over64 common material nodes andthree held-out times; includes simulation-only interior ground-truth nodes',
        secondary_force_metric='For bowl and tool separately: vector RMS in N across trailing20ms mean reactions at1.1/1.2/1.3s, accumulated from every timestep, posterior-mean andMAP predictions; marginal90% coordinate coverage/width. Sign: force exerted by soft object on rigid body. No force observations, no acquisition retuning using force outcomes.',
        friction_model='one unknown coefficient shared by bowl and pusher; Coulomb-capped viscous slip, no exact static friction',
        bowl_rotation='0.2rad about fixed spherical center; unchanged gravity, normal field below rim; informative tangential omega-cross-arm wall velocity',
        actions=list(ACTIONS),target_interaction='heldout_push: transverse y push, 45mm travel, after descending through open rim; calibration x push 25mm or 0.2rad bowl rotation',
        acquisition_budget='Exactly one reset calibration episode providing16 projected coordinates; initial/no-correction is a control. Fixed-budget action comparison, not proceed-or-acquire or energy/cost optimization.',
        control_inputs='Position-controlled pusher trajectory and prescribed bowl rotation; gravity known. Reaction forces are withheld outputs, never input measurements.',
        reset='Every action and held-out interaction starts at identical known initial rest state; calibration observations never come from target trajectory',
        inference_dt_s=DT,reference_dt_s=REFERENCE_DT,contact_penalty_N_per_m_per_reference_patch=PENALTY,reference_patch_area_m2=.0004,
        duration_s=1.3,cohorts={'matched':32,'mesh-mismatch':32},noise_replicates_per_material=2,
        independent_parameter_worlds=16,cohort_pairing='Same16 parameter worlds across two discretizations; independent predeclared noise seeds per cohort, not64 independent objects',
        secondary_resolution_mixture='32 hypotheses: P(resolution6)=P(resolution12)=0.5 and uniform16-parameter prior within each. Fine reference model lies in this mixture by construction. This is a controlled numerical-model-uncertainty comparator, not unknown-model calibration; primary16-coarse results remain separate.',
        secondary_shared_evidence='Resolution mixture also receives exactly the action observation chosen by the coarse-only policy (shared_coarse_active), in addition to its own policy and both fixed controls.',
        hard_deadline_utc='2026-11-07T21:44:00+00:00',checkpoint='Atomic restartable GPU state every approximately0.2 simulated seconds; source/parameter validation on resume',
        exclusions='None. Any solver/visibility failure retained and reported; no silent replacement. Required bank failure stops accuracy evaluation after attempting all hypotheses. Visibility failures preclude a camera-accessible claim, without filtering cases.',
        baselines=['MAP point prediction using exactly the same posterior/evidence','fixed gentle x-push','fixed small bowl rotation'],
        utilities='Expected posterior failure risk; all16 hypotheses,128 common Gaussian noise draws/hypothesis; seed110701',
        reporting_risk_threshold=.2,summary='RMS, Brier, selective risk at reporting threshold, coordinate90% intervals, paired active-minus-each-fixed; bootstrap clustered by16 material/contact worlds, not32 noise replicates',
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,source.with_name('bowl_physics.py'),source.with_name('bowl_native.py'),source.with_name('bowl_torch.py'),source.with_name('quantiles.py')]})
    path=REPORT/'protocol.json'
    if path.exists():raise RuntimeError('Protocol already exists; do not overwrite freeze')
    path.write_text(json.dumps(record,indent=2));print(path)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['protocol','generate','generate-gpu','seal','predict','score']);p.add_argument('--family',default='fem',choices=['fem','spring']);p.add_argument('--n',type=int,default=INFERENCE_N);p.add_argument('--interaction',choices=[*ACTIONS,'heldout_push']);a=p.parse_args()
    if a.mode=='protocol':protocol()
    elif a.mode=='generate':generate(a.family,a.n)
    elif a.mode=='generate-gpu':generate_gpu(a.n,a.interaction)
    elif a.mode=='seal':seal_banks()
    elif a.mode=='predict':predict()
    else:score_saved()
