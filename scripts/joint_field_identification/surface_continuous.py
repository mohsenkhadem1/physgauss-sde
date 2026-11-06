"""Surface-visible continuous-posterior diagnostic using only recorded evidence. Explicit adaptation of frozen v01 continuous_campaign; core SMC unchanged.

No continuous action selection is claimed: active/fixed actions are the already
recorded finite-dictionary policy actions. This tests whether their posterior
interpretation persists when nonlinear physical parameters become continuous.
Final test jobs require an explicit frozen JSON manifest of allowed job keys.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
import numpy as np
import torch
from . import surface as bank
from .mechanics import LinearGaussianBody
from .continuous import SMCConfig,run_smc,MarginalGaussianLikelihood
from .surface_campaign import metrics

BASE=bank.ROOT/'runs/joint-field-surface-v04'
REPORT=bank.ROOT/'reports/joint-field-surface-v04/continuous'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def job_key(shape,population,case,stage,particles,seed,cells):
    return dict(shape=shape,population=population,case=int(case),stage=stage,
                particles=int(particles),seed=int(seed),cells=list(cells))


class FEMMapping:
    """Keep body/interpolation alive; compute only observation maps in SMC."""
    def __init__(self,shape,cells,device,actions=(),batch=16,deadline=None):
        self.body=LinearGaussianBody(shape,cells=cells,device=device)
        self.points=bank.points();self.interpolation=self.body.interpolation(self.points)
        self.actions=list(map(int,actions));self.batch=int(batch)
        self.calls=0;self.states=0;self.seconds=0.
        self.deadline=deadline

    def data(self,theta):
        units=[];probes=[];materials=[];start=time.perf_counter()
        for i in range(0,len(theta),self.batch):
            if self.deadline is not None and time.time()>=self.deadline:
                raise TimeoutError('Campaign deadline reached; last completed SMC stage remains checkpointed')
            th=torch.as_tensor(theta[i:i+self.batch],dtype=torch.float64,device=self.body.device)
            with torch.no_grad():
                r=self.body.solve(th,probe_specs=bank.probe_specs())
                units.append(torch.einsum('pn,bncl->bpcl',self.interpolation,r['unit_response']).cpu().numpy())
                probes.append(torch.einsum('pn,bncl->bpcl',self.interpolation,r['probe_response']).cpu().numpy())
                materials.append(self.body.modulus_at_points(th,self.points).cpu().numpy())
        self.calls+=1;self.states+=len(theta);self.seconds+=time.perf_counter()-start
        return dict(theta=np.asarray(theta),unit=np.concatenate(units),probe=np.concatenate(probes),
                    material=np.concatenate(materials),points=self.points,cells=np.asarray(self.body.cells))

    @staticmethod
    def observation_map(data,actions):
        return bank.map_observations(data,actions)

    def __call__(self,theta):return self.observation_map(self.data(theta),self.actions)

    def parity(self):
        th=bank.BOUNDS.mean(1)[None]
        data=self.data(th);m=bank.maps(data);calculated=self.observation_map(data,self.actions)
        H=[m['H']];offset=[np.zeros(m['H'].shape[:2])]
        for action in self.actions:
            a=m['actions'][action];H.append(a['h'][:,None]);offset.append(a['offset'][:,None])
        a=float(np.max(np.abs(calculated['H']-np.concatenate(H,1))))
        b=float(np.max(np.abs(calculated['offset']-np.concatenate(offset,1))))
        if a>1e-14 or b>1e-14:raise AssertionError('Observation assembly differs from bank.maps')
        return dict(max_H_error=a,max_offset_error=b)


def acquired_evidence(shape,population,case,stage):
    path=BASE/shape/('development.npz' if population=='development' else f'{population}-truth.npz')
    # Only y and selected recorded action measurements leave this input loader.
    with np.load(path) as archive:
        y=archive['y'][case].copy();actionvalues=archive['measurements'][case]
        actions=[];source=None
        if stage!='initial':
            if population=='development':raise ValueError('Development pilot only evaluates initial posterior')
            source=bank.ROOT/f'reports/joint-field-surface-v04/{shape}/{population}/case-{case:03d}.json'
            doc=json.loads(source.read_text())
            rows=sorted([r for r in doc['rows'] if r['policy']==stage and r['step']>0],key=lambda r:r['step'])
            if not rows:raise ValueError('Requested completed acquisition records are absent')
            if [r['step'] for r in rows]!=list(range(1,len(rows)+1)):raise ValueError('Incomplete action history')
            actions=[int(r['action']) for r in rows]
            if len(set(actions))!=len(actions):raise ValueError('Unexpected repeated action')
            y=np.r_[y,actionvalues[actions]]
    return y,actions,path,source


def run_job(job,*,device='cuda:0',batch=16,moves=4,resume=False,manifest=None,
            stop_after_stages=None,pilot=False):
    deadline=None
    if pilot:
        expected=job_key('block','development',0,'initial',128,job['seed'],job['cells'])
        if job!=expected:raise ValueError('Unfrozen pilot restricted to development block case0 initial128')
    else:
        if manifest is None:raise ValueError('Final jobs require a frozen manifest')
        manifest=Path(manifest);definition=json.loads(manifest.read_text())
        if definition.get('deadline_utc'):
            deadline=datetime.fromisoformat(definition['deadline_utc'].replace('Z','+00:00')).timestamp()
            if time.time()>=deadline:raise TimeoutError('Campaign deadline already passed')
        if job not in definition.get('allowed_jobs',[]):raise ValueError('Job not predeclared in manifest')
        if definition.get('moves',moves)!=moves:raise ValueError('MH budget differs from frozen manifest')
        if definition.get('bounds') is not None and not np.array_equal(definition['bounds'],bank.BOUNDS):
            raise ValueError('Frozen bounds changed')
        for name,expected in definition.get('source_sha256',{}).items():
            if sha(bank.ROOT/name)!=expected:raise ValueError(f'Frozen source changed: {name}')
    y,actions,data_path,record_path=acquired_evidence(job['shape'],job['population'],job['case'],job['stage'])
    inputs=dict(job=job,actions=actions,observations=y.tolist(),moves=moves,bounds=bank.BOUNDS.tolist(),
                load_covariance=bank.load_cov().tolist(),noise_m=bank.NOISE,
                data_sha256=sha(data_path),record_sha256=None if record_path is None else sha(record_path),
                manifest_sha256=None if manifest is None else sha(manifest),
                source_sha256={str(p.relative_to(bank.ROOT)):sha(p) for p in
                    [Path(__file__),Path(bank.__file__),bank.ROOT/'scripts/joint_field_identification/mechanics.py',
                     bank.ROOT/'scripts/joint_field_identification/continuous.py',
                     bank.ROOT/'scripts/joint_field_identification/inference.py',
                     bank.ROOT/'scripts/joint_field_identification/surface_campaign.py']})
    fingerprint=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest()
    name=f"{job['shape']}-{job['population']}-case{job['case']:03d}-{job['stage']}-n{job['particles']}-s{job['seed']}"
    root=BASE/'continuous'/name;root.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    reportpath=REPORT/(name+'.json');checkpoint=root/'state.npz';started=time.perf_counter()
    if reportpath.exists():
        old=json.loads(reportpath.read_text())
        if old.get('target_id')!=fingerprint:raise ValueError('Completed output fingerprint differs; use a new run namespace')
        if old.get('complete'):return old
    (root/'inputs.json').write_text(json.dumps(inputs,indent=2))
    mapping=FEMMapping(job['shape'],tuple(job['cells']),device,actions,batch,deadline)
    parity=mapping.parity();likelihood=MarginalGaussianLikelihood(mapping,y,bank.load_cov(),bank.NOISE**2)
    config=SMCConfig(particles=job['particles'],moves=moves,batch_size=batch)
    result=run_smc(likelihood,bounds=bank.BOUNDS,config=config,seed=job['seed'],target_id=fingerprint,
                   checkpoint=checkpoint,resume=resume and checkpoint.exists(),stop_after_stages=stop_after_stages)
    doc=dict(**result.report(),job=job,actions=actions,device=device,pilot=pilot,parity=parity,
             protocol=inputs,checkpoint=str(checkpoint.relative_to(bank.ROOT)))
    if result.complete:
        finaldata=mapping.data(result.theta);m=bank.maps(finaldata)
        saved_mapping=FEMMapping.observation_map(finaldata,actions)
        final_likelihood=MarginalGaussianLikelihood(lambda unused:saved_mapping,y,bank.load_cov(),bank.NOISE**2)
        means,covs=final_likelihood.conditional(result.theta)
        from .inference import Posterior
        p=Posterior(result.weights,means,covs,result.log_evidence)
        predictions=p.target(m['T'],m['offset'])
        # Ground truth enters only the following evaluator and never the above
        # likelihood, SMC transitions or finite-policy evidence construction.
        with np.load(data_path) as archive:truth={k:archive[k] for k in archive.files}
        scored=metrics(p,finaldata,m,truth,job['case'],intervals=True)
        knots=np.linspace(.05,.95,19)
        cdf=np.einsum('n,ndk->dk',result.weights,result.unit_particles[:,:,None]<=knots)
        quantiles=np.quantile(result.theta,[.05,.5,.95],axis=0)  # final particles have equal weight
        force=p.target(np.eye(16));material_mean=p.weights@finaldata['material']
        np.savez(root/'posterior-predictions.npz',theta=result.theta,weights=result.weights,
                 force_means=means,force_covariances=covs,target_mean=predictions.mean,
                 target_variance=predictions.variance,material_mean=material_mean,
                 force_mean=force.mean,force_variance=force.variance,cdf_knots=knots,cdf=cdf)
        doc.update(metrics=scored,parameter_quantiles05_50_95=quantiles.tolist(),cdf_knots=knots.tolist(),
                   parameter_unit_cdf=cdf.tolist(),prediction_archive=str((root/'posterior-predictions.npz').relative_to(bank.ROOT)),
                   interpretation='Continuous posterior for the same acquired evidence; actions were selected by the finite policy, not this SMC.')
    doc.update(invocation_seconds=time.perf_counter()-started,fem_states_this_invocation=mapping.states,
               fem_seconds_this_invocation=mapping.seconds)
    reportpath.write_text(json.dumps(doc,indent=2))
    print(json.dumps({k:doc.get(k) for k in ['job','complete','invocation_seconds','fem_states_this_invocation','log_evidence','metrics']},indent=2),flush=True)
    return doc


def compare(paths):
    docs=[json.loads(Path(p).read_text()) for p in paths]
    if not all(d['complete'] for d in docs):raise ValueError('Cannot compare incomplete posteriors')
    first=docs[0]['job'];identity=['shape','population','case','stage','cells']
    if any(any(d['job'][k]!=first[k] for k in identity) for d in docs):raise ValueError('Comparisons must share data/model')
    pairs=[]
    for i in range(len(docs)):
        for j in range(i+1,len(docs)):
            a,b=docs[i],docs[j]
            with np.load(bank.ROOT/a['prediction_archive']) as aa,np.load(bank.ROOT/b['prediction_archive']) as bb:
                pairs.append(dict(first=str(paths[i]),second=str(paths[j]),
                    max_unit_cdf_difference=float(np.max(np.abs(aa['cdf']-bb['cdf']))),
                    target_mean_rms_difference_mm=float(np.sqrt(np.mean((aa['target_mean']-bb['target_mean'])**2))*1000),
                    target_sd_rms_difference_mm=float(np.sqrt(np.mean((np.sqrt(aa['target_variance'])-np.sqrt(bb['target_variance']))**2))*1000),
                    material_mean_relative_rms_difference=float(np.sqrt(np.mean((aa['material_mean']-bb['material_mean'])**2)/np.mean(aa['material_mean']**2))),
                    force_mean_rms_difference_N=float(np.sqrt(np.mean((aa['force_mean']-bb['force_mean'])**2))),
                    log_evidence_difference=float(a['log_evidence']-b['log_evidence'])))
    return dict(comparisons=pairs,scope='Independent seed/count numerical differences, not an automatic convergence or calibration certificate.')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--shape',choices=['block','step'],default='block')
    ap.add_argument('--population',choices=['development','matched','offgrid-fine','force-prior','continuous-coarse'],default='development')
    ap.add_argument('--case',type=int,default=0);ap.add_argument('--stage',choices=['initial','active','fixed'],default='initial')
    ap.add_argument('--particles',type=int,default=128);ap.add_argument('--seed',type=int,default=171001)
    ap.add_argument('--cells',nargs=3,type=int,default=[8,4,4]);ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--batch',type=int,default=16);ap.add_argument('--moves',type=int,default=4)
    ap.add_argument('--pilot',action='store_true');ap.add_argument('--resume',action='store_true')
    ap.add_argument('--manifest');ap.add_argument('--stop-after-stages',type=int)
    ap.add_argument('--compare',nargs='+');ap.add_argument('--comparison-output')
    args=ap.parse_args();torch.set_num_threads(1)
    if args.compare:
        result=compare(args.compare)
        if args.comparison_output:Path(args.comparison_output).write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2));return
    job=job_key(args.shape,args.population,args.case,args.stage,args.particles,args.seed,args.cells)
    run_job(job,device=args.device,batch=args.batch,moves=args.moves,resume=args.resume,
            manifest=args.manifest,stop_after_stages=args.stop_after_stages,pilot=args.pilot)


if __name__=='__main__':main()
