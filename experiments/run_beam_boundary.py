"""Reproduce the Gaussian boundary benchmark, numerical audit, and HTML report."""
import argparse
from dataclasses import asdict
import gzip
import hashlib
import html
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from beam_boundary import (Boundary, candidates, response, camera, force_covariance,
    observation_design, observation_matrix, ExactBoundaryPosterior, uncertainty_metrics,
    mixture_quantile, simulation_audit, gaussian_density, normal_cdf)

METHODS=['learned_boundaries','known_boundaries','map_boundary_plugin','wrong_fixed_boundary']
COLORS=['#007f86','#278344','#d77b1c','#a63045']


def run_population(config, stress=False):
    n=config['stress_cases'] if stress else config['cases']
    rng=np.random.default_rng(config['seed']+(100000 if stress else 0))
    boundaries=candidates();c=len(boundaries);x=np.linspace(0,1,81)
    labels=rng.integers(c,size=n)
    off_grid = stress in ('off_grid','combined')
    if off_grid:
        truths=[Boundary(float(rng.uniform(.075,.225)),float(rng.uniform(.035,.06)),
            float(np.exp(rng.uniform(np.log(2000),np.log(10000)))),float(rng.uniform(.85,1))) for _ in range(n)]
        truth_response=np.stack([response(b,x,config['truth_elements']) for b in truths])
    else:
        truths=[boundaries[i] for i in labels]
        truth_bank=np.stack([response(b,x,config['truth_elements']) for b in boundaries])
        truth_response=truth_bank[labels]
    k=force_covariance(np.arange(1,9),length=.8 if stress in ('shorter_force','combined') else 2.)
    force=rng.normal(size=(n,8))@np.linalg.cholesky(k).T
    true_motion=force[:,:,None]*truth_response[:,None,:]
    spatial=np.stack([response(b,x,config['inference_elements']) for b in boundaries])
    # Generate shared noise keyed by (time, spatial coordinate, axis), so all
    # common observations are identical between sparse and richer designs.
    noise=rng.normal(scale=config['pixel_noise_std'],size=(n,8,81,2))
    hidden=(x<.30)|(x>.80)
    results={};examples={}
    dataset={'reference_x_m':x.tolist(),'times_s':list(range(1,9)),
        'boundaries':[asdict(b) for b in truths], 'force_N':force.tolist(),
        'unit_force_reference_response_m_per_N':truth_response.tolist(),'observations':{}}
    for design_name in ['sparse','richer']:
        design=observation_design(design_name)
        y=np.stack([camera(t+1)[0][axis,1]*true_motion[:,t,int(round(pos*80))]
                    +noise[:,t,int(round(pos*80)),axis] for t,pos,axis in design],axis=1)
        engine=ExactBoundaryPosterior(boundaries,design,config['pixel_noise_std'],config['inference_elements'])
        dataset['observations'][design_name]={'design':design,'displacements_pixels':y.tolist()}
        weights,means=engine.fit(y)
        all_mean=np.einsum('nc,nct,cx->ntx',weights,means,spatial)
        map_index=weights.argmax(axis=1)
        rows=np.arange(n)
        methods={
            'learned_boundaries':weights,
            'map_boundary_plugin':np.eye(c)[map_index],
            'wrong_fixed_boundary':np.tile(np.eye(c)[0],(n,1))}
        # The matched oracle knows boundary geometry/compliance, but still has
        # to estimate force amplitudes from exactly the same noisy tracks.
        if not off_grid:methods['known_boundaries']=np.eye(c)[labels]
        metrics={}
        for name in METHODS:
            if name not in methods:continue
            w=methods[name]
            pred=np.einsum('nc,nct,cx->ntx',w,means,spatial)
            force_pred=np.einsum('nc,nct->nt',w,means)
            errors=np.sqrt(np.mean((pred[:,:,hidden]-true_motion[:,:,hidden])**2,axis=(1,2)))*1000
            observed_frames={t for t,_,_ in design};missing=[t for t in range(8) if t not in observed_frames]
            qmean=means[:,:,4]*spatial[None,:,80]
            qvar=engine.cov[:,4,4][None,:]*spatial[None,:,80]**2
            uncertainty=uncertainty_metrics(true_motion[:,4,80],w,qmean,qvar)
            force_uncertainty=uncertainty_metrics(force[:,4],w,means[:,:,4],engine.cov[:,4,4][None,:])
            se=np.std(errors,ddof=1)/np.sqrt(n)
            metrics[name]={'hidden_motion_rmse_mm':float(errors.mean()),
                'rmse_mean_mc_se_mm':float(se),
                'missing_time_hidden_rmse_mm':float(np.mean(np.sqrt(np.mean(
                    (pred[:,missing][:,:,hidden]-true_motion[:,missing][:,:,hidden])**2,axis=(1,2))))*1000),
                'force_rmse_N':float(np.mean(np.sqrt(np.mean((force_pred-force)**2,axis=1)))),
                'hidden_tip_at_5s':uncertainty,'force_at_5s':force_uncertainty}
        sorted_p=np.sort(weights,axis=1)[:,::-1]
        counts=(np.cumsum(sorted_p,axis=1)<.9).sum(axis=1)+1
        support_mu=weights@np.array([b.support_center for b in boundaries])
        load_mu=weights@np.array([b.load_center for b in boundaries])
        classification={'support_center_mae_mm':float(np.mean(abs(support_mu-np.array([b.support_center for b in truths])))*1000),
            'load_center_mae_mm':float(np.mean(abs(load_mu-np.array([b.load_center for b in truths])))*1000),
            'mean_boundary_entropy_nats':float(np.mean(-np.sum(weights*np.log(np.maximum(weights,1e-300)),axis=1))),
            'mean_90pct_credible_set_size':float(counts.mean()),
            'hypotheses':c}
        if not off_grid:
            order=np.argsort(weights,axis=1)[:,::-1]
            rank=np.argmax(order==labels[:,None],axis=1)
            classification['joint_boundary_map_accuracy']=float(np.mean(map_index==labels))
            classification['joint_boundary_credible_set_coverage']=float(np.mean(rank<counts))
            classification['joint_boundary_credible_set_mean_mass']=float(np.mean(np.take_along_axis(
                np.cumsum(sorted_p,axis=1),(counts-1)[:,None],axis=1)))
        results[design_name]={'observed_frames':len(observed_frames),'tracked_centers_per_frame':len(design)//(2*len(observed_frames)),
                              'methods':metrics,'boundary_identification':classification}
        # Predeclared first case, not selected on appearance or posterior quality.
        i=0
        qmeans=means[i,:,4,None]*spatial
        qvars=engine.cov[:,4,4,None]*spatial**2
        ww=np.broadcast_to(weights[i],(len(x),c))
        lo=mixture_quantile(.05,ww,qmeans.T,qvars.T)
        hi=mixture_quantile(.95,ww,qmeans.T,qvars.T)
        examples[design_name]={'x':x,'truth':true_motion[i], 'mean':all_mean[i],
            'lo':lo,'hi':hi,'weights':weights[i],'boundary':truths[i],
            'force':force[i],'y':y[i],'design':design,'means':means[i],
            'cov':engine.cov,'truth_response':truth_response[i]}
    max_slope=float(np.max(abs(np.diff(true_motion,axis=2)/np.diff(x)[None,None,:])))
    return {'cases':n,'stress':stress,'max_absolute_slope_approx_rad':max_slope,
            'max_absolute_displacement_m':float(np.max(abs(true_motion))),
            'designs':results},examples,dataset


def plots(output,results,examples):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(14,4.5))
    ex=examples['sparse'];x=ex['x'];b=ex['boundary']
    axes[0].fill_between(x,ex['lo']*1000,ex['hi']*1000,color=COLORS[0],alpha=.22,label='90% pointwise posterior')
    axes[0].plot(x,ex['truth'][4]*1000,'k',label='Reference deformation')
    axes[0].plot(x,ex['mean'][4]*1000,color=COLORS[0],label='Estimated mean')
    axes[0].axvspan(.35,.75,color='#888888',alpha=.13,label='Observable region')
    axes[0].set(title='First case: missing frame at 5 s',xlabel='Reference position (m)',ylabel='Displacement (mm)');axes[0].legend(fontsize=8)
    for j,key in enumerate(['support_center','load_center']):
        vals=sorted(set(getattr(v,key) for v in candidates()))
        prob=[sum(p for p,v in zip(ex['weights'],candidates()) if getattr(v,key)==value) for value in vals]
        axes[j+1].bar(np.arange(len(vals)),prob,color=COLORS[0])
        axes[j+1].set_xticks(np.arange(len(vals)),[str(v) for v in vals])
        axes[j+1].set(title=key.replace('_',' ').title()+f' (true {getattr(b,key):g} m)',xlabel='Candidate position (m)',ylabel='Posterior probability',ylim=(0,1))
    fig.tight_layout();fig.savefig(output/'first-case.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4.5))
    for k,design in enumerate(['sparse','richer']):
        d=results['designs'][design]['methods']
        for j,name in enumerate(METHODS):
            levels=[.5,.8,.9,.95]
            axes[k].plot(levels,[d[name]['hidden_tip_at_5s'][str(l)]['coverage'] for l in levels],
                         'o-',color=COLORS[j],label=name.replace('_',' '))
        axes[k].plot([.45,1],[.45,1],'k--',linewidth=1)
        axes[k].set(title=f'{design.title()}: hidden tip at missing time',xlabel='Nominal coverage',ylabel='Measured coverage',ylim=(0,1.02))
    axes[0].legend(fontsize=7)
    for j,name in enumerate(METHODS):
        axes[2].bar(np.arange(2)+(j-1.5)*.18,[results['designs'][d]['methods'][name]['hidden_motion_rmse_mm'] for d in ['sparse','richer']],width=.17,color=COLORS[j])
    axes[2].set_xticks([0,1],['Sparse','Richer']);axes[2].set(title='Hidden-region reconstruction',ylabel='Mean per-case RMSE (mm)')
    fig.tight_layout();fig.savefig(output/'comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(10,5),sharex=True)
    gx=np.linspace(0,1,501)
    w=np.exp(-.5*((gx[:,None]-np.array([0,.5,1]))/.3)**2);w/=w.sum(axis=1,keepdims=True)
    for j in range(3):axes[0].plot(gx,w[:,j],label=f'Material kernel {j+1}')
    axes[0].set(ylabel='Normalized material weight',title='Two Gaussian roles in reference coordinates');axes[0].legend(fontsize=8,ncol=3)
    for center,width,label,color in [(b.support_center,b.support_width,'Support footprint','#a63045'),(b.load_center,b.load_width,'Load footprint','#278344')]:
        axes[1].plot(gx,gaussian_density(gx,center,width),label=label,color=color)
    axes[1].axvspan(.35,.75,color='gray',alpha=.15,label='Observable region')
    axes[1].set(xlabel='Reference position (m)',ylabel='Boundary density (1/m)');axes[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(output/'gaussian-fields.png',dpi=150);plt.close(fig)


def render_views(output,example):
    """Depth-sorted alpha-composited Gaussian reference; illustrations only."""
    x=np.repeat(np.linspace(0,1,101),3)
    offsets=np.tile(np.array([-.005,0,.005]),101)
    width,height=500,160
    yy,xx=np.mgrid[0:height,0:width]
    fig,axes=plt.subplots(2,3,figsize=(12,4.7))
    for col,t in enumerate([0,3,7]):
        displacement=np.interp(x,example['x'],example['truth'][t])
        xyz=np.stack([x-.5,displacement+offsets,np.zeros_like(x)],axis=1)
        projection,normal=camera(t+1)
        uv=xyz@projection.T+np.array([width/2,height/2])
        depth=xyz@normal
        for row,masked in enumerate([True,False]):
            img=np.ones((height,width,3))*.96
            # normal points toward the camera: smaller dot product is farther.
            for i in np.argsort(depth):
                if masked and not (.35<=x[i]<=.75):continue
                alpha=.8*np.exp(-.5*(((xx-uv[i,0])/2.4)**2+((yy-uv[i,1])/2.4)**2))
                color=np.array([.10+.55*x[i],.47,.70-.35*x[i]])
                img=img*(1-alpha[...,None])+color*alpha[...,None]
            axes[row,col].imshow(img);axes[row,col].axis('off')
            axes[row,col].set_title(f'{"Masked input illustration" if masked else "Withheld full beam"}, t={t+1} s',fontsize=9)
    fig.suptitle('One moving camera; artificial visibility mask; known centre tracks used for inference',fontsize=11)
    fig.tight_layout();fig.savefig(output/'views.png',dpi=150);plt.close(fig)


def publish(output,config,results,stress_suite,audit,examples,elapsed):
    primary=[]
    for design in ['sparse','richer']:
        for method in ['learned_boundaries','known_boundaries']:
            for quantity in ['hidden_tip_at_5s','force_at_5s']:
                u=results['designs'][design]['methods'][method][quantity]
                primary.append(abs(u['0.9']['coverage']-.9)<=3*np.sqrt(.9*.1/config['cases'])
                               and u['pit_ks']<=u['pit_ks_99pct_reference'])
    report={'protocol':config,'matched_model':results,'stress_tests':stress_suite,
        'simulation_audit':audit,'prior_predictive_scalar_calibration_gate_pass':bool(all(primary)),
        'uncertainty_validated_under_model_mismatch':False,
        'elapsed_seconds':elapsed,'limitations':[
            'Known material, reference geometry, camera poses, masks and point correspondences.',
            'One-plane quasistatic Euler-Bernoulli bending; no inertia, contact switching, friction or 3D occlusion inference.',
            'Support is a Gaussian distributed elastic foundation; no hard-clamp discovery.',
            'Exact inference on a finite boundary dictionary; continuous/off-grid robustness checked separately.',
            'Pointwise prior-predictive coverage is not simultaneous trajectory coverage or universal real-data calibration.',
            'No encoder or conditional flow-matching model has been trained; this is its proposed reference benchmark.']}
    (output/'results.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    ex=examples['sparse']
    (output/'first-case.json').write_text(json.dumps({'selection':'first draw from predeclared seed',
        'boundary':asdict(ex['boundary']),'true_force_N':ex['force'].tolist(),
        'candidates':[asdict(b) for b in candidates()],
        'posterior_probabilities':ex['weights'].tolist()},indent=2)+'\n',encoding='utf-8')
    paths=['experiments/beam_boundary.py','experiments/run_beam_boundary.py','configs/beam_boundary.json']
    manifest={'python':platform.python_version(),'numpy':np.__version__,'matplotlib':matplotlib.__version__,
        'source_sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
        'dataset_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob('*.json.gz')},
        'base_git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'execution':'GitHub Actions CPU' if os.environ.get('GITHUB_ACTIONS')=='true' else 'local CPU',
        'workflow_run_id':os.environ.get('GITHUB_RUN_ID'), 'elapsed_seconds':elapsed}
    (output/'execution.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    rows=''
    for design in ['sparse','richer']:
        for method in METHODS:
            m=results['designs'][design]['methods'][method];u=m['hidden_tip_at_5s']['0.9']
            rows+=f'<tr><td>{design}</td><td>{method}</td><td>{m["hidden_motion_rmse_mm"]:.3f}</td><td>{100*u["coverage"]:.1f}%</td><td>{1000*u["mean_width"]:.2f}</td></tr>'
    stress_rows=''
    for stress_name,stress in stress_suite.items():
        for design in ['sparse','richer']:
            m=stress['designs'][design]['methods']['learned_boundaries']
            stress_rows+=f'<p>{stress_name}, {design}: hidden-motion RMSE {m["hidden_motion_rmse_mm"]:.3f} mm; 90% tip coverage {100*m["hidden_tip_at_5s"]["0.9"]["coverage"]:.1f}%.</p>'
    content=f'''<!doctype html><html><head><meta charset="utf-8"><title>Gaussian boundary beam benchmark</title>
<style>body{{max-width:1120px;margin:40px auto;padding:0 24px;font:17px/1.6 system-ui;color:#182b39;background:#fafaf7}}img{{width:100%;border:1px solid #ddd}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:9px;text-align:left;border-bottom:1px solid #ccc}}.note{{padding:18px;background:#e7f0ef}}a{{color:#007f86}}</style></head><body>
<h1>Learning hidden beam boundaries</h1><p>A static material model plus Gaussian support and load patches, viewed by one sparse moving camera.</p>
<div class="note"><strong>Scope:</strong> This first benchmark uses noisy 2D tracks with supplied correspondences. Images below illustrate the synthetic setup; no image encoder is trained. Material rigidity is known.</div>
<h2>Protocol</h2><p>{config['cases']} independent matched-prior cases, 24 boundary hypotheses, 80-element reference versus 40-element inference. Sparse: 3 frames × 3 tracks. Richer: 5 frames × 9 tracks. The support and load centres are outside the visible interval. Unknown Gaussian force histories are integrated analytically; all boundary hypotheses are retained.</p>
<img src="views.png" alt="Sparse masked views and withheld full-beam reference">
<img src="gaussian-fields.png" alt="Material Gaussian weights and boundary Gaussian support and load footprints">
<h2>Comparison</h2><p>Known boundaries supplies the true support and load geometry/compliance but estimates force amplitudes from the same tracks. Wrong fixed boundary always uses dictionary entry zero. MAP plug-in drops uncertainty over the boundary choice. The fully known boundary-and-force oracle reproduces the reference by construction and is not ranked.</p>
<table><tr><th>Views</th><th>Method</th><th>Hidden RMSE (mm)</th><th>90% tip coverage</th><th>Interval width (mm)</th></tr>{rows}</table>
<img src="comparison.png" alt="Coverage curves and hidden deformation errors">
<h2>First case, selected before inference</h2><img src="first-case.png" alt="Shape posterior and boundary location probabilities">
<p>Exact joint boundary MAP accuracy is {100*results['designs']['sparse']['boundary_identification']['joint_boundary_map_accuracy']:.1f}% with sparse observations and {100*results['designs']['richer']['boundary_identification']['joint_boundary_map_accuracy']:.1f}% with richer observations. Accurate hidden motion does not imply unique support or load identification. Posterior uncertainty over alternative boundaries is essential.</p>
<h2>Uncertainty validation</h2><p>Predeclared scalar calibration gate: <strong>{'PASS' if all(primary) else 'FAIL'}</strong>. The gate checks missing-time tip displacement and force in both designs, for learned and known boundaries. It uses 90% coverage within three binomial standard errors and a 99% reference PIT KS threshold. Coverage is across independent prior-predictive cases, not a claim about every individual boundary configuration.</p>
<h2>Model-mismatch stress tests</h2><p>{config['stress_cases']} cases per arm test continuous off-grid boundaries, a shorter true force correlation time (0.8 s versus the assumed 2 s), and both together, without refitting. The separate arms were added after the initial combined stress test exposed undercoverage.</p>{stress_rows}
<h2>Mechanics audit</h2><p>Mesh refinement, total applied force, reaction force/moment balance, and energy checks: <strong>{'PASS' if audit['pass'] else 'FAIL'}</strong>. Maximum matched-case approximate slope: {results['max_absolute_slope_approx_rad']:.3f} rad. Full values are in the machine-readable report.</p>
<h2>Reproduce and inspect</h2><p><a href="results.json">All metrics and limitations</a> · <a href="execution.json">Execution provenance</a> · <a href="first-case.json">Example boundary probabilities</a></p>
<p>Run: <code>python experiments/run_beam_boundary.py --config configs/beam_boundary.json --output examples/beam-boundary</code></p>
<p>Compressed JSON datasets contain all generated boundary parameters, true forces, reference responses and noisy observations. These evaluation truths are not inputs to the inferred-boundary estimator.</p>
</body></html>'''
    (output/'report.html').write_text(content,encoding='utf-8')
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/beam_boundary.json');p.add_argument('--output',default='examples/beam-boundary')
    args=p.parse_args();config=json.loads(Path(args.config).read_text());output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    # This is a frozen reference protocol: reject changes to fields that are
    # declared for provenance but not exposed as tunable experiment switches.
    fixed={'material_rigidity_N_m2':.8,'length_m':1.,'reference_camera_scale_pixels_per_metre':400,
           'force_prior_std_N':.06,'force_prior_length_s':2.,'load_width_m':.035,
           'support_centers_m':[.075,.15,.225],'support_widths_m':[.035,.06],
           'support_total_stiffness_N_per_m':[2000,10000],'load_centers_m':[.85,1.]}
    for key,value in fixed.items():
        if config[key]!=value:raise ValueError(f'{key} differs from frozen benchmark implementation')
    for kind in ['sparse','richer']:
        design=observation_design(kind)
        if config['designs'][kind]['times_s']!=sorted({t+1 for t,_,_ in design}):raise ValueError('Design time mismatch')
        if not np.allclose(config['designs'][kind]['visible_x_m'],sorted({x for _,x,_ in design})):raise ValueError('Design position mismatch')
    start=time.perf_counter();audit=simulation_audit(candidates())
    if not audit['pass']:raise RuntimeError(json.dumps(audit))
    print('Mechanics audit passed. Running matched-prior cases.',flush=True)
    results,examples,data=run_population(config)
    (output/'matched-dataset.json.gz').write_bytes(gzip.compress(json.dumps(data,separators=(',',':')).encode(),mtime=0))
    print('Matched-prior cases complete. Running declared mismatch stress cases.',flush=True)
    stress_suite={}
    for kind in ['off_grid','shorter_force','combined']:
        stress_suite[kind],_,data=run_population(config,kind)
        (output/f'{kind}-dataset.json.gz').write_bytes(gzip.compress(json.dumps(data,separators=(',',':')).encode(),mtime=0))
        print(f'{kind} stress complete.',flush=True)
    plots(output,results,examples);render_views(output,examples['sparse'])
    report=publish(output,config,results,stress_suite,audit,examples,time.perf_counter()-start)
    print(json.dumps({'calibration_gate':report['prior_predictive_scalar_calibration_gate_pass'],
        'elapsed_seconds':report['elapsed_seconds'],'output':str(output)},indent=2),flush=True)


if __name__=='__main__':main()
