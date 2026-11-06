"""Posthoc exact surface-traction diagnostic; not used to tune frozen inference."""
from pathlib import Path
import argparse,hashlib,json,datetime
import numpy as np
import torch
from . import surface,surface_campaign
from .mechanics import LinearGaussianBody
ROOT=surface.ROOT;R=ROOT/'reports/joint-field-surface-v04/traction';B=ROOT/'runs/joint-field-surface-v04'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def freeze():
    R.mkdir(parents=True,exist_ok=True);p=R/'protocol.json'
    assert not p.exists(),'Do not overwrite frozen diagnostic plan'
    sources=[Path(__file__),Path(surface.__file__),Path(surface_campaign.__file__),Path(__file__).with_name('mechanics.py')]
    plan=dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),label='Posthoc diagnostic frozen before examining field-error outcomes, not an original confirmatory endpoint',cases_per_shape=64,shapes=['block','step'],population='matched',stages=['initial','fixed','active'],budget=4,units='Pa=N/m^2',metric='sqrt(sum_t timeweight[t] integral_top+side ||mean traction-truth traction||^2 dA / (area_top+area_side))',time_weights=(np.array([.5,1,1,1,1,1,1,.5])/7).tolist(),quadrature='Same8x4x4surface triangles anddegree2quadrature asfrozenFEM; eachGaussian footprint normalizedbysurfaceintegral',dependence='Jointposterior mean sum_h p_h g_h(X) E[a_t|h,y] vs product sum_h p_hg_h(X) times sum_h p_h E[a_t|h,y], identicalevidence',aggregation='Mean per-case RMSE; paired4000bootstrap with64resamples per each fixedshape,seed193001. Bothmodearea weighting; nooutcome exclusions.',controls=['Unit footprint integrals','ExactFEMnodalforce-scatterparity','Totalposteriorforceintegrals recoveramplitude means everytime'],source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},bank_sha256={s:sha(B/f'{s}.npz') for s in ['block','step']})
    p.write_text(json.dumps(plan,indent=2));print('Frozen',p,flush=True)
def run():
    torch.set_num_threads(1);plan=json.loads((R/'protocol.json').read_text())
    for p,h in plan['source_sha256'].items():assert sha(ROOT/p)==h
    wt=np.array(plan['time_weights']);allrows=[];sources={};checks={}
    for shape in plan['shapes']:
        bp=B/f'{shape}.npz';assert sha(bp)==plan['bank_sha256'][shape];data=dict(np.load(bp));m=surface.maps(data);tp=B/shape/'matched-truth.npz';truth=dict(np.load(tp));sources[str(tp.relative_to(ROOT))]=sha(tp)
        body=LinearGaussianBody(shape,tuple(data['cells']));q=body.surface_points.numpy();area=body.surface_quadrature_weights.numpy();centers=np.unique(data['theta'][:,4]);group=np.searchsorted(centers,data['theta'][:,4]);profiles=[];areas=[];masks=[];maxscatter=0.
        for face in ['top','side']:
            mask=body._surface_masks[face].numpy();g=np.exp(-.5*((q[None,:,0]-centers[:,None])/body.footprint_width)**2)*mask[None];g/=np.sum(g*area,axis=1)[:,None]
            assert np.max(abs(g@area-1))<1e-12
            scatter=(g*area)@body.surface_interpolation.numpy();maxscatter=max(maxscatter,float(np.max(abs(scatter-body.footprint(centers,face).numpy()))));assert maxscatter<1e-12
            profiles.append(g);areas.append(float(area[mask].sum()));masks.append(mask)
        maxconservation=0.;shape_rows=[]
        for case in range(64):
            out=R/f'{shape}-case-{case:03d}.json';cp=ROOT/f'reports/joint-field-surface-v04/{shape}/matched/case-{case:03d}.json'
            if not cp.exists():raise RuntimeError(f'Missing case, do not silently skip: {cp}')
            doc=json.loads(cp.read_text());sources[str(cp.relative_to(ROOT))]=sha(cp);post={'initial':surface_campaign.initial(m,truth['y'][case])}
            for stage in ['fixed','active']:
                p=post['initial'];acts=[r['action'] for r in doc['rows'] if r['policy']==stage and r['step']>0];assert len(acts)==4
                for a in acts:p=surface_campaign.action_update(p,m,a,truth['measurements'][case,a])
                post[stage]=p
            truegroup=int(np.argmin(abs(centers-truth['theta'][case,4])));assert abs(centers[truegroup]-truth['theta'][case,4])<1e-12
            rows=[]
            for stage,p in post.items():
                grouped=np.array([np.sum(p.weights[group==j,None]*p.means[group==j],axis=0) for j in range(5)]).reshape(5,8,2);pg=np.bincount(group,weights=p.weights,minlength=5);amplitude=p.weights@p.means
                for estimator in ['joint','independent']:
                    errors=[]
                    for mode,g in enumerate(profiles):
                        pred=grouped[:,:,mode].T@g if estimator=='joint' else amplitude.reshape(8,2)[:,mode,None]*(pg@g)[None]
                        true=truth['force'][case].reshape(8,2)[:,mode,None]*g[truegroup]
                        conservation=float(np.max(abs(pred@area-amplitude.reshape(8,2)[:,mode])));maxconservation=max(maxconservation,conservation);assert conservation<1e-12
                        errors.append(float(np.sum(wt[:,None]*(pred-true)**2*area[None])))
                    row=dict(shape=shape,case=case,stage=stage,estimator=estimator,traction_rmse_Pa=float(np.sqrt(sum(errors)/sum(areas))),top_rmse_Pa=float(np.sqrt(errors[0]/areas[0])),side_rmse_Pa=float(np.sqrt(errors[1]/areas[1])))
                    rows.append(row);allrows.append(row);shape_rows.append(row)
            out.write_text(json.dumps(dict(protocol_sha256=sha(R/'protocol.json'),case_source_sha256=sha(cp),rows=rows),indent=2))
            if case%16==0:print(shape,case,'/64',flush=True)
        checks[shape]=dict(surface_areas_m2=dict(zip(['top','side'],areas)),max_nodal_scatter_error=maxscatter,max_total_force_integral_error_N=maxconservation)
    summary=[];paired=[];rng=np.random.default_rng(193001);idx=np.concatenate([rng.integers(64,size=(4000,64)),64+rng.integers(64,size=(4000,64))],axis=1)
    for stage in plan['stages']:
        vals={}
        for estimator in ['joint','independent']:
            rows=sorted([r for r in allrows if r['stage']==stage and r['estimator']==estimator],key=lambda r:(r['shape'],r['case']));assert len(rows)==128
            vals[estimator]=np.array([r['traction_rmse_Pa'] for r in rows]);summary.append(dict(stage=stage,estimator=estimator,n=128,**{k:float(np.mean([r[k] for r in rows])) for k in ['traction_rmse_Pa','top_rmse_Pa','side_rmse_Pa']}))
        delta=vals['joint']-vals['independent'];paired.append(dict(stage=stage,contrast='joint_minus_independent_Pa',mean=float(delta.mean()),ci95=np.quantile(delta[idx].mean(1),[.025,.975]).tolist()))
    result=dict(complete=True,cases=128,protocol_sha256=sha(R/'protocol.json'),summary=summary,paired=paired,checks=checks,sources=sources,scope=plan['label'])
    (R/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='sources'},indent=2),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--freeze',action='store_true');args=ap.parse_args();freeze() if args.freeze else run()
