"""Independent artifact audit; never invokes the predictor, selector or scorer.

Saved-frame determinants can check saved geometry, not every integration substep.
Integrated penetration/J diagnostics are checked against qualification thresholds;
this audit cannot independently reconstruct unsaved substeps.
"""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'runs/coupled-contact-v13'
REPORT=ROOT/'reports/coupled-contact-v13'
METHODS=('fem','native','rigid','position_ridge','position_rigid','persistence','oracle')

def read(path):return json.loads(path.read_text())
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def arrays(path):
    with np.load(path) as a:return {k:a[k].copy() for k in a.files}

def geometry_metrics(pred,target):
    if pred is None or pred.shape!=target.shape or not np.isfinite(pred).all() or not np.isfinite(target).all():return None
    shape=[]
    for a,b in zip(pred[1:],target[1:]):
        a=a-a.mean(0);b=b-b.mean(0)
        u,_,v=np.linalg.svd(a.T@b);sign=np.diag([1.,1.,np.linalg.det(u@v)])
        shape.append(np.mean((a@u@sign@v-b)**2))
    return {'coordinate_rmse_m':float(np.sqrt(np.mean((pred[1:]-target[1:])**2))),
            'shape_rmse_m':float(np.sqrt(np.mean(shape)))}

def minimum_j(frames,rest,tets):
    if tets.ndim!=2 or tets.shape[1]!=4 or np.any(tets<0) or np.any(tets>=len(rest)):raise ValueError('Invalid tetrahedron indices')
    base=np.linalg.det(np.swapaxes(rest[tets[:,1:]]-rest[tets[:,0,None]],1,2))
    if not np.isfinite(base).all() or np.any(base<=0):raise ValueError('Nonpositive/degenerate reference tetrahedra')
    ds=frames[:,tets[:,1:]]-frames[:,tets[:,0,None]]
    return float(np.min(np.linalg.det(np.swapaxes(ds,-1,-2))/base))

def mapping(fine,coarse):
    dist=np.sum((fine[:,None]-coarse[None,:])**2,axis=2);ids=np.argmin(dist,axis=0)
    if np.max(dist[ids,np.arange(len(coarse))])>1e-20:raise ValueError('Reference vertices do not match')
    return ids

class Audit:
    def __init__(self):self.errors=[];self.checks=0;self.pending=[];self.reference_cache={};self.case_audit=[]
    def check(self,condition,label):
        self.checks+=1
        if not bool(condition):self.errors.append(label)
    def close(self,a,b,label):
        self.check(a is not None and b is not None and np.isclose(a,b,rtol=1e-8,atol=1e-12,equal_nan=True),label)
    def hashes(self):
        frozen=read(REPORT/'method-freeze.json');prior=read(REPORT/'prior-preservation.json')
        for path,sha in {**frozen['source_sha256'],**prior['sha256']}.items():
            f=ROOT/path;self.check(f.exists() and digest(f)==sha,'hash '+path)
        for name,key in [('protocol.json','protocol_sha256'),('physics-development-qualification.json','physics_qualification_sha256'),('pre-freeze-tests.txt','tests_sha256')]:
            self.check(digest(REPORT/name)==frozen[key],'freeze '+name)
        return len(prior['sha256'])
    def reference(self,obj,scene,p):
        key=(obj,scene)
        if key in self.reference_cache:return self.reference_cache[key]
        valid=True;levels=[];details=[]
        for directory in ('reference','reference-refinement'):
            path=BASE/directory/obj/scene
            if not (path/'complete.json').exists():valid=False;details.append(directory+' absent');continue
            status=read(path/'complete.json')
            if not status['success']:valid=False;details.append(directory+' generation failed');continue
            a=arrays(path/'trajectory.npz');d=status['diagnostics'];levels.append(a)
            finite=np.isfinite(a['frames']).all() and np.isfinite(a['times']).all()
            for name in ('centers','reactions','velocities'):
                self.check(name in a and np.isfinite(a[name]).all(),f'{obj}/{scene}/{directory} finite '+name)
            self.check(len(a['times'])==len(a['frames']) and np.all(np.diff(a['times'])>0),f'{obj}/{scene}/{directory} monotone matched times')
            passed=finite and np.isfinite(d['min_j']) and d['min_j']>.02 and np.isfinite(d['max_penetration_m']) and d['max_penetration_m']<=p['gates']['penetration_m']
            valid=valid and passed
            if finite:
                j=minimum_j(a['frames'],a['rest'],a['tets'])
                self.check(d['min_j']<=j+1e-8,f'{obj}/{scene}/{directory} integrated J exceeds saved minimum')
                self.check(j>.02,f'{obj}/{scene}/{directory} saved J invalid')
                details.append(dict(level=directory,saved_min_j=j,integrated_min_j=d['min_j'],penetration_m=d['max_penetration_m']))
        rms=None
        if len(levels)!=2:valid=False
        else:
            a,b=levels
            same=a['frames'].shape==b['frames'].shape and a['times'].shape==b['times'].shape and np.allclose(a['times'],b['times'],rtol=0,atol=1e-10)
            if not same:valid=False
            else:
                rms=float(np.sqrt(np.mean((a['frames']-b['frames'])**2)))
                valid=valid and np.isfinite(rms) and rms<=p['gates']['trajectory_refinement_rms_m']
        result=dict(qualified=bool(valid),rms_m=rms,details=details)
        self.reference_cache[key]=result;return result
    def selection(self,o,p):
        obj=o['id'];path=BASE/'inference'/obj
        if not (path/'selection.json').exists():
            calibration_status=BASE/'reference'/obj/'squeeze'/'complete.json'
            if calibration_status.exists() and not read(calibration_status)['success']:
                self.check(True,obj+' unavailable calibration explicitly recorded');return
            self.pending.append('selection '+obj);return
        obs=arrays(path/'observations.npz');saved=read(path/'selection.json')
        ref=arrays(BASE/'reference'/obj/'squeeze'/'trajectory.npz')
        ids=mapping(ref['rest'],obs['rest']);expected=ref['frames'][:,ids].copy()
        rest=obs['rest'];surface=np.flatnonzero(np.any(np.isclose(rest,rest.min(0))|np.isclose(rest,rest.max(0)),axis=1))
        nodes=surface[np.linspace(0,len(surface)-1,p['observation_nodes']).round().astype(int)]
        declared_mask=np.zeros(expected.shape[:2],bool);declared_mask[:,nodes]=True
        self.check(np.array_equal(obs['mask'],declared_mask),obj+' declared observation mask')
        seed=p['observation_seed']+next(i for i,x in enumerate(p['objects']) if x['id']==obj)
        expected+=np.random.default_rng(seed).normal(0,p['observation_noise_std_m'],expected.shape);expected[0]=obs['rest'];expected[~obs['mask']]=np.nan
        self.check(np.array_equal(expected,obs['observed'],equal_nan=True),obj+' observations match declared reference/noise/mask')
        for name,key in [('tools','tool_positions'),('times','times'),('active','active')]:self.check(np.array_equal(obs[name],ref[key]),obj+' public '+name)
        for family,scales in [('fem',p['fem_young_grid']),('native',p['native_scale_grid']),('rigid',[6000.])]:
            losses=[];params=[]
            for i,scale in enumerate(scales):
                for j,mu in enumerate(p['friction_grid']):
                    q=BASE/'library'/family/f'{i}-{j}';st=read(q/'complete.json');loss=np.inf
                    if st['success']:
                        f=arrays(q/'trajectory.npz')['frames']
                        if f.shape==expected.shape and np.isfinite(f).all() and obs['mask'].any():loss=float(np.mean((f[obs['mask']]-obs['observed'][obs['mask']])**2))
                    losses.append(loss);params.append((scale,mu))
            r=saved[family];success=bool(np.isfinite(losses).any());self.check(r['success']==success,obj+'/'+family+' selection success')
            self.check(np.allclose(losses,r['scores_m2'],rtol=1e-10,atol=1e-16,equal_nan=True),obj+'/'+family+' all calibration losses')
            if success:
                index=int(np.argmin(losses));scale,mu=params[index]
                self.check(r['index']==index and r['scale']==scale and r['friction']==mu,obj+'/'+family+' selected grid cell')
                self.check(r['scale_at_grid_edge']==(scale in [min(scales),max(scales)]),obj+'/'+family+' scale edge')
                self.check(r['friction_at_grid_edge']==(mu in [min(p['friction_grid']),max(p['friction_grid'])]),obj+'/'+family+' friction edge')
                self.close(r['calibration_rmse_m'],np.sqrt(losses[index]),obj+'/'+family+' fit RMSE')
    def split(self,split,p):
        path=REPORT/f'{split}-results.json'
        if not path.exists():self.pending.append(split+' results');return None
        d=read(path);objects=[o for o in p['objects'] if o['split']==split]
        want={(o['id'],s) for o in objects for s in p['forecast_scenarios']};got=[(r['object'],r['scenario']) for r in d['cases']]
        self.check(len(got)==len(set(got)) and set(got)==want,split+' exact unique declared cases')
        self.check(d['expected_cases']==len(want),split+' expected count')
        for o in objects:self.selection(o,p)
        counts={m:0 for m in METHODS};scores={m:[] for m in METHODS};pairs={m:[] for m in METHODS if m!='fem'};joint=0
        for row in d['cases']:
            obj,s=row['object'],row['scenario'];label=obj+'/'+s
            rq=self.reference(obj,s,p);cq=self.reference(obj,'squeeze',p);qualified=rq['qualified'] and cq['qualified'];joint+=int(qualified)
            for key,value in [('reference_qualified',rq['qualified']),('calibration_reference_qualified',cq['qualified']),('joint_reference_qualified',qualified)]:self.check(row[key]==value,label+' '+key)
            predpath=BASE/'evaluation'/obj/s/'predictions.npz'
            if not predpath.exists():
                self.check(not row['results'],label+' missing predictions have no scored output');continue
            pred=arrays(predpath);ref=arrays(BASE/'reference'/obj/s/'trajectory.npz')
            observation=arrays(BASE/'inference'/obj/'observations.npz')
            coarse=observation['rest'].copy()
            # Every episode is a translated copy of the same declared rest grid.
            coarse+=ref['rest'].mean(0)-coarse.mean(0)
            ids=mapping(ref['rest'],coarse);target=ref['frames'][:,ids]
            case=dict(object=obj,scenario=s,split=split,joint_qualified=qualified,metrics={})
            for method in METHODS:
                result=row['results'][method];calculated=geometry_metrics(pred.get(method),target)
                self.check(result['complete']==(calculated is not None),label+'/'+method+' completeness')
                if calculated is None:continue
                counts[method]+=1;scores[method].append(calculated['coordinate_rmse_m']);case['metrics'][method]=calculated
                for key,value in calculated.items():self.close(result[key],value,label+'/'+method+' '+key)
                if method in ('fem','native','rigid','oracle'):
                    q=BASE/'prediction'/obj/s/method;arr=arrays(q/'trajectory.npz');status=read(q/'complete.json')
                    self.check(np.array_equal(arr['frames'],pred[method]),label+'/'+method+' saved prediction identity')
                    calculated['saved_min_j']=minimum_j(arr['frames'],arr['rest'],arr['tets'])
                    self.check(status['diagnostics']['min_j']<=calculated['saved_min_j']+1e-7,label+'/'+method+' J diagnostic lower bound')
                    if s=='moving_sphere':
                        bodyerror=float(np.sqrt(np.mean(np.sum((arr['centers'][1:,-1]-ref['centers'][1:,-1])**2,axis=1))))
                        self.close(result['dynamic_body_position_rmse_m'],bodyerror,label+'/'+method+' moving body error');calculated['body_rmse_m']=bodyerror
                    job=status['job'];selection=read(BASE/'inference'/obj/'selection.json')
                    expected=next(o for o in objects if o['id']==obj) if method=='oracle' else selection[method]
                    scale=expected['young'] if method=='oracle' else expected['scale']
                    self.check(job['scale']==scale and job['friction']==expected['friction'],label+'/'+method+' forecast uses selected parameters')
            if 'fem' in case['metrics']:
                for method in pairs:
                    if method in case['metrics']:pairs[method].append(case['metrics'][method]['coordinate_rmse_m']-case['metrics']['fem']['coordinate_rmse_m'])
            self.case_audit.append(case)
        return dict(objects=len(objects),cases=len(want),joint_qualified=joint,methods={m:dict(complete=counts[m],mean_mm=float(np.mean(scores[m])*1000) if scores[m] else None) for m in METHODS},
                    fem_comparisons={m:dict(pairs=len(v),wins=int((np.array(v)>1e-12).sum()),losses=int((np.array(v)<-1e-12).sum()),ties=int((abs(np.array(v))<=1e-12).sum()),gain_mm=float(np.mean(v)*1000) if v else None) for m,v in pairs.items()})

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--allow-incomplete',action='store_true');args=parser.parse_args()
    a=Audit();prior=a.hashes();p=read(REPORT/'protocol.json')
    a.check(len(p['objects'])==14 and len({o['id'] for o in p['objects']})==14,'14 unique objects')
    a.check(len(p['forecast_scenarios'])==4 and len(set(p['forecast_scenarios']))==4,'four forecast families')
    splits={s:a.split(s,p) for s in ('development','validation','reserved')}
    result=dict(passed=not a.errors and not a.pending,checks=a.checks,errors=a.errors,pending=a.pending,prior_preserved_files=prior,splits=splits,cases=a.case_audit,
                references=[dict(object=k[0],scenario=k[1],**v) for k,v in a.reference_cache.items()],
                scope='Independent saved-array metrics, determinant geometry, discrete selection reconstruction and frozen-file preservation; integrated unsaved diagnostics are checked but not independently replayed.',verifier_sha256=digest(Path(__file__)))
    outfile=REPORT/('independent-audit-partial.json' if args.allow_incomplete else 'independent-audit.json');outfile.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases','references')},indent=2))
    return 1 if a.errors or (a.pending and not args.allow_incomplete) else 0
if __name__=='__main__':sys.exit(main())
