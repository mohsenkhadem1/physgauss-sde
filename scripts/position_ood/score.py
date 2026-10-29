"""Separate scoring of stored predictions, with object-level summaries."""
from pathlib import Path
import argparse,json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.position_ood.study import BASE,OUT,protocol
from scripts.phystwin.sequential_fixture import points
from scripts.phystwin.sequential_reference import write


def metrics(predicted,reference):
    error=predicted-reference;front=np.isclose(points()[:,1],-.04);surface=np.any(np.isclose(abs(points()-[.12,0,0]),[.12,.04,.04]),axis=1);hidden=surface&~front
    return dict(coordinate_rmse_mm=(1000*np.sqrt(np.mean(error**2,axis=(1,2)))).tolist(),point_rmse_mm=(1000*np.sqrt(np.mean(np.sum(error**2,axis=-1),axis=1))).tolist(),visible_rmse_mm=(1000*np.sqrt(np.mean(error[:,front]**2,axis=(1,2)))).tolist(),hidden_surface_rmse_mm=(1000*np.sqrt(np.mean(error[:,hidden]**2,axis=(1,2)))).tolist(),maximum_point_error_mm=(1000*np.linalg.norm(error,axis=-1).max(1)).tolist())


def deformation(field):
    rest=points();rows=[]
    for u in field:
        x=rest+u;xc=x-x.mean(0);X=rest-rest.mean(0);U,_,V=np.linalg.svd(X.T@xc);D=np.eye(3);D[-1,-1]=np.linalg.det(U@V);R=U@D@V;nonrigid=xc-X@R
        total=float(np.sqrt(np.mean(u*u)));nr=float(np.sqrt(np.mean(nonrigid*nonrigid)));rows.append(dict(motion_rms_mm=1000*total,nonrigid_rms_mm=1000*nr,nonrigid_fraction=nr/max(total,1e-15)))
    return rows


def paired_interval(differences,seed=2909):
    x=np.asarray(differences);rng=np.random.default_rng(seed);samples=x[rng.integers(0,len(x),(10000,len(x)))].mean(1)
    return dict(objects=len(x),mean=float(x.mean()),paired_bootstrap_95_percentile=np.quantile(samples,[.025,.975]).tolist())


def run(split):
    if 'test' in split.split('_'):
        from scripts.position_ood.seal import verify
        verify()
    s=protocol();rows=[];missing=[]
    for i in range(len(s[split])):
        path=BASE/f'fields/{split}-{i}.npz'
        if not path.exists():missing.append(i);continue
        ref=np.load(path)['field'];methods={}
        for p in (BASE/f'inverse/{split}-{i}').glob('*.npz'):
            if 'uncertainty' not in p.name:methods[p.stem]=p
        for p in (BASE/f'phystwin/{split}-{i}').glob('*/predictions.npz'):methods['phystwin-'+p.parent.name]=p
        p=BASE/f'mpm/{split}-{i}/predictions.npz'
        if p.exists():methods['physgaussian-mpm']=p
        for mode in ['id','all']:
            p=BASE/f'neural-predictions/{split}-{i}-{mode}.npz'
            if p.exists():methods['neural-'+mode]=p
        measurements=np.load(BASE/f'observations/{split}-{i}.npz')['measurements'] if (BASE/f'observations/{split}-{i}.npz').exists() else None
        result={name:metrics(np.load(p)['predicted'],ref) for name,p in methods.items()};result['undeformed']=metrics(np.zeros_like(ref),ref);result['last_full_state_oracle_persistence']=metrics(np.repeat(ref[2:3],len(ref),axis=0),ref)
        rows.append(dict(object=i,methods=result,deformation=deformation(ref)))
    report=dict(split=split,declared_objects=len(s[split]),completed_reference_objects=len(rows),missing_references=missing,rows=rows,primary_metric='Coordinate RMSE = 1000 sqrt(sum over xyz and query points of squared displacement error / (3 N)); aggregate per object, then average objects. Euclidean point RMSE reported separately.',groups=dict(prefix=[0,1,2],interpolation=[3],new_interaction=[4,5,6,7,8],larger_amplitude=[9]),scope='Development diagnostics' if split=='development' else 'Frozen held-out evaluation')
    write(OUT/f'{split}-scores.json',report);return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',default='development');a=p.parse_args();r=run(a.split);print(r['split'],r['completed_reference_objects'],'/',r['declared_objects'])
