"""Fixed physical-probe/mass diagnostic of frozen nonlinear captures, no solves.

New metrics never replace the failed corner-node qualification. Probe locations,
weights and proper-rigid registration rule freeze before trajectory extraction.
"""
from __future__ import annotations
import argparse,hashlib,json,signal,time
from pathlib import Path
import numpy as np
try:
    from .campaign_nonlinear_excitation import tiny_geometry
except ImportError:
    from campaign_nonlinear_excitation import tiny_geometry


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def interpolate(rest,tets,positions,probes,tolerance=2e-12):
    X=np.asarray(rest);T=np.asarray(tets);P=np.asarray(probes);q=np.asarray(positions)
    corners=X[T];inverse=np.linalg.inv(np.swapaxes(corners[:,1:]-corners[:,:1],-1,-2))
    local=np.einsum('eij,pej->pei',inverse,P[:,None]-corners[None,:,0])
    bary=np.concatenate([1-local.sum(-1,keepdims=True),local],axis=-1)
    inside=np.all(bary>=-tolerance,axis=-1)&np.all(bary<=1+tolerance,axis=-1)
    if not np.all(inside.any(1)):raise ValueError('Physical probe outside authored domain')
    # Earliest authoredtet breaks ties on interfaces; P1 trace is continuous.
    ids=inside.argmax(1);weights=bary[np.arange(len(P)),ids]
    return np.einsum('pk,tpkc->tpc',weights,q[:,T[ids]]),ids,weights


def rigid_registration(a,b,weights):
    w=np.asarray(weights);w=w/w.sum();ma=np.einsum('p,tpc->tc',w,a);mb=np.einsum('p,tpc->tc',w,b)
    ca=a-ma[:,None];cb=b-mb[:,None];H=np.einsum('p,tpi,tpj->tij',w,ca,cb)
    U,_,V=np.linalg.svd(H);fix=np.repeat(np.eye(3)[None],len(a),axis=0);fix[:,2,2]=np.linalg.det(U@V)
    Q=U@fix@V;shift=mb-np.einsum('ti,tij->tj',ma,Q)
    return Q,shift


def error_metrics(a,b,weights):
    w=np.asarray(weights);w=w/w.sum();error=a-b
    return {'weighted_coordinate_RMS_m':float(np.sqrt(np.mean(np.einsum('p,tpc->tc',w,error**2)))),'maximum_coordinate_error_m':float(np.max(abs(error))),'maximum_weighted_frame_coordinate_RMS_m':float(np.sqrt(np.max(np.einsum('p,tpc->t',w,error**2)/3)))}


def mass_properties(rest,tets,density=625.):
    X=np.asarray(rest);T=np.asarray(tets);corners=X[T]
    volume=np.linalg.det(np.swapaxes(corners[:,1:]-corners[:,:1],-1,-2))/6
    if np.any(volume<=0):raise ValueError('Positive authoredvolume required')
    mass=density*volume;center=corners.mean(1);total=mass.sum();COM=np.einsum('e,ei->i',mass,center)/total
    offsets=corners-center[:,None];cov=np.einsum('eai,eaj->eij',offsets,offsets)/20
    exact_second=np.einsum('e,eij->ij',mass,cov+np.einsum('ei,ej->eij',center-COM,center-COM))
    exact_inertia=np.trace(exact_second)*np.eye(3)-exact_second
    nodal=np.zeros(len(X));np.add.at(nodal,T,mass[:,None]*np.ones((1,4))/4)
    lumpedCOM=nodal@X/total;d=X-lumpedCOM
    lumped_second=np.einsum('n,ni,nj->ij',nodal,d,d);lumped_inertia=np.trace(lumped_second)*np.eye(3)-lumped_second
    return {'total_mass_kg':float(total),'exact_COM_m':COM.tolist(),'lumped_COM_m':lumpedCOM.tolist(),'exact_inertia_COM_kg_m2':exact_inertia.tolist(),'lumped_inertia_COM_kg_m2':lumped_inertia.tolist(),'relative_lumped_inertia_Frobenius_error':float(np.linalg.norm(lumped_inertia-exact_inertia)/np.linalg.norm(exact_inertia)),'exact_principal_inertias_kg_m2':np.linalg.eigvalsh(exact_inertia).tolist(),'lumped_principal_inertias_kg_m2':np.linalg.eigvalsh(lumped_inertia).tolist()}


def prepare(path):
    X,T,F=tiny_geometry();rng=np.random.default_rng(2026120801);points=[];weights=[]
    for tet in T:
        volume=np.linalg.det((X[tet[1:]]-X[tet[0]]).T)/6
        points.extend(rng.dirichlet(np.ones(4),size=32)@X[tet]);weights.extend([volume/32]*32)
    tri=X[F];areas=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=-1)/2
    root=Path('runs/gaussian-campaign-v01/nonlinear-resolution-v02')
    sources=['experiments/diagnose_nonlinear_physical_probes.py','experiments/campaign_nonlinear_excitation.py','experiments/campaign_compliant_gaussian_linear.py']
    d={'schema':'fixed-physical-probe-diagnostic-v0.1','status':'FROZEN_BEFORE_NEW_PROBE_TRAJECTORY_EXTRACTION','CPU_threads':2,'wall_cap_s':60,'seed':2026120801,'volume_probe_law':'32uniformDirichlet(1,1,1,1) interiorprobesperauthoredtet;volume-stratifiedweights','volume_probes_m':np.asarray(points).tolist(),'volume_weights':(np.asarray(weights)/sum(weights)).tolist(),'surface_probes_m':tri.mean(1).tolist(),'surface_area_weights':(areas/areas.sum()).tolist(),'corner_probes_m':X[[1,4]].tolist(),'corner_weights':[.5,.5],
       'captures':{str(n):{'path':str(root/f'mesh{n}-hz240.npz'),'sha256':sha(root/f'mesh{n}-hz240.npz')} for n in [128,1024,8192]},'pairs':[[128,1024],[1024,8192],[128,8192]],
       'metrics':['Full49frame weightedcoordinateRMS andmaximumabsoluteerror','ProperrigidfitusingONLY64volumeprobes,applysamefittoallgroups','ExactuniformtetmassCOMinertiaandnodallumpedinertia','Original2cornerfailedgateunchanged'],
       'nonclaims':['FixedMonteCarlovolumeandfacecentroidquadraturesarefiniteprobediagnostics,notcertifiedcontinuumnorms','Rigidregistrationisdiagnostic,NOTallowedreplacementofqualificationgate','Inertiadriftandrigidremovabilitydonotbythemselvesproveacause','No newphysics, currentcohort or inference'], 'source_sha256':{p:sha(p) for p in sources}}
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(d,indent=2)+'\n');return d


def run(protocol,digest,output):
    if sha(protocol)!=digest:raise ValueError('Changedprobeprotocol')
    p=json.loads(Path(protocol).read_text())
    for path,expected in p['source_sha256'].items():
        if sha(path)!=expected:raise ValueError('Changedsource '+path)
    for r in p['captures'].values():
        if sha(r['path'])!=r['sha256']:raise ValueError('Changedcapture')
    out=Path(output);out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    signal.signal(signal.SIGALRM,lambda s,f:(_ for _ in ()).throw(TimeoutError('60sCPU2diagnosticbudget')));signal.alarm(60)
    result={'schema':'fixed-physical-probe-diagnostic-result-v0.1','completed':False,'protocol':{'path':str(protocol),'sha256':digest},'mass':{},'comparisons':{},'old_failed_qualification_unchanged':True,'new_physics_calls':0}
    try:
        groups=['volume','surface','corner'];all_probes=np.concatenate([p[g+'_probes_m'] for g in groups]);cuts=[slice(0,64),slice(64,70),slice(70,72)];fields={};raw={}
        for n,r in p['captures'].items():
            with np.load(r['path'],allow_pickle=False) as a:
                q=a['positions_m'];X=a['rest_m'];T=a['tets'];times=a['times_s']
            if q.shape[0]!=49 or not np.array_equal(times,np.arange(49)/240):raise ValueError('Frozenfullclockmismatch')
            fields[n],ids,bary=interpolate(X,T,q,all_probes);result['mass'][n]=mass_properties(X,T)
            raw[n]={'probes':fields[n],'tet_ids':ids,'barycentric_weights':bary}
        for low,high in p['pairs']:
            a=fields[str(low)];b=fields[str(high)];Q,shift=rigid_registration(a[:,:64],b[:,:64],p['volume_weights'])
            aligned=np.einsum('tpi,tij->tpj',a,Q)+shift[:,None];c={}
            for group,sl in zip(groups,cuts):
                w=p['volume_weights'] if group=='volume' else p['surface_area_weights'] if group=='surface' else p['corner_weights']
                before=error_metrics(a[:,sl],b[:,sl],w);after=error_metrics(aligned[:,sl],b[:,sl],w)
                fraction=1-(after['weighted_coordinate_RMS_m']/before['weighted_coordinate_RMS_m'])**2 if before['weighted_coordinate_RMS_m'] else 0.
                c[group]={'raw':before,'after_volume_rigid_registration':after,'fraction_squared_error_rigid_removable':fraction}
            angles=np.arccos(np.clip((np.trace(Q,axis1=1,axis2=2)-1)/2,-1,1))
            c['registration']={'maximum_angle_rad':float(angles.max()),'maximum_origin_translation_m':float(np.max(np.linalg.norm(shift,axis=-1))),'fit_uses':'64fixedvolumeprobesonly'}
            result['comparisons'][f'{low}_to_{high}']=c
        np.savez_compressed(out/'fixed-probe-trajectories.npz',**{str(n):a for n,a in fields.items()})
        result['completed']=True
    except Exception as exc:result['failure']=repr(exc)
    finally:signal.alarm(0);result['elapsed_s']=time.monotonic()-start;(out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='action',required=True)
    a=s.add_parser('prepare');a.add_argument('--protocol',required=True)
    a=s.add_parser('run');a.add_argument('--protocol',required=True);a.add_argument('--protocol-sha256',required=True);a.add_argument('--output',required=True)
    a=p.parse_args()
    if a.action=='prepare':prepare(a.protocol);print(sha(a.protocol))
    else:
        d=run(a.protocol,a.protocol_sha256,a.output);print(json.dumps(d,indent=2));raise SystemExit(0 if d['completed'] else 1)
