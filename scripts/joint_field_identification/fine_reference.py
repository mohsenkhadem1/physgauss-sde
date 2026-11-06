"""v03 sparse higher-resolution reference bank; frozen v01 remains untouched.

625 stiffness factorizations supply 3125 hypotheses: load location changes RHS,
not K. Reference-mesh log-modulus coordinates remain those of the frozen v01
8x4x4 bank, with the known quadrature-centering adjustment applied at each mesh.
"""
from pathlib import Path
import argparse,gc,hashlib,json,os,threading,time,datetime
import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import splu
from .mechanics import LinearGaussianBody
from . import bank

ROOT=bank.ROOT;BASE=ROOT/'runs/joint-field-fine-v03';REPORT=ROOT/'reports/joint-field-fine-v03'
INVERSE_MESH=(24,12,12);TRUTH_MESH=(32,16,16);NCASES=16


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def atomic_json(path,record):
    path=Path(path);path.parent.mkdir(exist_ok=True,parents=True);temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(record,indent=2));os.replace(temp,path)


def freeze():
    path=REPORT/'protocol.json'
    if path.exists():return json.loads(path.read_text())
    sources=[Path(__file__),Path(__file__).with_name('mechanics.py'),Path(__file__).with_name('bank.py'),Path(__file__).with_name('inference.py'),Path(__file__).with_name('campaign.py')]
    protocol=dict(version='joint-field-fine-v03',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),purpose='Paired coarse-v01 and fine-v03 inverse banks on same fresh finer-mesh truths; numerical discretization diagnosis, not continuum certification.',inverse_mesh=list(INVERSE_MESH),truth_mesh=list(TRUTH_MESH),grid_levels=5,hypotheses=3125,stiffness_factorizations=625,case_count_per_shape=NCASES,total_cases=32,seeds=dict(block=91001,step=91002),prior='Uniform original5^5 physical hypothesis dictionary; true parameters drawn fromsame dictionary, mesh differs.',noise_m=bank.NOISE,force_std_N=bank.FORCE_STD,probe_force_N=bank.PROBE_FORCE,budget=4,actions='Unmodified v01 bank.maps, same scalar measurement actions and held targets.',fixed_schedules='Reuse v01 development-selected fixed schedules, no new development tuning.',field_matching='Adjust logE by contrast*(newmesh material_log_center - original8x4x4 center), preserving continuous material field.',runtime_cap_per_worker_seconds=5400,cpu_threads_per_worker=2,memory_cap_per_worker_GiB=20,source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},v01_fixed_schedule_sha256={shape:sha(ROOT/f'reports/joint-field-identification-v01/{shape}/fixed-schedules.json') for shape in ['block','step']},caveat='32mesh truth is a finer numerical reference, not demonstrated continuum convergence; preserve refinement-cost.json evidence and tail errors.')
    atomic_json(path,protocol);return protocol


def validate_protocol():
    path=REPORT/'protocol.json';p=json.loads(path.read_text())
    for file,digest in p['source_sha256'].items():
        assert sha(ROOT/file)==digest,f'Frozen v03 source changed: {file}'
    for shape,digest in p['v01_fixed_schedule_sha256'].items():
        assert sha(ROOT/f'reports/joint-field-identification-v01/{shape}/fixed-schedules.json')==digest
    return p,sha(path)


class SparseReference:
    def __init__(self,shape,cells):
        self.body=LinearGaussianBody(shape,cells);self.points=bank.points();self.N=self.body.interpolation(self.points).numpy()
        old=LinearGaussianBody(shape,(8,4,4));self.center_reference=float(old.material_log_center);del old
        self.probes=self.body.custom_probe_forces(bank.probe_specs()).numpy()
        self.last_factor_seconds=0.

    def adjusted(self,theta):
        result=np.array(theta,copy=True);result[...,0]+=result[...,1]*(float(self.body.material_log_center)-self.center_reference);return result

    def unknown_forces(self,locations):
        th=torch.tensor(locations,dtype=torch.float64);top=self.body.footprint(th,'top').numpy();side=self.body.footprint(th,'side').numpy()
        columns=[]
        for a,b in zip(top,side):columns.extend([(a[:,None]*[0.,0.,-1.]).ravel(),(b[:,None]*[0.,-1.,0.]).ravel()])
        return np.stack(columns,-1)

    def factor(self,theta):
        begin=time.perf_counter();th=self.adjusted(theta);E=self.body.material_modulus(torch.tensor(th[None],dtype=torch.float64))[0].numpy();values=(E[:,None,None]*self.body.local_unit_stiffness.numpy()).ravel()
        index=self.body._assembly_index.numpy();K=sparse.coo_matrix((values,(index//self.body.ndof,index%self.body.ndof)),shape=(self.body.ndof,self.body.ndof)).tocsc()
        weight=self.body.footprint([th[2]],'bottom')[0].numpy();K+=sparse.diags(np.repeat(np.exp(th[3])*weight,3),format='csc')
        self.last_assembly_seconds=time.perf_counter()-begin;begin=time.perf_counter();factor=splu(K,permc_spec='MMD_AT_PLUS_A');self.last_factor_seconds=time.perf_counter()-begin
        return K,factor

    def solve(self,theta,locations):
        rhs=np.concatenate([self.unknown_forces(locations),self.probes],-1);K,factor=self.factor(theta)
        begin=time.perf_counter();solution=factor.solve(rhs);solve_seconds=time.perf_counter()-begin
        residual=float(np.linalg.norm(K@solution-rhs)/np.linalg.norm(rhs));assert residual<1e-8,residual
        points=np.einsum('pn,ncl->pcl',self.N,solution.reshape(self.body.n,3,-1));return points,dict(assembly_seconds=self.last_assembly_seconds,factor_seconds=self.last_factor_seconds,solve_seconds=solve_seconds,relative_residual=residual)

    def material(self,theta):return self.body.modulus_at_points(self.adjusted(theta),self.points).numpy()


def parity(shape):
    ref=SparseReference(shape,(4,2,2));theta=bank.BOUNDS.mean(1);locs=np.linspace(*bank.BOUNDS[4],5);response,meta=ref.solve(theta,locs)
    params=np.repeat(theta[None],5,axis=0);params[:,4]=locs
    dense=ref.body.solve(ref.adjusted(params),probe_specs=bank.probe_specs());unit=ref.body.point_response(dense['unit_response'],ref.points).numpy();probe=ref.body.point_response(dense['probe_response'],ref.points).numpy()
    err=max(float(np.max(np.abs(response[:,:,2*i:2*i+2]-unit[i]))) for i in range(5));probe_err=float(np.max(np.abs(response[:,:,10:]-probe[0])))
    assert max(err,probe_err)<1e-10
    return dict(shape=shape,sparse_dense_unit_max_difference_m_per_N=err,sparse_dense_probe_max_difference_m_per_N=probe_err,**meta)


def allocate(path,shape):
    if path.exists():
        result=np.load(path,mmap_mode='r+');assert result.shape==shape;return result
    return np.lib.format.open_memmap(path,mode='w+',dtype=np.float64,shape=shape)


def build_bank(shape,protocol_hash):
    root=BASE/shape;root.mkdir(exist_ok=True,parents=True);final=BASE/f'{shape}.npz';meta_path=REPORT/f'{shape}-bank.json'
    if final.exists() and meta_path.exists():
        meta=json.loads(meta_path.read_text());assert meta['protocol_sha256']==protocol_hash and meta['sha256']==sha(final);return
    checkpoint=root/'bank-progress.json';state=json.loads(checkpoint.read_text()) if checkpoint.exists() else dict(completed_groups=0,records=[],protocol_sha256=protocol_hash)
    assert state['protocol_sha256']==protocol_hash
    theta=bank.grid();unit=allocate(root/'unit.npy',(3125,20,3,2));probes=allocate(root/'probe.npy',(3125,20,3,10));material=allocate(root/'material.npy',(3125,20));ref=SparseReference(shape,INVERSE_MESH);started=time.monotonic();locations=np.linspace(*bank.BOUNDS[4],5)
    for group in range(state['completed_groups'],625):
        start=5*group;subset=theta[start:start+5];assert np.allclose(subset[:,:4],subset[0,:4]);assert np.allclose(subset[:,4],locations)
        response,meta=ref.solve(subset[0],locations)
        for j in range(5):unit[start+j]=response[:,:,2*j:2*j+2];probes[start+j]=response[:,:,10:]
        material[start:start+5]=ref.material(subset)
        unit.flush();probes.flush();material.flush();state['records'].append(dict(group=group,**meta));state['completed_groups']=group+1;state['elapsed_current_execution_seconds']=time.monotonic()-started;atomic_json(checkpoint,state)
        if group%25==0:print(shape,'bank',group+1,'/625','factor',round(meta['factor_seconds'],2),'elapsed',round(time.monotonic()-started,1),flush=True)
    tmp=BASE/f'{shape}-finalizing.npz';np.savez(tmp,theta=theta,unit=unit,probe=probes,material=material,points=ref.points,cells=np.array(INVERSE_MESH));os.replace(tmp,final)
    metadata=dict(shape=shape,protocol_sha256=protocol_hash,sha256=sha(final),completed=True,groups=625,hypotheses=3125,cells=list(INVERSE_MESH),body=ref.body.metadata(),field_coordinate_reference_mesh=[8,4,4],factorization_seconds_sum=sum(r['factor_seconds'] for r in state['records']),assembly_seconds_sum=sum(r['assembly_seconds'] for r in state['records']),solve_seconds_sum=sum(r['solve_seconds'] for r in state['records']),maximum_relative_residual=max(r['relative_residual'] for r in state['records']),parity=parity(shape))
    atomic_json(meta_path,metadata);print(shape,'bank complete',metadata['factorization_seconds_sum'],flush=True);del ref,unit,probes,material;gc.collect()


def build_truth(shape,protocol,protocol_hash):
    root=BASE/shape;root.mkdir(exist_ok=True,parents=True);path=root/'fine-truth.npz';meta_path=REPORT/f'{shape}-truth.json'
    if path.exists() and meta_path.exists():
        meta=json.loads(meta_path.read_text());assert meta['protocol_sha256']==protocol_hash and meta['sha256']==sha(path);return
    rng=np.random.default_rng(protocol['seeds'][shape]);indices=rng.integers(3125,size=NCASES);theta=bank.grid()[indices]
    response_dir=root/'truth-responses';response_dir.mkdir(exist_ok=True);ref=SparseReference(shape,TRUTH_MESH);unit=[];probes=[];material=[];cost=[]
    for i,th in enumerate(theta):
        filename=response_dir/f'case-{i:03d}.npz';recordpath=filename.with_suffix('.json')
        if filename.exists() and recordpath.exists():
            arrays=dict(np.load(filename));record=json.loads(recordpath.read_text());assert record['protocol_sha256']==protocol_hash;assert np.array_equal(arrays['theta'],th)
        else:
            response,record=ref.solve(th,[th[4]]);arrays=dict(theta=th,unit=response[:,:,:2],probe=response[:,:,2:],material=ref.material(th)[0]);np.savez(filename,**arrays);record.update(case=i,protocol_sha256=protocol_hash);atomic_json(recordpath,record)
        unit.append(arrays['unit']);probes.append(arrays['probe']);material.append(arrays['material']);cost.append(record);print(shape,'fine truth',i+1,'/16','factor',round(record['factor_seconds'],2),flush=True)
    data=dict(theta=theta,unit=np.stack(unit),probe=np.stack(probes),material=np.stack(material),points=ref.points);maps=bank.maps(data);force=rng.normal(size=(NCASES,16))@np.linalg.cholesky(bank.load_cov()).T
    y=np.einsum('hof,hf->ho',maps['H'],force)+rng.normal(scale=bank.NOISE,size=(NCASES,maps['H'].shape[1]));target=np.einsum('hdf,hf->hd',maps['T'],force)+maps['offset']
    measurements=np.stack([np.einsum('hf,hf->h',a['h'],force)+a['offset'] for a in maps['actions']],1)+rng.normal(scale=bank.NOISE,size=(NCASES,len(maps['actions'])))
    np.savez(path,theta=theta,material=data['material'],force=force,y=y,target=target,measurements=measurements,indices=indices,seed=np.array(protocol['seeds'][shape]),kind=np.array('grid-parameters-fine32-truth'),protocol_sha256=np.array(protocol_hash))
    meta=dict(shape=shape,protocol_sha256=protocol_hash,sha256=sha(path),cases=NCASES,seed=protocol['seeds'][shape],cells=list(TRUTH_MESH),known_parameter_indices=indices.tolist(),cost=cost,scope='Fresh same-grid physical parameters and GP loads; higher-mesh truth, shared measurements for coarse/fine inverse comparison. No filtering or outcome selection.')
    atomic_json(meta_path,meta);print(shape,'fine truth complete',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--freeze',action='store_true');ap.add_argument('--shape',choices=['block','step']);ap.add_argument('--phase',choices=['all','bank','truth'],default='all');args=ap.parse_args();BASE.mkdir(exist_ok=True,parents=True);REPORT.mkdir(exist_ok=True,parents=True)
    if args.freeze:
        print(json.dumps(freeze(),indent=2));return
    assert args.shape;protocol,digest=validate_protocol();torch.set_num_threads(2);started=time.monotonic();finished=threading.Event()
    def guard():
        while not finished.wait(5):
            with open('/proc/self/status') as f:rss=next(int(line.split()[1])/1024**2 for line in f if line.startswith('VmRSS:'))
            if time.monotonic()-started>5400 or rss>20:
                print('Resource guard stopped worker; group checkpoints retained',flush=True);os._exit(124)
    threading.Thread(target=guard,daemon=True).start()
    if args.phase in ('all','bank'):build_bank(args.shape,digest)
    if args.phase in ('all','truth'):
        validate_protocol();build_truth(args.shape,protocol,digest)
    validate_protocol();finished.set();print(args.shape,'completed',time.monotonic()-started,flush=True)

if __name__=='__main__':main()
