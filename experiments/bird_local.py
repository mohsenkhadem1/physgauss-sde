"""Local Bird inference: observed-only audit, frozen preconditioning and strict gate.

Historical chains train a proposal only. They never enter new estimates.
CPU mechanics and optional CUDA rendering evaluate the same float64 target.
"""
import argparse
import copy
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
torch.use_deterministic_algorithms(True)
from pacnerf_conditioned_transport import make_target
from pacnerf_floor import make_floor_model
from pacnerf_hybrid import conditional_reference_transform
from pacnerf_recovery import restore_correlations
from bird_dense_sampler import sample
from pacnerf_elliptical import atomic_save
from compare_pacnerf_discrepancy import replay, force_moments_stream
from physgauss.chain_diagnostics import diagnose_quantities


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def source_hashes():
    paths = [* (ROOT / 'src/physgauss').glob('*.py'),
             * (ROOT / 'scripts').glob('*.py'), Path(__file__),
             ROOT/'experiments/bird_dense_sampler.py']
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def load_model(inputs, gate):
    model, _ = make_floor_model(inputs, 8)
    restore_correlations(model, gate['source_protocol']['frozen_model'])
    return model


def label(name, modes=12, core=326):
    kind, _, suffix = name.rpartition('_')
    if kind not in ('physical', 'native'):
        return name
    index = int(suffix)
    if index in (0, 1):
        return f'{kind}: ' + ('log stiffness' if index == 0 else 'log damping')
    if index >= core:
        return f'{kind}: ' + ['mechanics discrepancy log scale', 'smooth load log scale', 'RGB discrepancy log scale'][index-core]
    if index < 2 + modes:
        return f'{kind}: initial velocity mode {index-2}'
    frame, slot = divmod(index - 2 - modes, 2*modes)
    return f'{kind}: frame {frame+1} ' + ('displacement' if slot < modes else 'velocity') + f' mode {slot % modes}'


class DeviceRenderer:
    """Run only full-image rendering on CUDA; preserve differentiable CPU copies.

    No pixels, precision, geometry, floor clipping or likelihood terms change.
    Auxiliary CPU-facing attributes are retained for all existing score code.
    """
    def __init__(self, renderer, device):
        self.cpu = renderer
        self.device = torch.device(device)
        self.gpu = copy.deepcopy(renderer)
        memo = set()

        def move(value):
            if isinstance(value, torch.Tensor):
                return value.to(self.device)
            if isinstance(value, dict):
                return {k: move(v) for k, v in value.items()}
            if isinstance(value, list):
                return [move(v) for v in value]
            if isinstance(value, tuple):
                return tuple(move(v) for v in value)
            if hasattr(value, '__dict__') and id(value) not in memo:
                memo.add(id(value))
                for k, v in vars(value).copy().items():
                    setattr(value, k, move(v))
            return value

        move(self.gpu)

    def __getattr__(self, name):
        return getattr(self.cpu, name)

    def __call__(self, q, specs, colors=None, pixel_indices=None):
        with torch.device(self.device):
            result = self.gpu(q.to(self.device), specs,
                colors=None if colors is None else colors.to(self.device),
                pixel_indices=None if pixel_indices is None else pixel_indices.to(self.device))
        return result.to(q.device)


@torch.no_grad()
def prepare(args):
    out = Path(args.output)
    if out.exists():
        raise ValueError('Use a fresh proposal output directory')
    out.mkdir(parents=True)
    gate = json.loads(Path(args.gate).read_text())
    diagnostic = json.loads((Path(args.gate).parent / 'diagnostics.json').read_text())
    failed = [{ 'quantity': k, 'meaning': label(k), **v }
              for k,v in diagnostic['quantities'].items() if not v['diagnostic_pass']]
    failed.sort(key=lambda row: row['rank_folded_split_rhat'] or 999, reverse=True)
    model = load_model(args.inputs, gate)
    parent = torch.load(args.reference, weights_only=True)
    old = make_target(model, parent)
    merged = torch.load(Path(args.gate).parent / 'merged.pt', weights_only=True)
    if sha(Path(args.inputs)/'bird-inputs.pt') != gate['source_protocol']['input_sha256']:
        raise ValueError('Historical input hash differs')
    if sha(args.reference) != gate['reference_sha256']:
        raise ValueError('Historical reference hash differs')
    proof = replay(model, old, merged)
    white = merged['white']
    native = old.coordinates(white.flatten(0,1))
    mean = native.mean(0)
    covariance = torch.cov(native.T)
    eigen = torch.linalg.eigvalsh(covariance)
    # Small fixed shrinkage stabilizes an empirical proposal, not the target.
    covariance = (1-args.shrinkage)*covariance + args.shrinkage*torch.diag(covariance.diag())
    covariance += args.ridge*torch.eye(model.dimension, dtype=mean.dtype)
    precision = torch.cholesky_inverse(torch.linalg.cholesky(covariance))
    reference = copy.deepcopy(parent)
    for key in ['sampling_mean', 'sampling_precision']:
        reference.pop(key, None)
    reference.update(mean=mean, precision=precision,
        reference_kind='empirical-full-bridge-covariance-observed-only-v1')
    transform = conditional_reference_transform(precision, model.core_dimension)
    endpoints = old.coordinates(white[:,-1])
    reference['initial_white'] = torch.linalg.solve(transform, (endpoints-mean).T).T
    target = make_target(model, reference)
    physical = old.decode(white[:,-1])
    error = float((target.decode(reference['initial_white'])-physical).abs().max())
    if error > 1e-8:
        raise ValueError('Preconditioning changed initial physical states')
    report = {'parent_gate_sha256':sha(args.gate), 'parent_reference_sha256':sha(args.reference),
        'parent_merged_sha256':sha(Path(args.gate).parent/'merged.pt'),
        'input_sha256':sha(Path(args.inputs)/'bird-inputs.pt'),
        'discarded_training_draws':len(native), 'failed_quantities':failed,
        'native_covariance_eigenvalue_range':[float(eigen[0]),float(eigen[-1])],
        'native_covariance_condition_ratio':float(eigen[-1]/eigen[0]),
        'shrinkage_to_empirical_diagonal':args.shrinkage, 'ridge':args.ridge,
        'maximum_initial_physical_state_error':error, 'historical_density_replay':proof,
        'future_geometry_used':False, 'training_draws_in_new_estimates':0,
        'physical_target_changed':False}
    reference['proposal_training'] = report
    atomic_save(out/'reference.pt', reference)
    write(out/'audit.json', report)
    write(out/'parent-gate.json', gate)
    print(json.dumps({k:v for k,v in report.items() if k!='failed_quantities'}),flush=True)


def gpu_check(model, device, physical):
    cpu = model.renderer
    q,_ = model.decode(physical)
    pose = q[:,model.frames].detach()
    with torch.enable_grad():
        a = pose.clone().requires_grad_()
        first = cpu(a,model.specs)
        g1, = torch.autograd.grad(first.square().sum(),a)
        gpu = DeviceRenderer(cpu,device)
        b = pose.clone().requires_grad_()
        second = gpu(b,model.specs)
        g2, = torch.autograd.grad(second.square().sum(),b)
    error = float((first-second).detach().abs().max())
    grad = float((g1-g2).detach().abs().max())
    if error > 5e-13 or not torch.allclose(g1,g2,rtol=1e-9,atol=1e-8):
        raise ValueError(f'CPU/CUDA rendering mismatch: RGB={error}, gradient={grad}')
    times = {}
    for name,renderer in [('cpu',cpu),('cuda',gpu)]:
        began=time.perf_counter()
        for _ in range(5):
            with torch.enable_grad():
                p=pose.clone().requires_grad_()
                result=renderer(p,model.specs)
                torch.autograd.grad(result.square().sum(),p)
        times[name]=(time.perf_counter()-began)/5
    return gpu, {'device':device,'maximum_rgb_difference':error,
        'maximum_pose_gradient_difference':grad,'render_and_backward_seconds':times}


def run(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    gate=json.loads(Path(args.gate).read_text())
    model=load_model(args.inputs,gate)
    reference=torch.load(args.reference,weights_only=True)
    target=make_target(model,reference)
    initial=reference['initial_white'][args.chain_index:args.chain_index+1]
    if args.device!='cpu':
        renderer,proof=gpu_check(model,args.device,target.decode(initial))
        write(out/'device-validation.json',proof)
        print(json.dumps(proof),flush=True)
        model.renderer=renderer
        model._image_cache={}
    protocol={'case_index':8,'chain_index':args.chain_index,'seed':args.seed,
        'warmup':args.warmup,'min_time':args.min_time,'max_time':args.max_time,
        'device':args.device,'input_sha256':sha(Path(args.inputs)/'bird-inputs.pt'),
        'reference_sha256':sha(args.reference),'source_sha256':source_hashes(),
        'frozen_model':model.specification(),'torch':torch.__version__,
        'python':platform.python_version(),'threads':torch.get_num_threads(),
        'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
        'cublas_workspace_config':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'future_geometry_used':False,'valid_for_ranking':False}
    path=out/'local_protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:
        raise ValueError('Frozen local protocol changed')
    write(path,protocol)
    if not (out/'reference.pt').exists():
        import shutil
        shutil.copyfile(args.reference,out/'reference.pt')
    began=time.time()
    chains=sample(target,model.dimension,model.core_dimension,out,args.draws,
        warmup=args.warmup,seed=args.seed,initial=initial,chains_count=1,
        min_time=args.min_time,max_time=args.max_time,interweave=True,
        progress=lambda row:print(json.dumps({'elapsed_s':time.time()-began,**row}),flush=True))
    proof=replay(model,target,chains)
    write(out/'replay.json',proof)
    print(json.dumps({'retained':chains['retained'],'density_replay':proof}),flush=True)


@torch.no_grad()
def aggregate(args):
    root=Path(args.shards);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    paths=[root/f'chain-{i}' for i in range(4)]
    protocols=[json.loads((p/'local_protocol.json').read_text()) for p in paths]
    common=lambda p:{k:v for k,v in p.items() if k not in ['chain_index','seed','device']}
    if any(common(p)!=common(protocols[0]) for p in protocols):
        raise ValueError('Different model/source/reference/environment across shards')
    if [p['chain_index'] for p in protocols]!=list(range(4)) or len({p['seed'] for p in protocols})!=4:
        raise ValueError('Expected four independently seeded chains')
    if protocols[0]['source_sha256']!=source_hashes():
        raise ValueError('Frozen inference source differs')
    if sha(Path(args.inputs)/'bird-inputs.pt')!=protocols[0]['input_sha256']:
        raise ValueError('Input hash differs')
    data=[torch.load(p/'chains.pt',weights_only=True) for p in paths]
    kernel=lambda d:{k:v for k,v in d['protocol'].items()
        if k not in ['seed','initial_state_sha256']}
    if any(kernel(d)!=kernel(data[0]) for d in data):
        raise ValueError('Different frozen sampler kernels across shards')
    for p,d,protocol in zip(paths,data,protocols):
        if sha(p/'reference.pt')!=protocol['reference_sha256']:
            raise ValueError('Reference hash differs')
        if d['white'].shape!=(1,args.draws,329) or d['completed_warmup']!=protocol['warmup'] or d['retained']!=args.draws:
            raise ValueError('Incomplete budgets')
        if d['protocol']['seed']!=protocol['seed'] or d['protocol']['chains']!=1:
            raise ValueError('Sampler/shard protocol differs')
        if (d['protocol']['warmup']!=protocol['warmup'] or
                d['protocol']['min_time']!=protocol['min_time'] or
                d['protocol']['max_time']!=protocol['max_time']):
            raise ValueError('Saved kernel settings differ from the frozen run')
    gate=json.loads(Path(args.gate).read_text());model=load_model(args.inputs,gate)
    reference=torch.load(paths[0]/'reference.pt',weights_only=True)
    if reference['frozen_model_specification']!=model.specification():
        raise ValueError('Physical model differs')
    target=make_target(model,reference)
    for i,d in enumerate(data):
        expected=hashlib.sha256(reference['initial_white'][i:i+1].contiguous().numpy().tobytes()).hexdigest()
        if expected!=d['protocol']['initial_state_sha256']:
            raise ValueError('Independent starting state differs')
    white=torch.cat([d['white'] for d in data]);ratios=torch.cat([d['log_ratio'] for d in data])
    proof=replay(model,target,{'white':white,'log_ratio':ratios})
    native=target.coordinates(white.flatten(0,1));physical=target.decode(white.flatten(0,1))
    fm,fc,sm,sc,normalizers=force_moments_stream(model,physical)
    density=ratios-.5*white.square().sum(-1)-target.center
    values=torch.cat((physical.reshape(4,args.draws,-1),native.reshape(4,args.draws,-1),
        density[...,None],fm.reshape(4,args.draws,-1),normalizers.reshape(4,args.draws,2)),-1)
    names=[f'physical_{i}' for i in range(model.dimension)]+[f'native_{i}' for i in range(model.dimension)]+['log_target']+[
        f'conditional_load_{i}' for i in range(model.force_dimension)]+['logdet_Q','logdet_A']
    diagnostic=diagnose_quantities(values,names)
    invalid=sum(sum(d['counts']['retained']['invalid']) for d in data)
    divergent=sum(sum(d['counts']['retained']['divergent']) for d in data)
    passed=(args.draws>=2000 and protocols[0]['warmup']>=500 and
        diagnostic['all_monitored_quantities_pass'] and invalid==0 and divergent==0)
    quantities=diagnostic['quantities']
    result={'all_monitored_quantities_pass':bool(passed),'draws_per_chain':args.draws,
        'warmup':protocols[0]['warmup'],'monitored_quantities':len(names),'failed_quantities':len(diagnostic['failed_quantities']),
        'worst_rhat':max(v['rank_folded_split_rhat'] or 999 for v in quantities.values()),
        'minimum_bulk_ess':min(v['bulk_ess'] or 0 for v in quantities.values()),
        'minimum_tail_ess':min(v['tail_ess'] or 0 for v in quantities.values()),
        'retained_invalid':invalid,'retained_divergent':divergent,'density_replay':proof,
        'four_independent_chains':True,'reference_sha256':protocols[0]['reference_sha256'],
        'source_protocol':common(protocols[0]),'devices':[p['device'] for p in protocols],
        'future_geometry_used':False,'valid_for_ranking':False}
    write(out/'diagnostics.json',diagnostic);write(out/'convergence.json',result)
    atomic_save(out/'merged.pt',{'white':white,'log_ratio':ratios})
    print(json.dumps(result),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run','aggregate'])
    parser.add_argument('--inputs',default='runs/inputs')
    parser.add_argument('--gate',default='runs/bird-8000/convergence.json')
    parser.add_argument('--reference',default='runs/reference-original/reference.pt')
    parser.add_argument('--output',required=True)
    parser.add_argument('--shards')
    parser.add_argument('--shrinkage',type=float,default=.02)
    parser.add_argument('--ridge',type=float,default=.001)
    parser.add_argument('--chain-index',type=int,choices=range(4),default=0)
    parser.add_argument('--seed',type=int,default=1070009)
    parser.add_argument('--warmup',type=int,default=500)
    parser.add_argument('--draws',type=int,default=2000)
    parser.add_argument('--min-time',type=float,default=1.)
    parser.add_argument('--max-time',type=float,default=4.)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--threads',type=int,default=2)
    args=parser.parse_args()
    torch.set_num_threads(args.threads)
    import torch.backends.opt_einsum as backend
    backend.enabled=True;backend.strategy='greedy'
    globals()[args.action](args)


if __name__=='__main__':
    main()
