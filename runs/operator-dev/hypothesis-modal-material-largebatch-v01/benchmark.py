"""Authorized B256 CPU/GPU0 and B512 GPU0 only; no posterior/observations."""
import gc,hashlib,json,resource,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'experiments'))
from campaign_hypothesis_inputs import known_hardware_from_public,make_prior
from campaign_material_marginal import MaterialMarginalPrior
from campaign_linear_hypothesis import LinearHypothesisDynamics
from campaign_modal_hypothesis import FixedModalLinearDynamics
from campaign_modal_static_lift import complete_static_lift
from campaign_modal_material_forward import BatchedModalMaterialForward
from campaign_exciting_commands import make_exciting_commands

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
directory=Path(__file__).parent;destination=directory/'benchmark.json'
if destination.exists():raise ValueError('Fresh benchmark report required')
torch.set_num_threads(2);torch.cuda.set_device(0)
design_path=ROOT/'runs/gaussian-campaign-v01/hypothesis-inputs-v01/manifest.json'
design=json.loads(design_path.read_text());public=next(h for h in design['public_hardware'] if h['id']=='H0')
template=next(h['spec'] for h in design['hypotheses'] if h['id']=='H0__m0__bc0')
source_names=['experiments/campaign_modal_material_forward.py','experiments/campaign_modal_static_lift.py',
    'experiments/campaign_modal_hypothesis.py','experiments/campaign_material_marginal.py','experiments/campaign_linear_hypothesis.py']
pins={name:sha(ROOT/name) for name in source_names}
protocol={'protocol':'modal-proper-material-largebatch-throughput-v0.1','seed':2026102104,'draws':512,
    'draw_law':'MaterialMarginalPrior(make_prior(publicH0)); freshproper draws, no observation or response conditioning.',
    'calls':[{'batch':256,'device':'cpu','indices':[0,256]},{'batch':256,'device':'cuda:0','indices':[0,256]},
        {'batch':512,'device':'cuda:0','indices':[0,512]}],
    'parity':'CPU256 vsGPU256 andCPU256 vsGPU512 first256; latter256GPU512 have finite/domain checks only.',
    'cpu_threads':2,'GPU_used_only':0,'template':'H0__m0__bc0','known_program':'exciting_world_a_v01',
    'fixed_basis':'96nominal modes plus3canonical static-lift shapes; no dimension adaptation.',
    'source_sha256':pins,'benchmark_source_sha256':sha(__file__),'input_manifest_sha256':sha(design_path),
    'B32_negative_preserved':True,'finite_truth_used':False,'posterior_launched':False}
protocol_path=directory/'protocol.json'
if protocol_path.exists():raise ValueError('Fresh benchmark protocol required')
protocol_path.write_text(json.dumps(protocol,indent=2)+'\n')
prior=MaterialMarginalPrior(make_prior(public));state=prior.sample(512,np.random.default_rng(protocol['seed']))
physical=prior.physical_coordinates(state);theta=torch.tensor(physical,dtype=torch.float64)
started=time.monotonic();full=LinearHypothesisDynamics.from_hypothesis(known_hardware_from_public(public),template)
modal=complete_static_lift(FixedModalLinearDynamics(full,96));construction=time.monotonic()-started
model=BatchedModalMaterialForward(modal);clock=make_exciting_commands('exciting_world_a_v01');timings=[];cpu=None;parity=[]
for device,batch in [('cpu',256),('cuda:0',256),('cuda:0',512)]:
    if device.startswith('cuda'):torch.cuda.synchronize(0)
    start=time.monotonic();model._prepare(clock,device)
    if device.startswith('cuda'):torch.cuda.synchronize(0);torch.cuda.reset_peak_memory_stats(0)
    preparation=time.monotonic()-start
    with torch.no_grad():
        start=time.monotonic();result=model.predict(clock,theta[:batch],device=device)
        if device.startswith('cuda'):torch.cuda.synchronize(0)
        elapsed=time.monotonic()-start
    row={'batch':batch,'device':device,'theta_independent_prepare_s':preparation,'forward_s':elapsed,
        'queries_per_second':batch/elapsed,'per_query_s':elapsed/batch,
        'process_peak_RSS_bytes_cumulative_includes_construction':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated(0) if device.startswith('cuda') else None,
        'GPU_peak_reserved_bytes':torch.cuda.max_memory_reserved(0) if device.startswith('cuda') else None}
    if not bool(torch.isfinite(result['positions_m']).all()):raise ValueError('Nonfinite positions')
    if device=='cpu':cpu=result['positions_m'].numpy().copy()
    else:
        reference=result['positions_m'][:256].cpu().numpy();error=reference-cpu
        parity.append({'GPU_batch':batch,'compared_candidates':256,'CPU_reference_batch':256,
            'maximum_absolute_coordinate_difference_m':float(np.max(abs(error))),
            'coordinate_RMSE_m':float(np.sqrt(np.mean(error**2))),'passed':bool(np.max(abs(error))<=1e-10)})
        del reference,error
    timings.append(row);print(json.dumps(row),flush=True);del result;gc.collect()
    if device.startswith('cuda'):torch.cuda.empty_cache()
if pins!={name:sha(ROOT/name) for name in source_names}:raise ValueError('Frozen source changed')
report={'status':'COMPLETED_AUTHORIZED_LARGEBATCH_ONLY','protocol':{'path':str(protocol_path.relative_to(ROOT)),'sha256':sha(protocol_path)},
    'construction_s':construction,'timings':timings,'parity':parity,'GPU_name':torch.cuda.get_device_name(0),
    'fresh_material_draw_intrinsic_coordinates_sha256':hashlib.sha256(state.coordinates.tobytes()).hexdigest(),
    'fresh_material_draw_physical_coordinates_sha256':hashlib.sha256(physical.tobytes()).hexdigest(),
    'proper_material_prior':prior.metadata(),'all512_domain_and_finite_checks_passed':True,
    'B512_independent_CPU_parity_scope':'First256only; noCPU512forward executed.',
    'finite_truth_or_posterior_used':False,'memory_note':'GPU peaks include cachedgeometry buffers and outputstorage. CPUpeakRSS cumulative includes construction and cachedCPU256reference.',
    'importance_reference_launched':False}
destination.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({'parity':parity}),flush=True)
