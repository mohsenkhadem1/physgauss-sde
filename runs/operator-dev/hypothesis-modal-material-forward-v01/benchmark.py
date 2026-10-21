"""One predeclared32-candidate CPU/GPU0 query, no observations or posterior."""
import hashlib,json,sys,time
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

torch.set_num_threads(2)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
destination=Path(__file__).parent/'benchmark.json'
if destination.exists():raise ValueError('Fresh benchmark report required')
design=json.loads((ROOT/'runs/gaussian-campaign-v01/hypothesis-inputs-v01/manifest.json').read_text())
public=next(h for h in design['public_hardware'] if h['id']=='H0')
template=next(h['spec'] for h in design['hypotheses'] if h['id']=='H0__m0__bc0')
prior=MaterialMarginalPrior(make_prior(public));state=prior.sample(32,np.random.default_rng(2026102103))
physical=prior.physical_coordinates(state);theta=torch.tensor(physical,dtype=torch.float64)
started=time.monotonic();full=LinearHypothesisDynamics.from_hypothesis(known_hardware_from_public(public),template)
modal=complete_static_lift(FixedModalLinearDynamics(full,96));construction=time.monotonic()-started
model=BatchedModalMaterialForward(modal);clock=make_exciting_commands('exciting_world_a_v01')
timings={};results={}
for device in ['cpu','cuda:0']:
    if device.startswith('cuda'):torch.cuda.set_device(0);torch.cuda.synchronize(0)
    start=time.monotonic();model._prepare(clock,device)
    if device.startswith('cuda'):torch.cuda.synchronize(0)
    prepare=time.monotonic()-start
    with torch.no_grad():
        start=time.monotonic();result=model.predict(clock,theta,device=device)
        if device.startswith('cuda'):torch.cuda.synchronize(0)
        elapsed=time.monotonic()-start
    results[device]=result['positions_m'].cpu().numpy()
    timings[device]={'theta_independent_prepare_s':prepare,'single_B32_forward_s':elapsed,
        'per_candidate_s':elapsed/32,'candidate_queries_per_second':32/elapsed}
error=results['cpu']-results['cuda:0'];maximum=float(np.max(abs(error)));rmse=float(np.sqrt(np.mean(error**2)))
report={'status':'COMPLETED_B32_CPU_GPU0_PROPER_MARGINAL_QUERY','seed':2026102103,'batch':32,'geometry_template':'H0__m0__bc0',
    'known_program':'exciting_world_a_v01','private_hz':120,'vibration_modes':96,'enriched_modes':modal.mode_count,
    'basis_construction_s':construction,'timings':timings,'CPU_GPU_max_absolute_coordinate_difference_m':maximum,
    'CPU_GPU_coordinate_RMSE_m':rmse,'parity_tolerance_m':1e-10,'parity_passed':maximum<=1e-10,
    'proper_marginal_metadata':prior.metadata(),'physical_candidate_coordinates_E_rho_nu_drag':physical.tolist(),
    'sources_sha256':{p:sha(ROOT/p) for p in ['experiments/campaign_modal_material_forward.py','experiments/campaign_modal_static_lift.py',
        'experiments/campaign_modal_hypothesis.py','experiments/campaign_material_marginal.py','experiments/campaign_linear_hypothesis.py']},
    'benchmark_source_sha256':sha(__file__),'cpu_threads':2,'GPU_used_only':0,'GPU_name':torch.cuda.get_device_name(0),
    'finite_GT_or_evaluation64_accessed':False,'posterior_launched':False,'calibration_claim':False,
    'limitation':'Known-support linear modal query throughput only. One batch perdevice withfixedgeometry/command preparation measured separately; no posterior or sensor likelihood.'}
destination.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:report[k] for k in ['basis_construction_s','timings','CPU_GPU_max_absolute_coordinate_difference_m','CPU_GPU_coordinate_RMSE_m','parity_passed']}),flush=True)
