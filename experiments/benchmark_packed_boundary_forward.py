"""One prospective mixed55-category CPU/GPU parity/throughput query.

No observations, likelihoods or truth. Geometry IDs cycle through ALL55; this
timing schedule is not a draw from support-prior probabilities. GPU512 is tried
once; only CUDA OOM permits a retained-failure GPU256 fallback.
"""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
try:
    from campaign_support_basis_cache import load_support_basis_cache,sha,read,resolve,dump
    from campaign_exciting_commands import make_exciting_commands
except ImportError:
    from experiments.campaign_support_basis_cache import load_support_basis_cache,sha,read,resolve,dump
    from experiments.campaign_exciting_commands import make_exciting_commands


def run(protocol_path,protocol_sha256):
    path=resolve(protocol_path)
    if sha(path)!=protocol_sha256:raise ValueError('Frozen benchmark protocol changed')
    p=read(path)
    if p['particles']!=512 or p['fallback_particles']!=256 or p['CPU_threads']!=2 or p['physical_GPU']!=2 or os.environ.get('CUDA_VISIBLE_DEVICES')!='2':raise ValueError('Exact declared512/GPU2 scope required')
    if any(sha(resolve(f))!=h for f,h in p['source_sha256'].items()):raise ValueError('Frozen benchmark source changed')
    output=resolve(p['output']);output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    status=dict(pid=os.getpid(),terminal=False,completed=False,posterior_or_truth_inputs=False);dump(output/'execution.json',status)
    try:
        start=time.monotonic();forward,public,ids,distribution=load_support_basis_cache(p['cache']['path'],p['cache']['sha256']);load_time=time.monotonic()-start
        rng=np.random.default_rng(p['seed']);count=512
        theta=np.stack([np.exp(rng.uniform(np.log(1500),np.log(20000),count)),rng.uniform(400,850,count),rng.uniform(.28,.42,count),rng.uniform(.35,.85,count)],1)
        geometry=np.resize(np.array(ids,dtype=np.int64),count)
        np.savez_compressed(output/'known-query-inputs.npz',geometry_ids=geometry,physical_coordinates=theta)
        clock=make_exciting_commands('exciting_world_a_v01')
        with torch.no_grad():
            start=time.monotonic();cpu=forward.predict(clock,torch.tensor(geometry),torch.tensor(theta),device='cpu');cpu_time=time.monotonic()-start
            start=time.monotonic();torch.cuda.set_device(0);torch.cuda.synchronize();context_time=time.monotonic()-start
            memory_before=torch.cuda.mem_get_info();torch.cuda.reset_peak_memory_stats();tried=[]
            try:
                start=time.monotonic();gpu=forward.predict(clock,torch.tensor(geometry),torch.tensor(theta),device='cuda:0');torch.cuda.synchronize();gpu_time=time.monotonic()-start;used=512
                tried.append(dict(particles=512,completed=True))
            except torch.OutOfMemoryError as error:
                tried.append(dict(particles=512,completed=False,error=str(error)));dump(output/'gpu512-failure.json',tried[-1])
                gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
                start=time.monotonic();gpu=forward.predict(clock,torch.tensor(geometry[:256]),torch.tensor(theta[:256]),device='cuda:0');torch.cuda.synchronize();gpu_time=time.monotonic()-start;used=256
                tried.append(dict(particles=256,completed=True))
            start=time.monotonic();differences={name:float((value.cpu()-cpu[name][:used]).abs().max()) for name,value in gpu.items() if name in ['positions_m','handle_centers_m','handle_rotation_coordinates_rad']};transfer_compare_time=time.monotonic()-start
        passed=all(differences[k]<=p['parity_max_abs'][k] for k in differences)
        result=dict(completed=True,parity_passed=passed,particles_used=used,geometries_preserved=len(ids),mode_groups=forward.catalogue()['groups'],
            cache_load_hash_validation_seconds=load_time,CPU512_cold_query_seconds=cpu_time,
            GPU_context_seconds=context_time,GPU_cold_query_seconds=gpu_time,GPU_to_CPU_transfer_and_compare_seconds=transfer_compare_time,
            timings_include_geometry_packing_and_input_transfers=True,timing_repetitions=1,CPU_GPU_same_particle_count=(used==512),
            differences_max_abs=differences,attempts=tried,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(),GPU_free_total_before_bytes=list(memory_before),
            GPU_free_total_after_bytes=list(torch.cuda.mem_get_info()),GPU_device_name=torch.cuda.get_device_name(0),
            torch_version=torch.__version__,CUDA_version=torch.version.cuda,CPU_threads=2,physical_GPU=2,
            support_prior_sampling_performed=False,observations_truth_or_likelihood_used=False,posterior_performed=False,
            scope='One fixed mixed-category query; not large-scale scaling or posterior accuracy evidence')
        dump(output/'summary.json',result)
        status.update(terminal=True,completed=passed);dump(output/'execution.json',status)
        if not passed:raise RuntimeError('Predeclared CPU/GPU parity failed')
        return result
    except BaseException as error:
        status.update(terminal=True,completed=False,error=type(error).__name__,message=str(error));dump(output/'execution.json',status);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',required=True);parser.add_argument('--protocol-sha256',required=True)
    args=parser.parse_args();print(json.dumps(run(args.protocol,args.protocol_sha256)),flush=True)
