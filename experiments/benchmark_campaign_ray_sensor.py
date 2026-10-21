"""Small reproducible throughput/correctness diagnostic on one allowed TRAIN case.

No control, calibration-audit, validation or held-out trajectory is opened.
CUDA is opt-in; tensor peak allocation is reported separately from driver context.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT', '1')
import numpy as np
import torch

try:
    from campaign_ray_sensor import TorchMeshRaySensor
    from gaussian_acquisition_sensor import public_ray_queries
    from validate_isaac_dataset import ray_depth
except ImportError:
    from experiments.campaign_ray_sensor import TorchMeshRaySensor
    from experiments.gaussian_acquisition_sensor import public_ray_queries
    from experiments.validate_isaac_dataset import ray_depth


def benchmark(output, gpu=None, particles=128, repeats=3):
    torch.set_num_threads(2)
    episode=Path('runs/gaussian-boundary-neural-v01/episodes/t0000')
    summary=json.loads((episode/'summary.json').read_text())
    if summary['split']!='train':raise ValueError('Only this declared TRAIN case is allowed')
    with np.load(episode/'reference.npz') as z:reference={k:z[k] for k in z.files}
    with np.load(episode/'observations/cameras.npz') as z:
        pose=z['camera_to_world_usd_row'][0];k=z['K']
    frames=[1,10,20,40,60]
    with np.load(episode/'truth/trajectory.npz') as z:q=z['positions_m'][frames]
    query=public_ray_queries(reference,pose,k,count=24)
    positions=np.tile(q,(particles,1,1))
    times=[]
    for _ in range(repeats):
        start=time.perf_counter()
        expected=np.stack([ray_depth(p,reference['surface_faces'],pose,k,query['xs'],query['ys']) for p in positions])
        times.append(time.perf_counter()-start)
    numpy_seconds=float(np.median(times))
    records=[]
    for device in ['cpu']+([] if gpu is None else [f'cuda:{gpu}']):
        for dtype in [torch.float64,torch.float32]:
            chunks={} if device=='cpu' else {'particle_chunk':64,'ray_chunk':24,'triangle_chunk':576}
            sensor=TorchMeshRaySensor(reference['surface_faces'],query,device=device,dtype=dtype,**chunks)
            states=torch.as_tensor(positions,device=device,dtype=dtype)
            sensor(states[:16])
            if device!='cpu':
                torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
            durations=[]
            for _ in range(repeats):
                start=time.perf_counter();actual=sensor(states)
                if device!='cpu':torch.cuda.synchronize(device)
                durations.append(time.perf_counter()-start)
            predicted=actual.depth_m.detach().cpu().numpy();visible=actual.visible.cpu().numpy()
            jointly_visible=visible&np.isfinite(expected)
            elapsed=float(np.median(durations))
            records.append({'device':device,'dtype':str(dtype),'median_seconds':elapsed,
                'chunks':{'particle_chunk':16,'ray_chunk':64,'triangle_chunk':128,**chunks},
                'numpy_loop_speedup':numpy_seconds/elapsed,
                'mesh_states_per_second':len(positions)/elapsed,
                'hit_mask_mismatches':int(np.sum(visible!=np.isfinite(expected))),
                'maximum_joint_hit_depth_error_m':float(np.max(np.abs(predicted[jointly_visible]-expected[jointly_visible]))),
                'peak_cuda_tensor_allocation_bytes':None if device=='cpu' else torch.cuda.max_memory_allocated(device)})
            del sensor,states,actual
            if device!='cpu':torch.cuda.empty_cache()
    result={'episode':str(episode),'split':'train','selected_frames':frames,
        'particles':particles,'states':len(positions),'rays_per_state':24,
        'surface_triangles':len(reference['surface_faces']),'repeats':repeats,
        'cpu_torch_threads':2,'numpy_loop_median_seconds':numpy_seconds,
        'dtype_reference':'numpy float64','timing_includes':'sensor forward; input device transfer excluded, likelihood conversion excluded',
        'qualification':'Repeated one TRAIN case measures throughput/numerical equivalence, not physical accuracy or generalization. CUDA tensor peaks exclude driver/runtime context. Triangle-edge/grazing classification remains numerically ill conditioned.',
        'source_sha256':hashlib.sha256(Path(__file__).with_name('campaign_ray_sensor.py').read_bytes()).hexdigest(),
        'results':records}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--gpu',type=int)
    p.add_argument('--particles',type=int,default=128);p.add_argument('--repeats',type=int,default=3)
    a=p.parse_args();print(json.dumps(benchmark(a.output,a.gpu,a.particles,a.repeats),indent=2))
