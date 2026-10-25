"""Bounded, separate profiling of one independent integral; no qualification.

Uses the pilot's saved observation and admitted atlas. No simulation, RNG,
parameter changes or writes to existing runs. Profiling adds timing overhead.
"""
import cProfile
import hashlib
import json
import os
from pathlib import Path
import pstats
import resource
import signal
import threading
import time
import numpy as np
from experiments import run_surface_nonlinear_posterior_cost_v02 as cost


def main():
    if os.sched_getaffinity(0) != {18} or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU18/noGPU profiling only')
    if any(os.environ.get(k) != '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')):
        raise ValueError('One BLAS thread')
    digest = 'ee27581ec076dbc611e25e8ea3506423ebf1717b812af9087a85008198038b8e'
    protocol, _ = cost.validate(digest)
    cost.release_gate(protocol, digest)
    out = cost.ROOT/'runs/operator-dev/surface-oracle-profile-v01'
    out.mkdir(exist_ok=False)
    state = dict(completed=False, pid=os.getpid(), cost_protocol_SHA256=digest,
                 source_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 profiling_only=True, new_RNG_calls=0, new_PDE_calls=0)
    start = time.monotonic()
    def stop(*_):
        raise TimeoutError('Bounded profiling deadline/RSS limit')
    signal.signal(signal.SIGALRM, stop)
    signal.signal(signal.SIGTERM, stop)
    signal.alarm(120)
    done = threading.Event()
    def watch():
        while not done.wait(.25):
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > 4*1024**3:
                os.kill(os.getpid(), signal.SIGTERM)
                return
    threading.Thread(target=watch, daemon=True).start()
    profiler = cProfile.Profile()
    try:
        from experiments.campaign_nonlinear_residual_atlas_v01 import NonlinearResidualAtlas
        from experiments.campaign_surface_p2_static_v02 import SparseStaticForward, point_query_matrix
        from experiments.campaign_surface_p2_force_v03 import load_catalog
        from experiments.campaign_surface_p2_study_design_v01 import observation_points
        from experiments.campaign_surface_p2_inference_v01 import LabelledModal
        from experiments.campaign_surface_conditional_amplitude_v01 import RatioModal, ExactRatioMemo
        from experiments.campaign_surface_response_cache_v01 import model_fingerprint
        from experiments.campaign_nonlinear_conditional_reference_v02 import NonlinearConditionalReferenceV02
        from experiments.campaign_nonlinear_adaptive_oracle_v01 import nonlinear_adaptive_integral
        with np.load(cost.ROOT/protocol['origin_v01']['full_atlas']['archive']['path'], allow_pickle=False) as z:
            atlas = NonlinearResidualAtlas(z['delta'], tuple(str(v) for v in z['labels']),
                provenance_fingerprint=protocol['full_protocol_SHA256'])
        with np.load(cost.ROOT/protocol['origin_v01']['observation']['path'], allow_pickle=False) as z:
            observation = z['observation_m'].copy()
        fwd = SparseStaticForward(4)
        linear = LabelledModal.from_forward(fwd, loads=load_catalog(4), points=observation_points(),
                                            condensed=True, query_matrix=point_query_matrix)
        ratio = RatioModal(linear.modal, linear.labels)
        memo = ExactRatioMemo(lambda z, labels: ratio(np.exp(z), labels),
            model_fingerprint=model_fingerprint(linear), maximum_bytes=256*1024**2, minimum_rows=1)
        reference = NonlinearConditionalReferenceV02(memo, None, [(protocol['prefix'], observation)],
            atlas.labels, residual_predict=atlas.predict, retain_hypotheses=False, evaluation_chunk=1024)
        state['construction_s'] = time.monotonic()-start
        signal.alarm(90)
        profiler.enable()
        result = nonlinear_adaptive_integral(reference.full_loglik, reference._observed_ratio,
            reference.y, sigma=reference.sigma, epsrel=1e-6)
        profiler.disable()
        state.update(completed=True, integral=result)
    except BaseException as error:
        profiler.disable()
        state['stop'] = dict(type=type(error).__name__, message=str(error))
    finally:
        done.set()
        signal.alarm(0)
        profiler.dump_stats(str(out/'profile-LOCAL.pstats'))
        with (out/'hotspots.txt').open('x') as stream:
            pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats('cumulative').print_stats(25)
        state.update(wall_s=time.monotonic()-start,
                     peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        (out/'summary.json').write_text(json.dumps(state, indent=2)+'\n')
    print(json.dumps({k: state[k] for k in ('completed', 'wall_s', 'peak_RSS_bytes')}))


if __name__ == '__main__':
    main()
