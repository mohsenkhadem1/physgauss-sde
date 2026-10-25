"""Synthetic CPU callback timing/parity; no posterior qualification claim."""
import hashlib
import json
import os
from pathlib import Path
import resource
import time
import numpy as np
from experiments.campaign_nonlinear_residual_atlas_v01 import NonlinearResidualAtlas
from experiments.campaign_nonlinear_residual_atlas_fast_v01 import FastResidualAtlas


def main():
    if os.sched_getaffinity(0) != {18} or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU18/noGPU only')
    if any(os.environ.get(k) != '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')):
        raise ValueError('One thread required')
    rng = np.random.default_rng(9044)
    labels = [f'synthetic-{j}' for j in range(10)]
    base = NonlinearResidualAtlas(rng.normal(size=(9, 9, 10, 24))*1e-5, labels,
                                  provenance_fingerprint='synthetic-CPU-callback-cost')
    fast = FastResidualAtlas(base)
    scalar_queries = rng.random((3000, 2))
    batch_queries = rng.random((16384, 2))
    results = {}
    for name, queries, selected in [('scalar', scalar_queries, labels[:1]),
                                     ('batch', batch_queries, labels)]:
        timings = {}
        outputs = {}
        for variant, atlas in [('original', base), ('cached_constants_matmul', fast)]:
            repetitions = []
            for _ in range(3):
                start = time.perf_counter()
                if name == 'scalar':
                    values = np.concatenate([atlas.predict(q[None], selected) for q in queries])
                else:
                    values = atlas.predict(queries, selected, chunk=1024)
                repetitions.append(time.perf_counter()-start)
            timings[variant] = float(np.median(repetitions))
            outputs[variant] = values
        error = float(abs(outputs['original']-outputs['cached_constants_matmul']).max())
        if error > 2e-19:
            raise ValueError('Polynomial arithmetic parity failure')
        results[name] = dict(queries=len(queries), labels=len(selected), channels=24,
            median_seconds=timings, speed_ratio=timings['original']/timings['cached_constants_matmul'],
            maximum_absolute_difference_m=error)
    root = Path(__file__).resolve().parents[1]
    out = root/'runs/operator-dev/nonlinear-atlas-fast-CPU-control-v01'
    out.mkdir(exist_ok=False)
    report = dict(passed=True, synthetic_only=True, pid=os.getpid(), CPU=18,
        grid=9, repetitions=3, results=results,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        source_SHA256={p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in (
            'experiments/campaign_nonlinear_residual_atlas_fast_v01.py',
            'tests/test_nonlinear_residual_atlas_fast_v01.py',
            'experiments/benchmark_nonlinear_atlas_cpu_v01.py')},
        scope='Callback timing only; no complete-posterior speedup, model or calibration claim')
    (out/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
