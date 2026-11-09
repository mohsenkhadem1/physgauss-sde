"""Chain the GIC stages without the interactive session (run as a systemd user unit).
1. wait for D0 fit -> D2 forecast from D0 (GPU 0)
2. per world: wait for D1 appearance, write D1 config (init from D0 fit), run native physics refit
   with the D1 force adapter, then D2 forecast from the refit.
Every step is idempotent: completed outputs are skipped. Status in runs/stem-trial-v01/supervisor.json.
"""
import json, math, os, subprocess, time
from pathlib import Path
R = Path(__file__).resolve().parents[2]
T = R/'runs/stem-trial-v01'; G = T/'gic'; L = T/'logs'
V = R/'runs/gap-crossmethod-v02/vendor/gic'; PY = str(R/'runs/gap-crossmethod-v02/gic-env/bin/python')
S = R/'scripts/stem_trial_v01'
status = {}


def log(k, v):
    status[k] = v; status['updated'] = time.strftime('%F %T')
    (T/'supervisor.json').write_text(json.dumps(status, indent=1))


def run(cmd, gpu, logf, extra=None):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4')
    env.update(extra or {})
    with open(logf, 'ab') as f:
        return subprocess.Popen(cmd, cwd=V, env=env, stdout=f, stderr=subprocess.STDOUT)


def wait_file(p, proc=None):
    while not Path(p).exists():
        if proc is not None and proc.poll() is not None and not Path(p).exists():
            return False
        time.sleep(30)
    return True


def forecast(model, out, gpu, logf):
    if out.exists():
        return 0
    p = run([PY, '-u', str(S/'gic_stem_predict.py'), str(model), str(out)], gpu, logf,
            {'STEM_FORCE_JSON': str(G/'force-D2.json'), 'STEM_FORCE_AUDIT': str(out.with_suffix('.audit.json'))})
    return p.wait()


def main():
    log('stage', 'waiting for D0 fit')
    wait_file(G/'D0/0-pred.json')
    d0 = json.loads((G/'D0/0-pred.json').read_text())
    log('D0_fit', d0['mat_params'] | {'vel': d0['vel']})
    rc = forecast(G/'D0', G/'forecast-D0.npz', 0, L/'gic-forecast-D0.log'); log('forecast-D0', rc)
    procs = {}
    for gpu, w in zip([1, 2, 3], ['W500', 'W1000', 'W2000']):
        wait_file(G/f'D1-{w}/point_cloud/iteration_40000')
        cfg = G/f'config-D1-{w}.json'
        if not (G/f'D1-{w}/0-pred.json').exists():
            subprocess.run([PY, str(S/'gic_config.py'), 'D1', str(cfg), str(math.log10(d0['mat_params']['E'])),
                            str(d0['mat_params']['nu'])], check=True)
            procs[w] = run([PY, '-u', str(S/'gic_stem_entry.py'), '-c', str(cfg), '-s', str(T/f'scenes/D1-{w}'),
                            '-m', str(G/f'D1-{w}'), '--reg_scale', '--reg_alpha'], gpu, L/f'gic-D1-{w}-physics.log',
                           {'STEM_FRAMES': '37', 'STEM_TI_MEM': '0.18', 'STEM_FORCE_JSON': str(G/'force-D1.json'),
                            'STEM_FORCE_AUDIT': str(G/f'D1-{w}-force-audit.json')})
            log(f'D1-{w}', 'physics running')
    for gpu, w in zip([1, 2, 3], ['W500', 'W1000', 'W2000']):
        ok = wait_file(G/f'D1-{w}/0-pred.json', procs.get(w))
        if not ok:
            log(f'D1-{w}', f'FAILED rc={procs[w].returncode}'); continue
        log(f'D1-{w}', json.loads((G/f'D1-{w}/0-pred.json').read_text())['mat_params'])
        rc = forecast(G/f'D1-{w}', G/f'forecast-D1-{w}.npz', gpu, L/f'gic-forecast-D1-{w}.log'); log(f'forecast-D1-{w}', rc)
    log('stage', 'done')


if __name__ == '__main__':
    main()
