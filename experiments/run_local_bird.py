"""Bounded local four-GPU Bird execution; never dispatch the wider comparison."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def write(path,value):
    temporary=Path(path).with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def run(args):
    root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=True)
    lock=(root/'coordinator.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    config=json.loads(Path(args.config).read_text())
    proof=json.loads((ROOT/'examples/bird-local-sampler-audit/actual-gpu-recovery-proof.json').read_text())
    if not proof.get('all_pass') or not proof.get('deterministic_algorithms'):
        raise ValueError('Actual deterministic GPU recovery proof is required')
    settings=config['production'];shards=root/'shards'
    metadata={'config':config,'reference':str(Path(args.reference).resolve()),
        'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'pid':os.getpid(),'status':'starting','started_unix_s':time.time(),
        'larger_comparison_dispatched':False,'attempts':[]}
    previous=root/'execution.json'
    if previous.exists():
        old=json.loads(previous.read_text())
        if old['config']!=config or old['reference']!=metadata['reference']:
            raise ValueError('Local execution settings changed')
        metadata['attempts']=old['attempts']
    for budget in settings['draw_budgets']:
        report=root/f'diagnostics-{budget}'/'convergence.json'
        if report.exists():
            gate=json.loads(report.read_text())
            if gate['all_monitored_quantities_pass']:
                break
            continue
        metadata.update(status='sampling',draws_per_chain_requested=budget)
        processes=[];logs=[];attempt={'budget':budget,'started_unix_s':time.time()}
        metadata['attempts'].append(attempt);write(previous,metadata)
        for index,(seed,device) in enumerate(zip(settings['seeds'],settings['devices'])):
            command=[sys.executable,'-u','experiments/bird_local.py','run',
                '--reference',args.reference,'--inputs',args.inputs,'--gate',args.gate,
                '--output',str(shards/f'chain-{index}'),'--chain-index',str(index),
                '--seed',str(seed),'--device',device,'--warmup',str(settings['warmup']),
                '--draws',str(budget),'--min-time',str(settings['min_time']),
                '--max-time',str(settings['max_time']),
                '--threads',str(settings['torch_threads_per_chain'])]
            log=(root/f'chain-{index}.log').open('a');logs.append(log)
            processes.append(subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT))
        metadata['worker_pids']=[p.pid for p in processes];write(previous,metadata)
        codes=[p.wait() for p in processes]
        for log in logs:log.close()
        attempt.update(finished_unix_s=time.time(),worker_exit_codes=codes)
        if any(codes):
            metadata['status']='worker_failure_checkpoints_preserved';write(previous,metadata)
            raise RuntimeError('A local worker failed; inspect logs before exact recovery')
        metadata['status']='replaying_observed_convergence';write(previous,metadata)
        with (root/f'aggregate-{budget}.log').open('w') as log:
            subprocess.run([sys.executable,'-u','experiments/bird_local.py','aggregate',
                '--inputs',args.inputs,'--gate',args.gate,'--shards',str(shards),
                '--output',str(report.parent),'--draws',str(budget)],check=True,
                stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
        gate=json.loads(report.read_text())
        attempt['convergence']={k:gate[k] for k in ['all_monitored_quantities_pass',
            'failed_quantities','worst_rhat','minimum_bulk_ess','minimum_tail_ess',
            'retained_invalid','retained_divergent']}
        write(previous,metadata)
        print(json.dumps({'budget':budget,**attempt['convergence']}),flush=True)
        if gate['all_monitored_quantities_pass']:
            break
    else:
        metadata['status']='convergence_failed_at_declared_cap'
        write(previous,metadata)
        return
    metadata['status']='independent_uncertainty_validation';write(previous,metadata)
    with (root/'validation.log').open('w') as log:
        subprocess.run([sys.executable,'-u','experiments/validate_local_bird.py',
            '--inputs',args.inputs,'--gate',args.gate,'--shards',str(shards),
            '--draws',str(gate['draws_per_chain']),'--baseline',args.baseline,
            '--output',str(root/'validation')],check=True,stdout=log,
            stderr=subprocess.STDOUT,cwd=ROOT)
    result=json.loads((root/'validation/uncertainty-validation.json').read_text())
    metadata.update(status='finished',readiness=result['readiness'],finished_unix_s=time.time())
    write(previous,metadata);print(json.dumps(result['readiness']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/bird_local_preconditioned.json')
    parser.add_argument('--reference',default='runs/bird-preconditioned-reference/reference.pt')
    parser.add_argument('--inputs',default='runs/inputs')
    parser.add_argument('--gate',default='runs/bird-8000/convergence.json')
    parser.add_argument('--baseline',default='runs/baseline')
    parser.add_argument('--output',default='runs/bird-local-production')
    run(parser.parse_args())
