"""Check preserved prefixes, frozen kernels, RNG continuation and target caches.

This is not reconstruction scoring or a finite/global convergence claim.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from pacnerf_comparison import cases,make_model,Target,write
from inspect_pacnerf_attempts import find
from physgauss.cli import source_fingerprint


@torch.no_grad()
def verify(before,after,inputs,index,expected_additional=None):
    scene,views,noise,force=cases()[index];name=f'{scene}__{views}views__{noise}__{force}'
    roots=[Path(before),Path(after)];chains=[];references=[];settings=[]
    for root in roots:
        chain=find(root/f'chains-{index}',name,'chains.pt')
        reference=find(root/f'chains-{index}',name,'reference.pt')
        record=find(root/f'results-{index}',name,'settings.json')
        if any(p is None for p in [chain,reference,record]):raise ValueError('Missing continuation artifacts')
        chains.append(torch.load(chain,weights_only=True));references.append(reference)
        settings.append(json.loads(record.read_text()))
    if settings[0]!=settings[1]:raise ValueError('Original worker settings changed')
    if settings[0]['source_fingerprint']!=source_fingerprint():raise ValueError('Physical source differs')
    if hashlib.sha256((Path(inputs)/f'{scene}-inputs.pt').read_bytes()).hexdigest()!=settings[0]['input_sha256']:
        raise ValueError('Prepared inputs differ')
    if references[0].read_bytes()!=references[1].read_bytes():raise ValueError('Fitted reference was changed')
    old,new=chains;c,n,d=old['white'].shape;added=new['white'].shape[1]-n
    if added<=0 or new['white'].shape[0]!=c or new['white'].shape[2]!=d:raise ValueError('No valid extended prefix')
    if expected_additional is not None and added!=expected_additional:raise ValueError('Unexpected retained count')
    for key in ['white','log_ratio','energy_errors']:
        if not torch.equal(old[key],new[key][:,:n]):raise ValueError(f'Original {key} prefix changed')
    if old['step_size']!=new['step_size'] or new['config']['warmup']!=0:raise ValueError('Kernel or warmup changed')
    for key in ['chains','seed','min_time','max_time']:
        if old['config'][key]!=new['config'][key]:raise ValueError('Frozen kernel configuration changed')
    if old['counts']['warmup']!=new['counts']['warmup']:raise ValueError('Warmup counts changed')
    for key in ['accepted','invalid','divergent']:
        changes=[b-a for a,b in zip(old['counts']['retained'][key],new['counts']['retained'][key])]
        if any(not 0<=delta<=added for delta in changes):raise ValueError('Invalid retained counts')
    # Reproduce only random-number consumption, independently of all forces,
    # gradients, accept/reject decisions and checkpoint chunk boundaries.
    rng=torch.Generator();rng.set_state(old['rng_state']);split_steps=0
    for _ in range(added):
        duration=old['config']['min_time']+(old['config']['max_time']-old['config']['min_time'])*float(torch.rand((),generator=rng))
        split_steps+=max(1,round(duration/old['step_size']))
        torch.randn(c,d,dtype=old['white'].dtype,generator=rng)
        torch.rand(c,dtype=old['white'].dtype,generator=rng)
    if not torch.equal(rng.get_state(),new['rng_state']):raise ValueError('RNG stream differs from exact continuation')
    boundary_replays=new['gradient_evaluations']-old['gradient_evaluations']-split_steps
    if not 1<=boundary_replays<=max(1,added//16):raise ValueError('Unexpected gradient counter increment')
    model,_=make_model(inputs,index);reference=torch.load(references[0],weights_only=True)
    target=Target(model,reference['mean'],reference['precision'])
    ids=torch.arange(n,n+added) if added<=16 else torch.linspace(n,n+added-1,4).round().long()
    probes=new['white'][:,ids].reshape(-1,d);stored=new['log_ratio'][:,ids].flatten();replayed=[]
    for batch in probes.split(16):
        value,bad=target.log_ratio_no_grad(batch)
        if bool(bad.any()):raise ValueError('Invalid cached continuation state')
        replayed.append(value)
    cache_error=float((torch.cat(replayed)-stored).abs().max())
    if cache_error>1e-6:raise ValueError('Continuation target cache differs')
    return {'case':name,'original_draws_per_chain':n,'additional_draws_per_chain':added,
        'saved_draws_per_chain':n+added,'original_prefix_bit_identical':True,
        'reference_bytes_unchanged':True,'original_settings_unchanged':True,'warmup_added':0,
        'frozen_step_size':new['step_size'],'exact_rng_consumption_replay_pass':True,
        'split_gradient_batches_added':split_steps,'extra_gradient_replays_at_boundaries':boundary_replays,
        'maximum_new_cached_target_error':cache_error,'new_cached_states_checked':len(probes),
        'all_pass':True,'qualification':'continuation integrity only; no reconstruction scoring or final eligibility; counters exclude unsaved work discarded by earlier timeouts'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--before',required=True);p.add_argument('--after',required=True)
    p.add_argument('--inputs',required=True);p.add_argument('--case-index',type=int,required=True)
    p.add_argument('--expected-additional',type=int);p.add_argument('--output',required=True)
    a=p.parse_args();torch.set_num_threads(2)
    import torch.backends.opt_einsum as backend
    backend.enabled=True;backend.strategy='greedy'
    result=verify(a.before,a.after,a.inputs,a.case_index,a.expected_additional)
    write(a.output,result);print(json.dumps(result),flush=True)
