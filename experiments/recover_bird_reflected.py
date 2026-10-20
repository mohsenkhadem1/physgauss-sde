"""Continue one immutable Bird shard without changing its worker or target."""
import argparse,hashlib,json,shutil,subprocess,sys
from pathlib import Path
import torch


def digest(path):return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n',b'\n')).hexdigest()


def validate(worker,root,index,draws,source_ref):
    worker=Path(worker).resolve();root=Path(root).resolve()
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=worker,text=True).strip()
    if revision!=source_ref:raise ValueError('Recovery worker is not the exact original commit')
    protocol=json.loads((root/'development_protocol.json').read_text())
    if protocol['chain_index']!=index or protocol['seed']!=970009+index or protocol['warmup']!=500 or not protocol['interweave']:
        raise ValueError('Recovery chain identity or kernel differs')
    actual={p.name:digest(p) for p in (worker/'scripts').glob('*.py')}
    if actual!=protocol['source_sha256_lf'] or digest(worker/'experiments/bird_reflected.py')!=protocol['runner_sha256_lf']:
        raise ValueError('Recovery source files differ')
    if hashlib.sha256((root/'reference.pt').read_bytes()).hexdigest()!=protocol['reference_sha256']:
        raise ValueError('Recovery reference differs')
    path=root/'chains.pt' if (root/'chains.pt').exists() else root/'reflected_warmup.pt'
    saved=torch.load(path,weights_only=True)
    if saved['retained']>draws:raise ValueError('Recovery cannot shrink chains')
    if saved['protocol']['chains']!=1 or saved['protocol']['seed']!=protocol['seed'] or saved['protocol']['warmup']!=500:
        raise ValueError('Saved chain protocol differs')
    if saved['protocol'].get('centered_parameter_interweaving')!='exact-physical-density-v1':
        raise ValueError('Saved chain does not use the interwoven kernel')
    return protocol,path


def prefix_proof(before,after):
    old=torch.load(before,weights_only=True);new=torch.load(after,weights_only=True)
    if old['protocol']!=new['protocol']:raise ValueError('Continued kernel protocol differs')
    n=old['retained']
    if new['retained']<n or new['completed_warmup']<old['completed_warmup']:raise ValueError('Recovery progress went backwards')
    for key in ['white','log_ratio','energy_errors']:
        if n and not torch.equal(old[key],new[key][:,:n]):raise ValueError('Recovery changed saved prefix: '+key)
    return {'original_prefix_draws':n,'new_draws':new['retained'],'prefix_bit_identical':True,
        'completed_warmup_before':old['completed_warmup'],'completed_warmup_after':new['completed_warmup']}


def run(args):
    worker=Path(args.worker).resolve();root=Path(args.shard).resolve()
    protocol,path=validate(worker,root,args.index,args.draws,args.source_ref)
    before=root.parent/f'before-chain-{args.index}.pt';shutil.copyfile(path,before)
    settings=root.parent/f'original-model-{args.index}.json'
    settings.write_text(json.dumps({'model':protocol['frozen_model']}),encoding='utf-8')
    command=[sys.executable,'-u',str(worker/'experiments/bird_reflected.py'),
        '--inputs',str(Path(args.inputs).resolve()),'--reference',str(root/'reference.pt'),
        '--settings',str(settings),'--output',str(root),'--draws',str(args.draws),
        '--warmup','500','--seed',str(protocol['seed']),'--chain-index',str(args.index),'--interweave']
    try:subprocess.run(command,cwd=worker,check=True)
    finally:
        latest=root/'chains.pt' if (root/'chains.pt').exists() else root/'reflected_warmup.pt'
        proof=prefix_proof(before,latest)
        proof.update({'frozen_worker':args.source_ref,'source_files_exact':True,'reference_exact':True})
        (root/'continuation-proof.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(proof),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['worker','shard','inputs','source-ref']:p.add_argument('--'+key,required=True)
    p.add_argument('--index',type=int,choices=range(4),required=True)
    p.add_argument('--draws',type=int,choices=[2000,4000,8000],required=True)
    run(p.parse_args())
