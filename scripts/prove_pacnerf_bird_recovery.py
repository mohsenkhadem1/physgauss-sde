"""Verify original Bird recovery using a separate, unmodified worker checkout.

This adds a bounded diagnostic prefix, not a new production stopping rule.
No held-out positions/images are scored and no convergence is claimed.
"""
import argparse,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
from pacnerf_recovery import restore_correlations,differences

FROZEN_WORKER='a14766e33fe7642a681441f7f919be40a2fe1f7e'
CASE='bird__4views__constant__marginal'


def run(args):
    import torch
    locations={name:Path(getattr(args,name)).resolve() for name in ['worker','inputs','states','results','output']}
    worker=locations['worker'];out=locations['output']/CASE
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=worker,text=True).strip()!=FROZEN_WORKER:
        raise ValueError('Recovery requires the unmodified original worker commit')
    saved=json.loads((locations['results']/'settings.json').read_text())
    for name,digest in saved['inference_files_sha256_lf'].items():
        if hashlib.sha256((worker/'scripts'/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=digest:
            raise ValueError('Frozen inference source differs: '+name)
    if out.exists():raise ValueError('Use a fresh diagnostic output directory')
    shutil.copytree(locations['results'],out)
    shutil.copytree(locations['states'],out,dirs_exist_ok=True)
    sys.path.insert(0,str(worker/'src'));sys.path.insert(0,str(worker/'scripts'))
    os.chdir(worker)
    import compare_pacnerf_discrepancy as original
    import pacnerf_hybrid as hybrid
    torch.set_num_threads(2)
    import torch.backends.opt_einsum as backend
    backend.enabled=True;backend.strategy='greedy'
    before=torch.load(out/'chains.pt',weights_only=True)
    n=before['white'].shape[1];report={'frozen_worker':FROZEN_WORKER,'original_draws_per_chain':n,
        'future_geometry_used':False,'valid_for_ranking':False,'additional_draws':args.additional_draws,
        'support_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'support_module_sha256':hashlib.sha256(Path(__file__).with_name('pacnerf_recovery.py').read_bytes()).hexdigest()}
    make_model=original.make_inference_model;make_target=original.make_target;sample=hybrid.sample
    class DiagnosticBoundary(Exception):pass
    def model_factory(inputs,index):
        model,bundle=make_model(inputs,index)
        report['recomputed_model_differences']=differences(saved['model'],model.specification(),'model')
        report['constant_restoration']=restore_correlations(model,saved['model'])
        return model,bundle
    def target_factory(model,reference):
        target=make_target(model,reference)
        report['original_density_replay']=original.replay(model,target,before)
        report['original_settings_guard_passed']=True
        if args.audit_only:raise DiagnosticBoundary
        return target
    def bounded_sample(*call,**kwargs):
        call=list(call);call[4]=n+args.additional_draws
        direct=sample(*call,**kwargs)
        for key in ['white','log_ratio','energy_errors']:
            if not torch.equal(before[key],direct[key][:,:n]):raise ValueError('Original sample prefix changed: '+key)
        report['original_prefix_preserved_exactly']=True
        split=locations['output']/'split-proof'
        split.mkdir();shutil.copyfile(locations['states']/'chains.pt',split/'chains.pt')
        call[3]=split;call[4]=n+args.additional_draws//2
        sample(*call,**kwargs)
        call[4]=n+args.additional_draws
        resumed=sample(*call,**kwargs)
        for key in ['white','log_ratio','energy_errors','rng_state','position','position_log_ratio']:
            if not torch.equal(direct[key],resumed[key]):raise ValueError('Direct/split recovery differs: '+key)
        if direct['counts']!=resumed['counts'] or direct['parameter_slice_attempts']!=resumed['parameter_slice_attempts']:
            raise ValueError('Direct/split acceptance or parameter updates differ')
        report['direct_vs_split_recovery_bit_identical']=True
        report['final_draws_per_chain']=direct['white'].shape[1]
        raise DiagnosticBoundary
    original.make_inference_model=model_factory;original.make_target=target_factory;hybrid.sample=bounded_sample
    try:original.run(locations['inputs'],locations['output'],8)
    except DiagnosticBoundary:report['all_pass']=True
    finally:
        (locations['output']/'recovery-proof.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['worker','inputs','states','results','output']:p.add_argument('--'+name,required=True)
    p.add_argument('--audit-only',action='store_true');p.add_argument('--additional-draws',type=int,default=32)
    a=p.parse_args()
    if a.additional_draws<32 or a.additional_draws%2:raise ValueError('Need an even diagnostic prefix of at least32 draws')
    run(a)
