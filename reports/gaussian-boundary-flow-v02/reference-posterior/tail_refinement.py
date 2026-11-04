"""Frozen supplemental world2 tail check: retain spacing as domain expands."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from scripts.gaussian_boundary_flow_v02.reference_posterior import (
    ROOT, OUT, REPORT, SIGMA, Likelihood, RefinedBlock, integrate, compare_quadratures, write_json,
)


def main():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    original=json.loads((OUT/'plan.json').read_text());failed=json.loads((OUT/'world-2/result.json').read_text())
    if failed['converged']:raise RuntimeError('Supplement is specifically for the preserved failed case')
    directory=OUT/'world-2-fixed-spacing';directory.mkdir(exist_ok=True)
    plan={'scope':'Supplemental numerical check of previously failed world2 only; original failure retained',
          'original_plan_sha256':hashlib.sha256((OUT/'plan.json').read_bytes()).hexdigest(),
          'failed_result_sha256':hashlib.sha256((OUT/'world-2/result.json').read_bytes()).hexdigest(),
          'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'tail_grids':[[4.,129],[6.,193],[8.,257]],'tail_spacing':.0625,
          'resolution_grids':[[8.,129],[8.,257]],
          'thresholds':original['tail_thresholds'],'max_edge_mass':original['max_edge_mass'],
          'max_new_evaluations':120000,'budget_seconds':600}
    if (directory/'plan.json').exists():
        if json.loads((directory/'plan.json').read_text())!=plan:raise RuntimeError('Supplement plan changed')
    else:write_json(directory/'plan.json',plan)
    remaining=min(600.,max(0.,1200-(time.time()-(OUT/'plan.json').stat().st_mtime)))
    deadline=time.monotonic()+remaining
    obs=torch.load(OUT/'observations.pt',weights_only=True);block=RefinedBlock(dtype=torch.float64,device='cuda:0')
    engine=Likelihood(block,obs['truth_z'][2:3].to('cuda:0'),obs['observed'][2:3].to('cuda:0'),deadline,120000)
    center=np.array(failed['map_q']);transform=np.array(failed['transform'])
    result={'scope':plan['scope'],'initial_budget_remaining_seconds':remaining,'tail_checks':[],'resolution_checks':[],'converged':False}
    previous=None;tail_ok=False;start=time.monotonic()
    try:
        for L,n in plan['tail_grids']:
            current=integrate(engine,center,transform,L,n,directory)
            check=None if previous is None else compare_quadratures(previous,current,plan['thresholds'])
            result['tail_checks'].append({'quadrature':current,'change':check});previous=current
        tail_ok=all(row['change']['passed'] for row in result['tail_checks'][1:]) and previous['outer_edge_mass']<plan['max_edge_mass']
        previous=None
        for L,n in plan['resolution_grids']:
            current=integrate(engine,center,transform,L,n,directory)
            check=None if previous is None else compare_quadratures(previous,current,plan['thresholds'])
            result['resolution_checks'].append({'quadrature':current,'change':check});previous=current
        result['converged']=bool(tail_ok and check['passed'])
        result['posterior']=current
        truth=torch.sigmoid(obs['truth_z'][2,[1,3]]).numpy();q=np.array(current['quantiles_q'])
        result['truth_q']=truth.tolist();result['truth_in_conditional_90_intervals']=((truth>=q[:,0])&(truth<=q[:,2])).tolist()
        C=np.array(current['covariance_q']);result['posterior_correlation']=float(C[0,1]/np.sqrt(C[0,0]*C[1,1]))
    except Exception as e:result['failure']=f'{type(e).__name__}: {e}'
    result['new_evaluations']=engine.count;result['elapsed_seconds']=time.monotonic()-start
    write_json(REPORT/'world2-fixed-spacing.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
