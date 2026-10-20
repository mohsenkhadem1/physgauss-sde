"""Bounded free-CPU follow-up of this one immutable Bird experiment."""
import argparse,json,math,os,urllib.request
from pathlib import Path
ROOT_RUN=39399618548
REFERENCE='78d304d09456ea8fff4011322b37325e358dabaf52bd989f167d14848c37daaa'
FINGERPRINT='eb703c41c125e971edae446bcad13d8f80a0ca5a34efb9f6a7b156762bf9f655'
REPOSITORY='mohsenkhadem1/physgauss-sde'


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):return None


def decision(report,parent_run):
    protocol=report['source_protocol']
    if (report['case_index']!=8 or report['source_fingerprint']!=FINGERPRINT or report['reference_sha256']!=REFERENCE
            or protocol['reference_sha256']!=REFERENCE or protocol['warmup']!=500 or not protocol['interweave']
            or protocol['frozen_model']['version']!='floor-conditioned-hierarchical-v2'
            or not report['four_independent_chains'] or not report['density_replay']['all_pass']):
        raise ValueError('Follow-up is restricted to the frozen four-chain Bird experiment')
    n=report['draws_per_chain']
    if n not in [2000,4000,8000]:raise ValueError('Unrecognized retained budget; no dispatch')
    result={'root_run':ROOT_RUN,'parent_run':int(parent_run),'completed_draws_per_chain':n,
        'free_standard_cpu_only':True,'larger_comparison_dispatched':False}
    if report['all_monitored_quantities_pass']:
        numbers=[report[k] for k in ['worst_rhat','minimum_bulk_ess','minimum_tail_ess']]
        if (not all(math.isfinite(x) for x in numbers) or numbers[0]>=1.01 or min(numbers[1:])<400
                or report['retained_invalid'] or report['retained_divergent']):
            raise ValueError('Convergence flag contradicts numeric diagnostics')
        result.update({'action':'validate_uncertainty','workflow':'pacnerf-bird-reflected-validate.yml',
            'inputs':{'parent_run':str(parent_run),'draws':str(n)}})
    elif n<8000:
        result.update({'action':'continue_same_chains','workflow':'pacnerf-bird-reflected-continue.yml',
            'inputs':{'parent_run':str(parent_run),'draws':str(2*n)}})
    else:
        result.update({'action':'stop_without_comparison','reason':'Bird remains unconverged at the predeclared8000-draw cap; preserve failure for inference audit'})
    return result


def main(args):
    if os.environ.get('GITHUB_REPOSITORY',REPOSITORY)!=REPOSITORY:raise ValueError('Wrong repository')
    report=json.loads(Path(args.report).read_text());route=decision(report,args.parent_run)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise ValueError('A route record already exists; inspect the previous dispatch before retrying')
    output.write_text(json.dumps(route,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(route),flush=True)
    if args.dispatch and 'workflow' in route:
        token=os.environ['GITHUB_TOKEN']
        url=f'https://api.github.com/repos/{REPOSITORY}/actions/workflows/{route["workflow"]}/dispatches'
        request=urllib.request.Request(url,data=json.dumps({'ref':'main','inputs':route['inputs']}).encode(),
            headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json','User-Agent':'physgauss-bounded-bird-followup'},method='POST')
        with urllib.request.build_opener(NoRedirect()).open(request,timeout=40) as response:
            if response.status!=204:raise ValueError('Unexpected dispatch response; inspect remote state before retrying')
        route['dispatch_accepted']=True
        output.write_text(json.dumps(route,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['report','output']:p.add_argument('--'+key,required=True)
    p.add_argument('--parent-run',type=int,required=True);p.add_argument('--dispatch',action='store_true')
    main(p.parse_args())
