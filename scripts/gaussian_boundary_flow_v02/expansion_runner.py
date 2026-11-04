"""Sequential, failure-recording GPU worker for the frozen exposed-case expansion.

Each process owns one GPU and one explicit case list. It never restarts a failed
case within a run or reads evaluator targets. Scoring is a separate process.
"""
import argparse,json,subprocess,time,hashlib
from pathlib import Path
from scripts.gaussian_boundary_flow_v02.object_score import load_sealed
from scripts.gaussian_boundary_flow_v02.export_inputs import load_package


def write(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',required=True);p.add_argument('--cases',nargs='+',required=True);args=p.parse_args()
    root=Path('runs/gaussian-boundary-flow-v03-expansion');root.mkdir(exist_ok=True)
    geometry=json.loads(Path('reports/gaussian-boundary-flow-v02/expansion-geometry.json').read_text())
    eligible={r['case']:r for r in geometry['rows']if r['eligible']}
    jobs=[]
    for case in args.cases:
        if case not in eligible:raise ValueError('No admitted initial geometry: '+case)
        row=eligible[case];data=load_package(row['package'])
        if data['name']!=case:raise ValueError('Case identity mismatch')
        jobs.append({'case':case,'package':row['package'],'geometry':row['geometry'],'output':str(root/case)})
    worker=root/('worker-'+args.device.replace(':','-'));worker.mkdir(exist_ok=True)
    plan={'device':args.device,'jobs':jobs,'samples':128,'steps':5000,'draws':32,
          'driver_sha256':hashlib.sha256(Path('scripts/gaussian_boundary_flow_v02/package_campaign_v03.py').read_bytes()).hexdigest(),
          'scope':'Fixed exposed development cases; no evaluator imports or fresh trajectories'}
    if (worker/'plan.json').exists():
        if json.loads((worker/'plan.json').read_text())!=plan:raise ValueError('Changed worker plan')
    else:write(worker/'plan.json',plan)
    results=[]
    for job in jobs:
        run=Path(job['output']);start=time.time()
        if (run/'complete.json').exists():
            load_sealed(run);results.append(dict(case=job['case'],state='verified_completed'));continue
        command=['.venv/bin/python','-m','scripts.gaussian_boundary_flow_v02.package_campaign_v03',
                 '--geometry',job['geometry'],'--package',job['package'],'--output',job['output'],
                 '--samples','128','--steps','5000','--draws','32','--device',args.device]
        write(worker/'status.json',{'state':'running','case':job['case'],'completed_jobs':results,'started_unix':start})
        with (worker/(job['case']+'.log')).open('a')as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        row=dict(case=job['case'],returncode=result.returncode,elapsed_s=time.time()-start,
                 state='complete'if result.returncode==0 else 'failed')
        if result.returncode==0:load_sealed(run)
        results.append(row);write(worker/'status.json',{'state':'between_jobs','completed_jobs':results})
        print(json.dumps(row),flush=True)
    write(worker/'complete.json',{'state':'terminal','jobs':results,'all_succeeded':all(r['state']in ('complete','verified_completed')for r in results)})

if __name__=='__main__':main()
