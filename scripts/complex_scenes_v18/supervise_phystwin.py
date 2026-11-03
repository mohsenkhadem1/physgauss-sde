"""One GPU-sharing worker for all twelve native-kernel adapter pilot rollouts."""
import json,os,subprocess
from scripts.complex_scenes_v18.run_development import ROOT,write


if __name__=='__main__':
    base=ROOT/'runs/complex-scenes-v18';out=base/'phystwin-adapter-pilot';out.mkdir(parents=True,exist_ok=False)
    cases=json.loads((base/'development-prefixes/manifest.json').read_text())['cases'];done=[]
    for job in cases:
        name=f"{job['case']}-v{job['variant']}"
        with open(out/(name+'.log'),'ab',buffering=0)as log:
            cmd=[str(base/'baseline-env/bin/python'),'-m','scripts.complex_scenes_v18.phystwin_scene','--case',job['case'],'--variant',str(job['variant'])]
            proc=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            write(out/'supervisor.json',dict(pid=os.getpid(),running=dict(case=job['case'],variant=job['variant'],pid=proc.pid),finished=done,planned=len(cases),terminal=False))
            code=proc.wait();done.append(dict(case=job['case'],variant=job['variant'],exit_code=code))
    write(out/'supervisor.json',dict(pid=os.getpid(),running=None,finished=done,planned=len(cases),terminal=True))
