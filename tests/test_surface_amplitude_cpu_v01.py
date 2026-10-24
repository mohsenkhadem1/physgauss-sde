import os,subprocess,json
from experiments import benchmark_surface_amplitude_cpu_v01 as b

def test_generated_CPU_guard_and_wrong_GPU_refusal():
    child=subprocess.run(b.command('0'*64,guard_only=True),cwd=b.ROOT,capture_output=True,text=True,timeout=20)
    assert child.returncode==0,child.stderr
    assert json.loads(child.stdout)['guard_passed']
    command=b.command('0'*64,guard_only=True);command[1]='CUDA_VISIBLE_DEVICES=0'
    bad=subprocess.run(command,cwd=b.ROOT,capture_output=True,text=True,timeout=20)
    assert bad.returncode!=0 and 'guard' in bad.stderr
