"""Actual generated guard commands only; no GPU allocation or model call."""
import json,subprocess,os
from experiments import benchmark_surface_amplitude_concurrency_v01 as benchmark


def test_three_generated_owned_worker_guards_and_wrong_device_refusal():
    for slot in range(3):
        result=subprocess.run(benchmark.command(slot,'0'*64,guard_only=True),cwd=benchmark.ROOT,capture_output=True,text=True,timeout=20)
        assert result.returncode==0,result.stderr
        assert json.loads(result.stdout)['slot']==slot
    command=benchmark.command(0,'0'*64,guard_only=True)
    command[1]='CUDA_VISIBLE_DEVICES='
    result=subprocess.run(command,cwd=benchmark.ROOT,capture_output=True,text=True,timeout=20)
    assert result.returncode!=0 and 'environment required' in result.stderr
