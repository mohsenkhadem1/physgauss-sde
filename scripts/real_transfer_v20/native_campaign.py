"""Bounded baseline replays and all five predeclared transfer diagnostics."""
import json,subprocess,time
from .registration import PAIRS
from .development_run import ROOT,BASE,write

def main():
    for source,target in [('single_lift_zebra','single_lift_zebra')]+PAIRS:
        out=BASE/'native-transfer'/f'{source}--{target}'
        if (out/'complete.json').exists():continue
        log=BASE/f'native-{source}--{target}.log'
        with log.open('w') as f:
            code=subprocess.run([str(ROOT/'runs/phystwin-env/bin/python'),'-u','-m','scripts.real_transfer_v20.native_transfer',
                '--source',source,'--target',target,'--device','cuda:0'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=900).returncode
        write(BASE/f'native-{source}--{target}-status.json',dict(returncode=code,finished=time.time()))
        print(source,target,code,flush=True)

if __name__=='__main__':main()
