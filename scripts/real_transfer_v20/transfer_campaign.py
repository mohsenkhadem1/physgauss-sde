"""Frozen v19 transfer diagnostic on every declared pair, including failures."""
import argparse,concurrent.futures,subprocess,time,threading
from .registration import PAIRS
from .development_run import ROOT,BASE,write

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--label',default='original-frozen')
    parser.add_argument('--fit-root',default='runs/real-transfer-v19/fits/exploratory-01');args=parser.parse_args()
    jobs=[(source,target,method) for source,target in PAIRS for method in ('neural','analytic')]
    lock=threading.Lock()
    def worker(device):
        while True:
            with lock:
                if not jobs:return
                source,target,method=jobs.pop(0)
            key=f'{source}--{target}--{method}'
            out=BASE/'frozen-transfer'/args.label/f'{source}--{target}'/method
            if (out/'complete.json').exists():continue
            with (BASE/f'transfer-{args.label}-{key}.log').open('w') as f:
                try:
                    code=subprocess.run([str(ROOT/'runs/phystwin-env/bin/python'),'-u','-m','scripts.real_transfer_v20.neural_transfer',
                        '--source',source,'--target',target,'--method',method,'--device',device,
                        '--label',args.label,'--fit-root',args.fit_root],cwd=ROOT,
                        stdout=f,stderr=subprocess.STDOUT,timeout=1800).returncode
                except subprocess.TimeoutExpired:code=124
            write(BASE/f'transfer-{args.label}-{key}-status.json',dict(returncode=code,finished=time.time()))
            print(key,code,flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for future in [pool.submit(worker,f'cuda:{i}') for i in (1,2)]:future.result()

if __name__=='__main__':main()
