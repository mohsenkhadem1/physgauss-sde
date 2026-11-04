"""Three bounded GPU workers for the development object's calibrated views."""
import argparse,concurrent.futures,json,os,subprocess,threading,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'runs/real-transfer-v19'

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--object',default='046-sponge')
    parser.add_argument('--episode',type=int,default=0)
    parser.add_argument('--devices',nargs='+',default=['cuda:0','cuda:1','cuda:2'])
    args=parser.parse_args()
    episode=BASE/'Deform360/processed'/args.object/f'episode_{args.episode:04d}'
    cameras=sorted(p.parent.name for p in episode.glob('*/undistorted.mp4'))
    queue=BASE/'deform360-mask-queues'/args.object/f'episode_{args.episode:04d}'
    queue.mkdir(parents=True,exist_ok=True)
    for camera in cameras:
        path=queue/(camera+'.json')
        if not path.exists():path.write_text(json.dumps(dict(camera=camera,state='pending')))
    lock=threading.Lock()
    def write(path,data):
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)
    def worker(device):
        while True:
            with lock:
                pending=[(p,json.loads(p.read_text())) for p in sorted(queue.glob('*.json')) if json.loads(p.read_text())['state']=='pending']
                if not pending:return
                path,job=pending[0];job.update(state='running',device=device,started=time.time());write(path,job)
            command=[str(ROOT/'runs/deform360-env/bin/python'),'-m','scripts.real_transfer_v19.deform360_masks',
                     '--object',args.object,'--episode',str(args.episode),'--device',device,'--cameras',job['camera']]
            with path.with_suffix('.log').open('w') as log:
                try:code=subprocess.run(command,cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='2'),stdout=log,stderr=subprocess.STDOUT,timeout=600).returncode
                except subprocess.TimeoutExpired:code=124
            meta=episode/job['camera']/'mask_refined.meta.json'
            job.update(state='complete' if code==0 else 'failed',returncode=code,finished=time.time(),metadata_available=meta.exists());write(path,job)
            print(job['camera'],job['state'],flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.devices)) as pool:
        for f in [pool.submit(worker,d) for d in args.devices]:f.result()

if __name__=='__main__':main()
