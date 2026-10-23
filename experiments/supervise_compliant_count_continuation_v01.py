"""Independent1800s CPU/noGPU supervisor; reviewed shared ENOENT/ESRCH owner."""
from pathlib import Path
import argparse,hashlib,json,sys
from experiments.supervise_surrogate_sbc_continuation import supervise
ROOT=Path(__file__).resolve().parents[1]


def main(protocol,digest):
 path=ROOT/protocol
 if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('Exact reviewed reference recipe')
 p=json.loads(path.read_text());op=path.parent
 command=['env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2','MKL_NUM_THREADS=2','taskset','-c','10,11',str(ROOT/'.venv/bin/python'),'-u','-m','experiments.run_compliant_count_continuation_v01','run','--protocol',protocol,'--digest',digest]
 # Atomic active-journal symlink exposes each current stage to independent
 # ownership tracking, including fallback cleanup if controller is killed.
 r=supervise(command,op/'actual-reference.log',op/'actual-supervisor-receipt.json',journal_path=ROOT/p['output']/'active-journal.json',wall_s=1800,RSS_bytes=2*1024**3);print(json.dumps(r));return 0 if r['completed']else 1


if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--protocol',required=True);a.add_argument('--digest',required=True);v=a.parse_args();sys.exit(main(v.protocol,v.digest))
