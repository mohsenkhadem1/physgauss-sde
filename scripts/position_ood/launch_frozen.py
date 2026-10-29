"""Launch resumable services only after the method manifest has been frozen."""
from pathlib import Path
import subprocess,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.position_ood.study import ROOT,BASE,within_deadline
from scripts.position_ood.seal import verify

def launch(name,script,args,gpu=None,threads=3):
 unit='position-ood-v09-s10-'+name
 state=subprocess.run(['systemctl','--user','is-active',unit],capture_output=True,text=True)
 if state.stdout.strip()=='active':return
 env={'POSITION_CAMPAIGN':BASE.name,'POSITION_BOUNDARY':'smooth','POSITION_INVERSE_DEVICE':'cpu' if gpu is None else 'cuda:0','CUDA_VISIBLE_DEVICES':'' if gpu is None else str(gpu),'OMP_NUM_THREADS':str(threads),'OPENBLAS_NUM_THREADS':'1'}
 command=['systemd-run','--user','--unit='+unit,'--working-directory='+str(ROOT),*[f'--setenv={k}={v}' for k,v in env.items()],str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/position_ood'/script),*map(str,args)]
 subprocess.run(command,check=True)

def main():
 within_deadline();verify()
 for worker in range(6):launch(f'eval-cpu-{worker}','evaluation_cpu.py',['--worker',worker])
 for gpu in range(3):launch(f'eval-gpu-{gpu}','campaign_jobs.py',['--kind','gpu','--worker',gpu],gpu,3)
 for worker in range(2):launch(f'active-cpu-{worker}','active_sequence.py',['--kind','cpu','--worker',worker,'--workers',2])
 for gpu in range(3):launch(f'active-gpu-{gpu}','active_sequence.py',['--kind','gpu','--worker',gpu,'--workers',3],gpu,3)
 launch('ablation-cpu','active_ablation.py',['--kind','cpu'])
 launch('ablation-gpu','active_ablation.py',['--kind','gpu'],1,3)
 launch('active-scoring','active_scoring_wait.py',[],None,2)
if __name__=='__main__':main()
