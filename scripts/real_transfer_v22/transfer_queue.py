"""Re-evaluate all five frozen transfers and orbit both prespecified passing pairs."""
import subprocess,time,os,json
from .campaign import ROOT,BASE,LABEL,write

def main():
    start=time.monotonic()
    while not all((ROOT/'runs/real-transfer-v19/fits'/LABEL/c/m/'complete.json').exists() for c in ('single_lift_sloth','single_lift_zebra') for m in ('neural','analytic')):
        if time.monotonic()-start>7200:raise TimeoutError('Source fits incomplete')
        time.sleep(10)
    python=str(ROOT/'runs/phystwin-env/bin/python')
    subprocess.run([python,'-u','-m','scripts.real_transfer_v20.transfer_campaign','--label','v22-corrected-frozen','--fit-root',f'runs/real-transfer-v19/fits/{LABEL}'],cwd=ROOT,check=True)
    subprocess.run([python,'-m','scripts.real_transfer_v20.score_transfer','--label','v22-corrected-frozen'],cwd=ROOT,check=True)
    reg=json.loads((ROOT/'reports/real-transfer-v20/initial-registration.json').read_text())['pairs']
    for pair in reg:
        if not pair['initial_geometry_gate']:continue
        subprocess.run([str(ROOT/'runs/deform360-reconstruction-env/bin/python'),'-u','-m','scripts.real_transfer_v22.novel_views','--case',pair['target'],'--source',pair['source']],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OPENBLAS_NUM_THREADS='1'),check=True,timeout=1800)
    write(BASE/'transfer-complete.json',dict(complete=True))

if __name__=='__main__':main()
