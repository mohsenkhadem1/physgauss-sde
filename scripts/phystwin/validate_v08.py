"""Validate completed v08 artifacts and preserve their reproducibility inventory."""
from pathlib import Path
import hashlib,json,re,subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'runs/phystwin-informative-v08';OUT=ROOT/'reports/phystwin-informative-v08'
def write(name,value):(OUT/name).write_text(json.dumps(value,indent=2)+'\n')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 seals=[ROOT/'runs/phystwin-sequential-support-v06'/name/'seal.json' for name in ['analysis','analysis-rgbd']]+[ROOT/'runs/phystwin-cost-aware-v07/seal.json',BASE/'seal.json']
 for path in seals:
  seal=json.loads(path.read_text())
  for name,digest in seal['hashes'].items():assert sha(ROOT/name)==digest,name
 summary=json.loads((OUT/'comparison.json').read_text());assert summary['reference_qualification']['count']==60
 fits=list((BASE/'phystwin').glob('*/summary-dt1.json'));assert len(fits)==72,len(fits)
 for path in fits:
  row=json.loads(path.read_text());member=row['member'];dev=row['split']=='development'
  expected=(31,3,900) if member else ((61,27,1800) if dev else (91,27,2700))
  assert (row['training_frames'],row['visible_markers'],row['optimizer_updates'])==expected,path
  if member:assert row['fit_actions']==['initial'] and not (path.parent/'held_oblique-dt1.npz').exists()
  else:assert (path.parent/'summary-dt2.json').exists()
 links=re.findall(r'\]\(([^)]+)\)',(OUT/'REPORT.md').read_text())
 for name in links:assert (OUT/name).is_file(),name
 for path in BASE.glob('test-*-*.npz'):
  with np.load(path) as data:
   for key in ['mean','sd','truth']:assert np.isfinite(data[key]).all(),(path,key)
 # Decode every frame using the isolated PhysTwin environment's OpenCV.
 code="""import cv2,json,sys
from pathlib import Path
root=Path(sys.argv[1]);result={}
for name,expected in [('actual-vs-predicted.mp4',144),('phystwin-fixed-vs-ensemble.mp4',186)]:
 c=cv2.VideoCapture(str(root/name));assert c.isOpened();fps=c.get(cv2.CAP_PROP_FPS);size=[int(c.get(cv2.CAP_PROP_FRAME_WIDTH)),int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))];n=0
 while True:
  ok,frame=c.read()
  if not ok:break
  assert frame is not None;n+=1
 c.release();assert n==expected,(name,n);result[name]=dict(decoded_frames=n,fps=fps,size=size)
print(json.dumps(result))
"""
 videos=json.loads(subprocess.check_output([str(ROOT/'runs/phystwin-env/bin/python'),'-c',code,str(OUT)],text=True));write('video-validation.json',dict(videos=videos))
 source_paths=sorted((ROOT/'scripts/phystwin').glob('*v08*.py'))+[ROOT/'tests/test_v08_informative.py']
 provenance=dict(implementation_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_hashes={str(p.relative_to(ROOT)):sha(p) for p in source_paths},upstream_v06_manifest=json.loads((ROOT/'reports/phystwin-cost-aware-v07/implementation-provenance.json').read_text())['upstream_v06_manifest'],raw_data=str(BASE.relative_to(ROOT)),simulation_only=True)
 write('implementation-provenance.json',provenance);write('sealed-input-hashes.json',json.loads((BASE/'seal.json').read_text()))
 tests=(OUT/'tests.txt').read_text();assert '21 passed' in tests
 write('artifact-validation.json',dict(reference_cases=60,qualified_reference_cases=60,held_out_objects=24,phystwin_selection_members=27,development_response_fits=21,matched_test_response_fits=24,optimizer_updates_per_test_fit=2700,extra_ensemble_selection_updates=2700,tests_passed=21,report_links_checked=links,v06_v07_v08_seals_verified=True,video_checks=videos,maximum_phystwin_half_step_score_change_mm=max(abs(r['rmse_mm']-r['refined_rmse_mm']) for r in summary['phystwin']),raw_datasets_published=False))
 print('VALIDATED 60 references, 72 PhysTwin fits, four seals, two fully decoded videos')
if __name__=='__main__':main()
