"""Released development appearance audit; no fresh-cohort reads."""
from pathlib import Path
import json,hashlib,time
import numpy as np
from skimage.metrics import structural_similarity
R=Path(__file__).resolve().parents[2];D=R/'reports/compositional-dynamics-v17';B=R/'runs/compositional-dynamics-v17/matched-appearance-inference-recovery/full'
def main():
 h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();path=D/'matched-appearance-inference-recovery/dynamic-results.json';r=json.loads(path.read_text());c=json.loads((B/'prediction-commit.json').read_text());errors=[];checks=[];ev={str(path.relative_to(R)):h(path),str((B/'prediction-commit.json').relative_to(R)):h(B/'prediction-commit.json')}
 def ck(v,s):checks.append(s);errors.extend([]if v else[s])
 def eq(x,y,s):ck(np.isclose(x,y,rtol=1e-7,atol=1e-10),s)
 ck(r['prediction_commit_sha256']==h(B/'prediction-commit.json'),'commit');ck(len(r['rows'])==260 and len({(x['method'],x['frame'],x['camera'])for x in r['rows']})==260,'260census');preds={(x['method'],x['frame'],x['camera']):x for x in c['records']};sample=[]
 for row in r['rows']:
  key=row['method'],row['frame'],row['camera'];a=preds[key];pp=R/a['path'];rp=B/'references'/f"{row['frame']:03d}-{row['camera']}.npz";ck(h(pp)==row['prediction_sha256']==a['sha256'],'predictionhash');ck(h(rp)==row['reference_sha256'],'referencehash');ev[str(rp.relative_to(R))]=h(rp)
  if row['frame']not in [0,13,25]:continue
  with np.load(pp)as f:rgb=f['rgb'].astype(float);pm=f['body_visible'];pd=f['body_visible_depth'].astype(float)
  with np.load(rp)as f:truth=f['rgb'].astype(float);tm=f['body_visible'];td=f['body_visible_depth'].astype(float)
  union=pm|tm;joint=pm&tm;squared=(rgb-truth)**2;ss,sm=structural_similarity(rgb,truth,data_range=1.,channel_axis=2,gaussian_weights=True,sigma=1.5,use_sample_covariance=False,full=True);inner=union[5:-5,5:-5];v=dict(rgb_global_mse=squared.mean(),rgb_foreground_union_mse=squared[union].mean(),psnr_global_db=-10*np.log10(max(squared.mean(),1e-12)),psnr_foreground_union_db=-10*np.log10(max(squared[union].mean(),1e-12)),ssim_rgb_global=ss,ssim_rgb_foreground_union=sm[5:-5,5:-5][inner].mean(),silhouette_iou=joint.sum()/union.sum(),reference_body_pixels=tm.sum(),predicted_body_pixels=pm.sum(),missing_predicted_pixels=(tm&~pm).sum(),extra_predicted_pixels=(pm&~tm).sum(),reference_depth_coverage=joint.sum()/tm.sum(),depth_joint_rmse_m=np.sqrt(np.mean((pd[joint]-td[joint])**2)))
  for k,value in v.items():eq(value,row['metrics'][k],str(key)+k)
  sample.append(key)
 for method,groups in r['summary'].items():
  for role,summary in groups.items():
   rows=[x for x in r['rows']if x['method']==method and x['role']==role];ck(len(rows)==summary['rows'],'groupcensus')
   for key,s in summary['metrics'].items():
    vals=[x['metrics'][key]for x in rows if x['metrics'][key]is not None];ck(len(vals)==s['available']and len(rows)==s['expected'],'coverage');eq(np.mean(vals),s['mean'],'mean')
 for n,v in r['reference_images_sha256'].items():ck(h(R/n)==v,'allrefhash')
 out=dict(passed=not errors,checks=len(checks),errors=errors,created_unix=time.time(),evidence_sha256=ev,sampled_keys=sample,summary=r['summary'],scope='All260prediction/reference hash/census and summary coverage;30fixedimagepairs atframes0/13/25 acrossall5cameras andbothmethods independently recomputed PSNR/officialSSIM/IoU/depth/coverage. Development oneobject only; sharedinitialappearance, mechanics+transport effects, no population or native-reconstruction claim. Fresh cohort targets remain unopened.',source_sha256={str(Path(__file__).relative_to(R)):h(__file__)})
 (D/'matched-appearance-development-final-independent-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k]for k in ['passed','checks','errors']}),flush=True)
if __name__=='__main__':main()
