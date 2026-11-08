"""Independent saved-image metric/provenance audit; no fitting or rendering."""
import json,hashlib,time
from pathlib import Path
import numpy as np
from skimage.metrics import structural_similarity
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'reports/compositional-dynamics-v17';B=ROOT/'runs/compositional-dynamics-v17/appearance-pilot'
def main():
 read=lambda p:json.loads(Path(p).read_text());h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();p=read(R/'appearance-pilot-plan.json');d=read(R/'appearance-pilot-results.json');commit=read(B/'prediction-commit.json');train=read(B/'training-complete.json');manifest=read(B/'input-manifest.json');errors=[];checks=0;evidence={}
 def ck(x,msg):
  nonlocal checks
  checks+=1
  if not x:errors.append(msg)
 def eq(x,y,msg):ck(np.allclose(x,y,rtol=1e-9,atol=1e-11),msg)
 for f,v in p['source_sha256'].items():ck(h(ROOT/f)==v,'source '+f)
 for f,v in manifest['files'].items():ck(h(ROOT/f)==v,'input '+f)
 ph=h(R/'appearance-pilot-plan.json');ck(ph==d['plan_sha256']==commit['plan_sha256']==train['plan_sha256']==manifest['plan_sha256'],'plan');ck(h(B/'prediction-commit.json')==d['prediction_commit_sha256'],'commit');ck(h(B/'training-complete.json')==commit['training_complete_sha256']and train['complete']and train['returncode']==0,'training');ck(h(ROOT/train['checkpoint'])==train['checkpoint_sha256']and 'iteration_1000/'in train['checkpoint'],'final checkpoint');ck(train['wall_seconds']<900,'budget');ck(h(R/'appearance-pilot-comparison.png')==d['gallery_sha256'],'gallery')
 ck([(x['name'],x['role'])for x in d['rows']]==[(x['name'],x['role'])for x in p['cameras']]==[(x['name'],x['role'])for x in commit['records']],'six exact views');outrows=[]
 for row,c in zip(d['rows'],commit['records']):
  pp=ROOT/c['path'];rp=B/'held-reference'/f"{row['name']}.npz";ck(h(pp)==row['prediction_sha256']==c['sha256']and h(rp)==row['reference_sha256'],'image hashes');evidence[str(pp.relative_to(ROOT))]=h(pp);evidence[str(rp.relative_to(ROOT))]=h(rp);ck(pp.stat().st_mtime_ns< (B/'prediction-commit.json').stat().st_mtime_ns<rp.stat().st_mtime_ns,'recorded writechronology')
  with np.load(pp)as q:rgb=q['rgb'].astype(float);mask=q['body_visible'];depth=q['body_visible_depth'];raw=q['raw_rgb'];alpha=q['alpha']
  with np.load(rp)as q:rrgb=q['rgb'].astype(float);rmask=q['body_visible'];rdepth=q['body_visible_depth']
  ck(np.array_equal(mask,alpha>=.5),'fixedalphathreshold');eq(rgb,np.clip(raw,0,1),'fixedclip');ck(c['clipped_channels']==int(((raw<0)|(raw>1)).sum()),'clipcount');ck(np.isfinite(rgb).all()and np.array_equal(np.isfinite(depth),mask)and np.array_equal(np.isfinite(rdepth),rmask),'depthmaskcontract')
  union=mask|rmask;joint=mask&rmask;mse=(rgb-rrgb)**2;ss,sm=structural_similarity(rgb,rrgb,data_range=1.,channel_axis=2,gaussian_weights=True,sigma=1.5,use_sample_covariance=False,full=True);inner=union[5:-5,5:-5];gm=float(mse.mean());um=float(mse[union].mean());vals=dict(rgb_global_mse=gm,rgb_foreground_union_mse=um,psnr_global_db=-10*np.log10(max(gm,1e-12)),psnr_foreground_union_db=-10*np.log10(max(um,1e-12)),ssim_rgb_global=ss,ssim_rgb_foreground_union=sm[5:-5,5:-5][inner].mean(),silhouette_iou=joint.sum()/union.sum(),depth_joint_rmse_m=np.sqrt(np.mean((depth[joint]-rdepth[joint])**2)),reference_body_pixels=rmask.sum(),predicted_body_pixels=mask.sum(),union_body_pixels=union.sum(),joint_depth_pixels=joint.sum(),missing_predicted_pixels=(rmask&~mask).sum(),extra_predicted_pixels=(mask&~rmask).sum(),reference_depth_coverage=joint.sum()/rmask.sum(),ssim_foreground_valid_center_count=inner.sum())
  for k,v in vals.items():eq(v,row['metrics'][k],k)
  outrows.append(dict(role=row['role'],values=vals))
 for role,co in d['summary'].items():
  rows=[x for x in outrows if x['role']==role];ck(len(rows)==co['count']==(4 if role=='train'else 2),'cohortdenominator')
  for k,v in co['metrics'].items():eq(np.mean([x['values'][k]for x in rows]),v['mean'],'summary '+k);ck(v['available']==v['expected']==len(rows),'summarycoverage')
 for path in [R/'appearance-pilot-plan.json',R/'appearance-pilot-results.json',B/'input-manifest.json',B/'prediction-commit.json',B/'training-complete.json',ROOT/train['checkpoint']]:evidence[str(path.relative_to(ROOT))]=h(path)
 o=dict(passed=not errors,checks=checks,errors=errors,created_unix=time.time(),evidence_sha256=evidence,summary=d['summary'],scope='All6savedimages independently rescored withofficialSSIM and directPSNR/depth/masks, checkpoint/source/inputcensus andfilesystemchronology checked. Source/test allpredictions-before-reference guard previouslyreviewed. Staticknown-geometry1000iterationpilot only, notnativePhysTwin inverse/dynamiccomparison orrenderparity. Large train/heldgap anddeptherror retained; fullimageSSIMbackground dilution exposed.')
 (R/'appearance-pilot-final-independent-audit.json').write_text(json.dumps(o,indent=2)+'\n');print(checks,errors)
if __name__=='__main__':main()
