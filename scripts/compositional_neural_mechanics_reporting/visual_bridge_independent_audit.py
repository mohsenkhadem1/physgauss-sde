import json,hashlib,time
from pathlib import Path
import numpy as np
root=Path.cwd();r=root/'reports/compositional-neural-mechanics-v16';b=root/'runs/compositional-neural-mechanics-v16/visual-bridge-development';p=json.loads((r/'visual-bridge-development-plan.json').read_text());d=json.loads((r/'visual-bridge-development-results.json').read_text());errors=[];count=0
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ck(x,s):
 global count
 count+=1
 if not x:errors.append(s)
def eq(x,y,s):ck(np.allclose(x,y,rtol=1e-10,atol=1e-14),s)
ck(d['plan_sha256']==sha(r/'visual-bridge-development-plan.json'),'plan')
for table in ['sources','inputs']:
 for path,h in p[table].items():ck(sha(root/path)==h,path)
for path,h in d['evidence_sha256'].items():ck(sha(root/path)==h,path)
q=np.array([[x,y,z]for x in [0,.5,1]for y in [0,.5,1]for z in [0,.5,1]if(x,y,z)!=(.5,.5,.5)])
for row in d['comparisons']:
 i=row['world'];method=row['method'];state=[]
 for name in [method,'analytic_n4_reference']:
  with np.load(b/f'world{i}-{name}-state.npz')as a:
   ids=np.argmin(np.sum((q[:,None,:]-a['material_nodes'][None,:,:])**2,axis=-1),axis=1);eq(a['material_nodes'][ids],q,'exact tracked nodes');state.append((a['displacement'][ids].copy(),a['wrenches'].copy()))
 diff=state[0][0]-state[1][0];w=state[0][1]-state[1][1]
 for k,v in [('position_coordinate_rmse_m',np.sqrt(np.mean(diff**2))),('reaction_force_coordinate_rmse_N',np.sqrt(np.mean(w[:,:3]**2))),('reaction_torque_coordinate_rmse_Nm',np.sqrt(np.mean(w[:,3:]**2)))]:eq(v,row['physical'][k],k)
 for cam,stored in row['views'].items():
  with np.load(b/f'world{i}-{method}-{cam}.npz')as a,np.load(b/f'world{i}-analytic_n4_reference-{cam}.npz')as c:
   ma=np.isfinite(a['depth']);mb=np.isfinite(c['depth']);union=ma|mb;joint=ma&mb;dr=a['rgb']-c['rgb'];val={'rgb_global_rmse':np.sqrt(np.mean(dr**2)),'rgb_foreground_union_rmse':np.sqrt(np.mean(dr[union]**2)),'depth_joint_rmse_m':np.sqrt(np.mean((a['depth'][joint]-c['depth'][joint])**2)),'silhouette_iou':joint.sum()/union.sum(),'foreground_union_pixels':union.sum(),'joint_depth_pixels':joint.sum(),'missing_predicted_pixels':(mb&~ma).sum(),'extra_predicted_pixels':(ma&~mb).sum()}
   for k,v in val.items():eq(v,stored[k],k)
a=d['scale_ambiguity'];eq(a['scaled_forces_N'],2*np.array(a['reference_forces_N']),'force scaling');eq(a['scaled_torques_Nm'],2*np.array(a['reference_torques_Nm']),'torque scaling');ck(a['same_equilibrium_scaled_residual_N']<1e-8 and a['images_bit_identical'],'scale identity diagnostics')
o={'passed':not errors,'checks':count,'errors':errors,'created_unix':time.time(),'plan_sha256':sha(r/'visual-bridge-development-plan.json'),'result_sha256':sha(r/'visual-bridge-development-results.json'),'physical_comparisons':[{'world':x['world'],'method':x['method'],**x['physical']}for x in d['comparisons']],'scope':'Independent saved-state exact-node physical metrics and12 RGB/depth/mask metric comparisons. Same solved Q2 field/known fixed material-coordinate appearance; no image warp, learned perception, calibrated sensor, or end-to-end vision claim. Co-scale control uses same validated equilibrium; its image identity statement source/test-backed, not separate saved control images.'}
(r/'visual-bridge-final-independent-audit.json').write_text(json.dumps(o,indent=2)+'\n');print(json.dumps(o,indent=2))
