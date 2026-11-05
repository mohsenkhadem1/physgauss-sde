from pathlib import Path
import json,numpy as np
from PIL import Image,ImageDraw,ImageFont
root=Path(__file__).resolve().parents[2];out=root/'reports/pidg-view-audit-v03';runs=root/'runs/pidg-view-audit-v03';font=ImageFont.truetype(str(Path.home()/'.local/share/fonts/arial/Arial.TTF'),22)
names=['refined-pidg','refined-ours-00001'];man=[json.loads((runs/n/'manifest.json').read_text()) for n in names]
canvas=Image.new('RGB',(7*320,6*215+60),'white');draw=ImageDraw.Draw(canvas);draw.text((10,5),'Off-path cameras: PIDG / PIDG + our material prior. No ground-truth images for nonzero angles.',font=font,fill='black')
for k,row in enumerate(man[0]['rows']):
 t=k//7;j=k%7
 for m,name in enumerate(names):
  image=Image.open(runs/name/row['image']);image=image.resize((320,180),Image.Resampling.LANCZOS);x=j*320;y=50+(t*2+m)*215;canvas.paste(image,(x,y));draw.text((x+4,y+182),f"{'PIDG' if m==0 else 'Integration'} | {row['yaw_deg']:+d} deg | t={row['time']:.2f}",font=font,fill='black')
canvas.save(out/'novel-camera-comparison.png')
base=root/'runs/pidg-comparison-v01/data/balls_reaction';fit=json.loads((base/'fit/transforms_train.json').read_text())['frames'];test=json.loads((base/'evaluation/transforms_test.json').read_text())['frames'];poses=np.array([f['transform_matrix'] for f in fit]);rows=[]
for f in test:
 p=np.array(f['transform_matrix']);u,_,vh=np.linalg.svd(poses[:,:3,:3]);rot=u@vh;u,_,vh=np.linalg.svd(p[:3,:3]);refrot=u@vh;relative=np.einsum('nij,jk->nik',rot.transpose(0,2,1),refrot);ang=np.degrees(np.arccos(np.clip((np.trace(relative,axis1=1,axis2=2)-1)/2,-1,1)))
 rows.append({'frame':f['file_path'],'nearest_training_orientation_deg':float(ang.min()),'nearest_training_center_distance':float(np.linalg.norm(poses[:,:3,3]-p[:3,3],axis=1).min())})
metrics={n:json.loads((root/'runs/pidg-comparison-v02'/n/'test-render-1600/metrics.json').read_text()) for n in names}
report={'paper':{'published_url':'https://ojs.aaai.org/index.php/AAAI/article/download/42474/46435','balls_reaction_table1_psnr_db':33.31,'training_resolution':[1600,900],'typo_established':False},'ours_comparison':'Frozen material-prior integration in PIDG, not complete mechanics solver. Both checkpoints trained at 320x180; rendering at1600x900 is not retraining or reproduction of the paper.','test_camera_audit':rows,'nearest_training_orientation_degrees':{'min':min(r['nearest_training_orientation_deg'] for r in rows),'max':max(r['nearest_training_orientation_deg'] for r in rows),'mean':float(np.mean([r['nearest_training_orientation_deg'] for r in rows]))},'results_1600':metrics,'off_path_camera_sweeps':man,'new_view_metrics':'Unavailable without reference images; no inferred PSNR/winner.','psnr_code_audit':'Official standalone evaluator passes NCHW and averages all RGB pixels correctly. Training CHW logger channel-averages PSNR; our reports separate both. This does not establish a paper error.'}
(out/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(report['nearest_training_orientation_degrees'])
