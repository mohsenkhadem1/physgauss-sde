"""Chronological protocol; old full-sequence checkpoints are prohibited."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/pidg-solver-forecast-v01/data';REPORT=ROOT/'reports/pidg-solver-forecast-v01';OUT.mkdir(exist_ok=True);rows=[]
for scene in ['balls_reaction','rubber_duck','motion_kuro']:
 source=ROOT/'runs/pidg-comparison-v01/data'/scene;out=OUT/scene;out.mkdir(exist_ok=True)
 raw=out/'released'
 if not raw.exists():raw.symlink_to(source/'released',target_is_directory=True)
 train=json.loads((source/'fit/transforms_train.json').read_text());val=json.loads((source/'fit/transforms_test.json').read_text());test=json.loads((source/'evaluation/transforms_test.json').read_text())
 groups={'fit':[f for f in train['frames'] if f['time']<=.6],'appearance_validation':[f for f in val['frames'] if f['time']<=.6],'forecast_validation':[f for f in train['frames']+val['frames'] if .6<f['time']<=.7],'forecast_test':[f for f in test['frames'] if f['time']>.7]}
 for name,frames in groups.items():
  frames.sort(key=lambda f:f['time']);(out/(name+'.json')).write_text(json.dumps(dict(camera_angle_x=train['camera_angle_x'],frames=frames),indent=2))
 fit=out/'fit';fit.mkdir(exist_ok=True)
 for split,key in [('train','fit'),('test','appearance_validation')]:
  (fit/f'transforms_{split}.json').write_text(json.dumps(dict(camera_angle_x=train['camera_angle_x'],frames=groups[key]),indent=2))
  for f in groups[key]:
   dest=fit/(f['file_path']+'.png');dest.parent.mkdir(exist_ok=True)
   if not dest.exists():dest.symlink_to((raw/(f['file_path']+'.png')).resolve())
 row=dict(scene=scene,counts={k:len(v) for k,v in groups.items()},fitting_end=.6,validation_end=.7,test_end=1.,old_checkpoints_used=False,frames={k:[f['file_path'] for f in v] for k,v in groups.items()},scope='Released training images up to 60% time only for fitting; prefix withheld images for appearance validation, training images 60-70% for forecast selection, released test images after 70% for final forecasts. Previously exposed public test imagery; a new chronological protocol, not a pristine dataset.')
 (out/'protocol.json').write_text(json.dumps(row,indent=2));rows.append(row)
(REPORT/'protocol.json').write_text(json.dumps(dict(scenes=rows,metrics=['PSNR','foreground PSNR','SSIM','LPIPS-Alex','image-space motion agreement','Jacobian/strain/force residual diagnostics'],unknowns='No true forces/materials/metric units supplied; no SI identification or new-action ground-truth claims.',future_field_rule='Infer interactions on prefix only. Extrapolation rule frozen before forecast scoring; no future target-conditioned forces.'),indent=2));print([(r['scene'],r['counts']) for r in rows])
