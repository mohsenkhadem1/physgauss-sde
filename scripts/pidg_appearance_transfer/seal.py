from pathlib import Path
import json,hashlib
root=Path.cwd();candidates=[]
for version in [1,2,3]:
 p=root/f'runs/pidg-appearance-transfer-v{version:02d}'
 for phase in ['appearance','joint','refined']:
  f=p/f'{phase}-complete.json'
  if not f.exists():continue
  complete=json.loads(f.read_text());assert complete['completed'];candidates.append({'run':str(p.relative_to(root)),'checkpoint':f'{phase}-best.pt','validation_foreground_psnr':complete['validation_foreground_psnr'],'phase':phase})
best=max(candidates,key=lambda r:r['validation_foreground_psnr']);best=dict(best,checkpoint_sha256=hashlib.sha256((root/best['run']/best['checkpoint']).read_bytes()).hexdigest(),completed=True,criterion='Highest mean validation foreground PSNR across all retained candidates; no new test-image fitting',candidates=candidates)
(root/'reports/pidg-appearance-transfer-v01/selection.json').write_text(json.dumps(best,indent=2)+'\n');print(best)
