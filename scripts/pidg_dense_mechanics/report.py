"""Aggregate all declared scenes without changing model selection after tests."""
from pathlib import Path
import json,math,hashlib
ROOT=Path(__file__).resolve().parents[2];REPORT=ROOT/'reports/pidg-dense-mechanics-v01'
SCENES=['balls_reaction','rubber_duck','motion_kuro'];rows=[];checks=[]
for scene in SCENES:
 p=REPORT/scene/'results.json'
 if not p.exists():raise RuntimeError('Incomplete declared scene: '+scene)
 r=json.loads(p.read_text());assert r['completed'];selected=r['selection']['selected'];assert selected==max([c for c in r['selection']['candidates'] if c['weight']>0],key=lambda c:c['validation']['foreground_psnr_db'])['name']
 for name,sha in r['selection']['checkpoint_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name
 for width in ['320','640','1600']:
  native=r['metrics']['native-refined'][width];ours=r['metrics'][selected][width];assert [x['frame'] for x in native['rows']]==[x['frame'] for x in ours['rows']];assert native['frame_count']==(38 if scene=='balls_reaction' else 50)
  for doc in [native,ours]:
   for k,v in doc['metrics'].items():assert math.isfinite(v) and abs(v-sum(x[k] for x in doc['rows'])/len(doc['rows']))<1e-6
  n=native['metrics'];o=ours['metrics'];delta={k:o[k]-n[k] for k in ['psnr_db','foreground_psnr_db','ssim','lpips_alex','sampled_inversion_fraction','sampled_abs_volume_change']};parity=delta['psnr_db']>=-.5 and delta['foreground_psnr_db']>=-.5 and delta['lpips_alex']<=.005
  rows.append({'scene':scene,'width':int(width),'images':native['frame_count'],'selected':selected,'native':n,'ours':o,'delta_ours_minus_native':delta,'meets_predeclared_render_parity':parity})
 checks.append({'scene':scene,'checkpoint_hashes_verified':True,'validation_choice_verified':True,'metrics_recomputed':True,'test_ids_matched':True})
out={'completed':True,'scenes':SCENES,'rows':rows,'checks':checks,'test_images':138,'primary_resolution':320,'primary_scenes_meeting_parity':sum(r['meets_predeclared_render_parity'] for r in rows if r['width']==320),'limits':'These are three scenes from one public dataset, not three independent datasets. Matched reproduction at 320 training resolution. The dense mechanical regularizer retains native appearance and motion parameterization. Numerical geometry diagnostics are not ground-truth physical accuracy; not a new-action forecast.'};(REPORT/'summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ['rows','checks']},indent=2))
lines=['PIDG dense mechanics: complete multi-scene comparison','',out['limits'],'']
for r in rows:
 lines.append(f"{r['scene']} width {r['width']}: PSNR {r['native']['psnr_db']:.3f} -> {r['ours']['psnr_db']:.3f}; foreground {r['native']['foreground_psnr_db']:.3f} -> {r['ours']['foreground_psnr_db']:.3f}; LPIPS {r['native']['lpips_alex']:.5f} -> {r['ours']['lpips_alex']:.5f}; parity={r['meets_predeclared_render_parity']}")
(REPORT/'REPORT.txt').write_text('\n'.join(lines)+'\n')
