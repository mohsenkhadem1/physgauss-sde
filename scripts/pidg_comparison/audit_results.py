"""Final independent bookkeeping checks against immutable saved predictions."""
import argparse,hashlib,json,re
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'runs/pidg-comparison-v01';REPORT=ROOT/'reports/pidg-comparison-v01'
def main():
    global BASE,REPORT
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='runs/pidg-comparison-v01');args=parser.parse_args();BASE=ROOT/args.root;REPORT=ROOT/'reports'/BASE.name
    selection=json.loads((BASE/'selection.json').read_text());report=json.loads((REPORT/'results.json').read_text());checks=[]
    def check(name,value):checks.append({'name':name,'passed':bool(value)})
    best=max(selection['candidates'],key=lambda r:r['validation_foreground_psnr_db'])['run'];check('selection follows declared validation metric',best==selection['chosen'])
    check('exactly 38 held-out frames',report['test_frames']==38 and len(report['per_frame'])==38)
    check('final test followed selection seal',all((Path(p)/'test/metrics.json').stat().st_mtime>selection['sealed_at'] for p in selection['test_runs']))
    rows=[]
    for path in selection['test_runs']:
        run=Path(path);meta=json.loads((run/'run.json').read_text());m=json.loads((run/'test/metrics.json').read_text());rows.append(m['rows'])
        expected=5000 if meta['arguments'].get('warm_from') else 50000
        check(run.name+' completed matched full budget',meta['status']=='training_complete' and meta['arguments']['steps']==expected and meta['arguments']['width']==320 and meta['arguments']['flow'])
        if meta['arguments'].get('warm_from'):
            parent=json.loads((ROOT/meta['arguments']['warm_from']/'run.json').read_text());check(run.name+' 50k pretraining plus 5k refinement',parent['status']=='training_complete' and parent['arguments']['steps']==50000)
        check(run.name+' finite and consistent metrics',all(np.isfinite(v) and abs(v-np.mean([r[k] for r in m['rows']]))<1e-9 for k,v in m['metrics'].items()))
    check('frame sets and timestamps identical',[(r['frame'],r['time']) for r in rows[0]]==[(r['frame'],r['time']) for r in rows[1]])
    for key,outkey in [('foreground_psnr_db','foreground_wins'),('psnr_db','whole_image_wins')]:check(outkey+' recomputes',sum(o[key]>b[key] for b,o in zip(*rows))==report[outkey])
    base,ours=map(Path,selection['test_runs'])
    for r in report['selected_successes']+report['least_favorable']:
        sub='test-render-640' if r['render_width']==640 else 'test';b=np.load(base/sub/f'{r["frame"]}.npz');o=np.load(ours/sub/f'{r["frame"]}.npz');x,y,w,h=r['crop_xywh'];gt=b['truth'][y:y+h,x:x+w]
        check('matched reference '+r['frame'],np.array_equal(b['truth'],o['truth']))
        for name,arr in [('pidg',b),('hybrid',o)]:check('crop score '+name+' '+r['frame'],abs(float(np.sqrt(((arr['prediction'][y:y+h,x:x+w]-gt)**2).mean()))-r['crop_rmse'][name])<1e-9)
    material=ROOT/'reports/pidg-comparison-v01/shared-material-law.pt';check('packaged weights unchanged',hashlib.sha256(material.read_bytes()).hexdigest()==json.loads((ours/'run.json').read_text())['runtime_changes']['constitutive_checkpoint_sha256'])
    result={'all_passed':all(c['passed'] for c in checks),'checks':checks,'scope':'Bookkeeping, selection, identical observations, metrics and figure crop values; not an independent validation of physical truth.'}
    (REPORT/'final-audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));assert result['all_passed']
if __name__=='__main__':main()
