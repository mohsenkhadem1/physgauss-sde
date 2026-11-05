"""Audit the real data split, initialization parity, and evaluation reload."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def main():
    data=ROOT/'runs/pidg-comparison-v01/data/balls_reaction';manifest=json.loads((data/'manifest.json').read_text());checks=[]
    def check(name,value):
        checks.append({'name':name,'passed':bool(value)})
    sets=[set(manifest[k+'_ids']) for k in ['train','validation','test']]
    check('150 distinct released images',len(set.union(*sets))==150)
    check('train validation test disjoint',not(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2]))
    fit=data/'fit320';tr=json.loads((fit/'transforms_train.json').read_text())['frames'];va=json.loads((fit/'transforms_test.json').read_text())['frames']
    check('training directory exposes no test images',all(not(fit/(f+'.png')).exists() for f in sets[2]))
    check('common timestamp convention',all(abs(f['time']-(int(Path(f['file_path']).name)-1)/149)<1e-12 for f in tr+va))
    check('training neighbors chronological',all(a['time']<b['time'] for a,b in zip(tr,tr[1:])))
    expected={f'flow_fwd_{Path(f["file_path"]).name}.flo' for f in tr[:-1]}|{f'flow_bwd_{Path(f["file_path"]).name}.flo' for f in tr[1:]}
    check('flows exactly cover retained training neighbors',{p.name for p in (fit/'flows_flo').glob('*.flo')}==expected)
    check('depth only on fitting images',{p.stem for p in (fit/'depth_distill').glob('*.npy')}=={Path(f['file_path']).name for f in tr})
    paths=[ROOT/'runs/pidg-comparison-v01'/n/'input/points3d.ply' for n in ['full-pidg','full-ours-001','full-ours-010']]
    check('identical random Gaussian initialization',len({hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})==1)
    log=(ROOT/'runs/pidg-comparison-v01/pilot-full-pidg.log').read_text();score=float(re.findall(r'Evaluating test: L1 [\d.e+-]+ PSNR ([\d.e+-]+)',log)[-1])
    reload=json.loads((ROOT/'runs/pidg-comparison-v01/pilot-full-pidg/validation/metrics.json').read_text())['metrics']['psnr_channel_mean_db']
    check('checkpoint reload reproduces author validation score',abs(score-reload)<1e-5)
    out={'all_passed':all(c['passed'] for c in checks),'checks':checks,'upstream_validation_score':score,'reloaded_validation_score':reload,'limitations':'Does not establish physical accuracy or author leaderboard reproduction. Mechanics unit tests are separate.'}
    (ROOT/'reports/pidg-comparison-v01/adapter-audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));assert out['all_passed']
if __name__=='__main__':main()
