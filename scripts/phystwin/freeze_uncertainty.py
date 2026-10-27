"""Freeze or verify the numerical inputs before held-out uncertainty evaluation."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.phystwin.uncertainty_reference import BASE, ROOT


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--verify',action='store_true');args=parser.parse_args()
    path=BASE/'analysis/pretest-manifest.json'
    if args.verify or path.exists():
        manifest=json.loads(path.read_text())
        assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in manifest.items()), 'Frozen inputs changed'
        print('Verified',len(manifest),'frozen input hashes')
        return
    files=list((BASE/'fields').glob('bank9-*.npz'))+list((BASE/'fields').glob('development-*.npz'))
    files += [BASE/'analysis/calibration-seal.json',BASE/'protocol.json',
              ROOT/'scripts/phystwin/uncertainty_reference.py',ROOT/'scripts/phystwin/uncertainty_posterior.py']
    assert len(files)==109, 'Expected completed 9x9 bank, development fields and calibration seal'
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    path.write_text(json.dumps(manifest,indent=2)+'\n')
    print('Frozen',len(manifest),'input hashes')

if __name__=='__main__':main()
