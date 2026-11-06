"""Package the audited paper source without experiment banks or recordings."""
from pathlib import Path
import hashlib,json,zipfile

ROOT=Path(__file__).resolve().parents[2]
PAPER=ROOT/'papers/cvpr2027-joint-identification'

def main():
    macros=(PAPER/'sections/results-macros.tex').read_text()
    assert 'Working draft' not in macros, 'Final result extraction must precede release'
    provenance=json.loads((PAPER/'results-provenance.json').read_text())
    for name,digest in provenance.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    files=[p for p in PAPER.iterdir() if p.suffix in ('.tex','.sty','.bib','.bst')]
    files += list((PAPER/'sections').glob('*.tex'))
    files += list((PAPER/'figures').glob('*.pdf'))
    files += [PAPER/name for name in ['README.txt','build.sh','legacy-supporting-evidence.pdf','results-provenance.json']]
    files=sorted(set(files)); release=PAPER/'release';release.mkdir(exist_ok=True)
    manifest={str(p.relative_to(PAPER)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (release/'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(release/'latex-source.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in files:
            info=zipfile.ZipInfo(str(p.relative_to(PAPER)),date_time=(2026,11,6,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=(0o100755 if p.name=='build.sh' else 0o100644)<<16
            z.writestr(info,p.read_bytes())
        z.writestr('source-manifest.json',json.dumps(manifest,indent=2)+'\n')
    print(f'Packaged {len(files)} files; {(release/"latex-source.zip").stat().st_size} bytes')

if __name__=='__main__':main()
