"""Compile the completed addendum; final page images still need visual review."""
import gzip,json,shutil,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.contact_benchmark_v12.common import ROOT,BASE,REPORT,sha256,write_json,datetime,timezone

def main():
    with gzip.open(REPORT/'model-selection-results.json.gz','rt') as f:result=json.load(f)
    if not result['accounting_complete']:raise RuntimeError('Paper requires all declared cases accounted for')
    paper=ROOT/'papers/contact-benchmark-v12';compiler=BASE/'tools/tectonic'
    if not compiler.exists():compiler=Path(shutil.which('tectonic') or '')
    if not compiler.is_file():raise RuntimeError('Install Tectonic to compile this source')
    sources=[paper/n for n in ['extension.tex','generated-results.tex','generated-feasibility-results.tex','generated-model-selection-results.tex']]
    sources += [REPORT/n for n in ['guarded-algorithm.png','model-selection-comparison.png']]
    if not all(p.is_file() for p in sources):raise RuntimeError('A required generated table or figure is missing')
    proc=subprocess.run([str(compiler),'--keep-logs','extension.tex'],cwd=paper,text=True,capture_output=True,timeout=180)
    (BASE/'paper-build.log').write_text(proc.stdout+proc.stderr)
    proc.check_returncode()
    if 'Overfull' in proc.stdout+proc.stderr:raise RuntimeError('Fix overfull paper boxes before delivery; see paper-build.log')
    destination=ROOT/'output/pdf/contact-benchmark-v12-extension.pdf';destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(paper/'extension.pdf',destination)
    info=subprocess.check_output(['pdfinfo',str(destination)],text=True)
    write_json(REPORT/'paper-build.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in sources},
        compiler_version=subprocess.check_output([str(compiler),'--version'],text=True).strip(),compiler_sha256=sha256(compiler),
        repository_pdf='papers/contact-benchmark-v12/extension.pdf',output_pdf=str(destination.relative_to(ROOT)),
        pdf_sha256=sha256(destination),pdfinfo=info,visual_review='Pending final rendered-page review'))
    print(destination)

if __name__=='__main__':main()
