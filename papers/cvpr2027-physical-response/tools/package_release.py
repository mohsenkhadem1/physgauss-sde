"""Package only self-contained review sources and verify a clean-room build."""
from pathlib import Path
import argparse, hashlib, json, re, shutil, subprocess, tempfile, zipfile
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--tectonic',default=shutil.which('tectonic') or '/usr/lib/chatgpt/resources/tectonic/tectonic');args=a.parse_args()
 files={R/n for n in ['main.tex','supplement.tex','references.bib','cvpr.sty','ieeenat_fullname.bst']};queue=[R/'main.tex',R/'supplement.tex']
 while queue:
  s=queue.pop().read_text()
  for name in re.findall(r'\\input\{([^}]+)\}',s):
   p=R/(name+'.tex')
   if p not in files:files.add(p);queue.append(p)
  files.update(R/n for n in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',s))
 archive=R/'release/latex-source.zip'
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
  for p in sorted(files):z.write(p,p.relative_to(R))
  z.write(R/'release/README-anonymous.txt','README.txt')
 manifest={str(p.relative_to(R)):sha(p) for p in sorted(files)}
 (R/'evidence/source-package-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 with tempfile.TemporaryDirectory(prefix='cvpr-cohesive-review-') as tmp:
  t=Path(tmp)
  with zipfile.ZipFile(archive) as z:z.extractall(t)
  checks={}
  for name in ['main','supplement']:
   run=subprocess.run([args.tectonic,str(t/(name+'.tex')),'--keep-logs'],capture_output=True,text=True)
   if run.returncode:raise RuntimeError(run.stdout+run.stderr)
   log=(t/(name+'.log')).read_text(errors='replace')
   assert not any(v in log for v in ['Overfull \\hbox','Overfull \\vbox','There were undefined references','Citation `']),name
   extracted=subprocess.check_output(['pdftotext','-layout',str(t/(name+'.pdf')),'-'],text=True)
   original=subprocess.check_output(['pdftotext','-layout',str(R/(name+'.pdf')),'-'],text=True)
   assert extracted==original,(name,'clean build text mismatch')
   assert '??' not in extracted and '\ufffd' not in extracted
   assert not any(v in extracted.lower() for v in ['bayespc2','mohsenkhadem','/home/','todo'])
   pages=[p for p in extracted.split('\f') if p.strip()]
   if name=='main':assert len(pages)==9 and 'LLM Usage Statement' in pages[8]
   checks[name]={'pages':len(pages),'text_matches_delivered':True,'source_pdf_sha256':sha(R/(name+'.pdf')),'clean_build_exit':0,'no_overfull_or_unresolved':True}
  report={'passed':True,'archive_sha256':sha(archive),'source_files':len(files),'builds':checks,'scope':'Build from isolated source archive without datasets; content/fonts/references checks. Human page-level visual reviews recorded separately.'}
  (R/'evidence/package-build-audit.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({'passed':True,'source_files':len(files),'pdf_pages':{k:v['pages'] for k,v in checks.items()}}))
if __name__=='__main__':main()
