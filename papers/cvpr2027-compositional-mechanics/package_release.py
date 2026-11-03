"""Package existing review artifacts without publishing raw arrays or checkpoints."""
from pathlib import Path
import hashlib,json,re,shutil,subprocess,zipfile
P=Path(__file__).resolve().parent;R=P.parents[1];O=R/'output/pdf/cvpr2027';D=P/'release'
O.mkdir(parents=True,exist_ok=True);D.mkdir(exist_ok=True)
readme='''Supplementary media: Compositional Neural Mechanics for Physical Prediction Beyond Visual Agreement

Read supplement.pdf for equations, complete-case and failed-case accounting,
reference sensitivity, and the scope of each experiment.

media/world-{0,1,2,3}-forecast-{40,80}mm-held0.mp4:
All eight declared held-view forecasts, with synchronized reference, our model,
and native PhysTwin. World 3 has unavailable native predictions, visibly marked.
31 saved frames cover 0.30 simulated seconds. Playback is 10 fps (10x slow motion).
There is no deformation magnification. Reference means finite-resolution
simulation, not real captured footage or a continuum-converged ground truth.

Four development examples are also included:
position-compression-development.mp4: position-driven compression/hold/unload.
finite-striker-development.mp4: conservative finite-mass sphere impact demo.
drop-bounce-settle-development.mp4: separate viscous drop/bounce/settle demo.
development-orbit-actual-ours-native-v2.mp4: development comparison with orbiting camera.
These are development examples, not additional held-cohort test successes.
Striker and viscous demonstrations have the qualification limits in supplement.pdf.
No external links, raw datasets or identifying author information are required.
'''
media=sorted((R/'reports/compositional-dynamics-v17/fresh-cohort-videos-v2').glob('*.mp4'))
media += [R/'reports/compositional-dynamics-v17/videos'/n for n in ['position-compression-development.mp4','finite-striker-development.mp4','drop-bounce-settle-development.mp4','development-orbit-actual-ours-native-v2.mp4']]
assert len(media)==12
with zipfile.ZipFile(D/'supplement-with-videos.zip','w',zipfile.ZIP_DEFLATED)as z:
 z.writestr('README.txt',readme);z.write(P/'supplement.pdf','supplement.pdf')
 for f in media:z.write(f,'media/'+f.name)
source=[P/n for n in ['main.tex','supplement.tex','references.bib','cvpr.sty','ieeenat_fullname.bst','README.txt','generate_figures.py','package_release.py']]
source+=sorted((P/'sections').glob('*.tex'))+sorted((P/'figures').glob('*.pdf'))
with zipfile.ZipFile(D/'latex-source.zip','w',zipfile.ZIP_DEFLATED)as z:
 for f in source:z.write(f,str(f.relative_to(P)))
for f in [P/'main.pdf',P/'supplement.pdf',D/'latex-source.zip',D/'supplement-with-videos.zip']:shutil.copy2(f,O/f.name)
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
check={}
for name in ['main','supplement']:
 f=P/(name+'.pdf');info=subprocess.check_output(['pdfinfo',str(f)],text=True);text=subprocess.check_output(['pdftotext',str(f),'-'],text=True)
 pages=[p for p in text.split('\f') if p.strip()]
 log=(P/(name+'.log')).read_text(errors='replace')
 assert not re.search(r'Overfull|undefined references|Citation .* undefined|Missing character',log)
 assert not any(x in text.lower()for x in ['/home/bayespc2','mohsenkhadem','github.com/mohsen'])
 assert '\ufffd' not in text
 assert all('612 x 792' in info for _ in [0])
 if name=='main':
  assert len(pages)==9 and 'LLM Usage Statement' in pages[8] and 'References' in pages[8]
  assert '7. Conclusion' in pages[7]
 check[name]={'pages':len(pages),'sha256':sha(f),'no_overfull_or_undefined_references':True,'no_author_identity_detected':True}
for name in ['cvpr.sty','ieeenat_fullname.bst']:
 assert (P/name).read_bytes()==(Path('/tmp/physgauss-cvpr2027-author-kit')/name).read_bytes()
manifest={'created_date':'2026-11-03','status':'Prepared for author review; not submitted; registration pending','pdf_checks':check,'official_template_commit':'a2369342011145be8c85e5ad21a9b61f9fa53473','content_pages':8,'bibliography_entries':len(re.findall(r'^@', (P/'references.bib').read_text(),re.M)),'media':[{'name':f.name,'sha256':sha(f),'bytes':f.stat().st_size}for f in media],'deliverables':[{'name':f.name,'sha256':sha(f),'bytes':f.stat().st_size}for f in sorted(O.iterdir())if f.suffix in ['.pdf','.zip']],'independent_review':{'method_source_review':'coupled_physics_v13','literature_review':'baselines_research','claims_audit':'protocol_audit_v13','corrections_applied':['geometry parameterization scope','CMA upper budget and early stopping','coordinate/vector metric qualifier','seeded random policy label','viscous/conservative striker qualification separation','native fitted damping distinguished','DeepSets hidden-layer count','image slot/missing forecast count']},'raw_datasets_included':False,'new_experiments_run':False}
(P/'evidence/release-validation.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'pdfs':check,'media':len(media),'package_bytes':sum(f.stat().st_size for f in D.glob('*.zip'))},indent=2))
