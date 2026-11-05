"""Freeze measured rendering evidence and generate the manuscript table."""
from pathlib import Path
import json,hashlib,math,shutil
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'reports/pidg-appearance-transfer-v01';PAPER=ROOT/'papers/cvpr2027-physical-response'
selection=json.loads((OUT/'selection.json').read_text());run=ROOT/selection['run'];r=json.loads((run/'evaluation/results.json').read_text());old=json.loads((ROOT/'runs/pidg-mechanics-v04/evaluation/results.json').read_text());native={}
for w,folder in [(320,'test'),(640,'test-render-640'),(1600,'test-render-1600')]:
 p=ROOT/f'runs/pidg-comparison-v02/refined-pidg/{folder}/metrics.json';native[str(w)]=json.loads(p.read_text())['metrics']
assert json.loads((run/'freeze-reload-audit.json').read_text())['maximum_difference_from_cached_validation_db']<1e-4
assert len(r['rows'])==114 and len(r['novel_views'])==21
for width in [320,640,1600]:
 rows=[x for x in r['rows'] if x['width']==width];assert len(rows)==38 and len({x['frame'] for x in rows})==38
 for k,v in r['means'][str(width)].items():assert math.isfinite(v) and abs(sum(x[k] for x in rows)/38-v)<1e-8
assert hashlib.sha256((run/selection['checkpoint']).read_bytes()).hexdigest()==selection['checkpoint_sha256']
assert selection['validation_foreground_psnr']==max(x['validation_foreground_psnr'] for x in selection['candidates'])
splits={}
for name,path in [('fit','fit/transforms_train.json'),('validation','fit/transforms_test.json'),('test','evaluation/transforms_test.json')]:
 d=json.loads((ROOT/'runs/pidg-comparison-v01/data/balls_reaction'/path).read_text());splits[name]={Path(f['file_path']).name for f in d['frames']}
assert [len(splits[k]) for k in ['fit','validation','test']]==[101,11,38]
assert not (splits['fit']&splits['validation'] or splits['fit']&splits['test'] or splits['validation']&splits['test'])
report={'completed':True,'selection':selection,'split_counts':{k:len(v) for k,v in splits.items()},'native':native,'coarse':old['means'],'transferred':r['means'],'rows':r['rows'],'novel_views':r['novel_views'],'mechanics_diagnostics':r['mechanics_diagnostics'],'limitations':r['limits'],'training_resolution':[320,180],'transfer_gaussians':50524,'material_descriptors':'Inherited and fixed during this refinement','conclusions':'Improved rendering over our coarse initialized mechanics adapter; no claimed native PIDG superiority or quantitative novel-camera accuracy.'}
(OUT/'results.json').write_text(json.dumps(report,indent=2)+'\n')
for v in [1,2,3]:
 src=ROOT/f'runs/pidg-appearance-transfer-v{v:02d}';dest=OUT/f'candidate-v{v:02d}';dest.mkdir(exist_ok=True)
 for p in src.glob('*.json'):shutil.copyfile(p,dest/p.name)
lines=[r'\paragraph{Measured reconstruction results.}',r'Table~\ref{tab:pidg-rendering} reports all 38 test images. Transferring appearance and refining boundary targets improves the connected model, while the unconstrained PIDG reconstruction remains stronger on recorded cameras. Larger output images use the same $320\times180$ training. These are held-image reconstruction scores, not forecasts of a new force.',r'\begin{center}',r'\captionof{table}{\textbf{PIDG Balls Reaction, matched image split.} Full-image / foreground PSNR (dB; higher is better), SSIM (higher) and AlexNet LPIPS (lower). The transferred model uses PIDG appearance initialization.}',r'\label{tab:pidg-rendering}',r'\begin{tabular}{llrrrr}',r'\toprule Output width & Method & PSNR & FG PSNR & SSIM & LPIPS \\',r'\midrule']
for w in ['320','640','1600']:
 for label,d in [('PIDG',native[w]),('Ours: coarse appearance',old['means'][w]),('Ours: transferred appearance',r['means'][w])]:lines.append(f"{w} & {label} & {d['psnr_db']:.2f} & {d['foreground_psnr_db']:.2f} & {d['ssim']:.4f} & {d['lpips_alex']:.4f} "+r'\\')
 if w!='1600':lines.append(r'\midrule')
lines += [r'\bottomrule\end{tabular}\end{center}',r'\begin{center}\includegraphics[width=\textwidth]{figures/pidg-rendering-loop.pdf}',r'\captionof{figure}{\textbf{Image loss reaches the physical state.} Boundary controls update solved equilibrium through an implicit adjoint. Appearance-only refinement then holds the selected mechanics fixed; query-time images use mechanics-transported Gaussians.}\end{center}',r'\clearpage',r'\begin{center}\includegraphics[width=\textwidth]{figures/pidg-rendering-comparison.pdf}',r'\captionof{figure}{\textbf{Appearance refinement and camera sweep.} The middle declared test time is fixed in advance. Top: identical reference-derived crop and per-frame full-image PSNR. Bottom: fixed camera offsets, with no reference RGB at nonzero angles. The spherical geometry prior contributes to coherence; these views do not prove hidden-surface accuracy or material identification.}\end{center}',r'\begin{center}\includegraphics[width=\textwidth]{figures/pidg-rendering-metrics.pdf}',r'\captionof{figure}{\textbf{Aggregate recorded-camera reconstruction at width 640.} Foreground scores prevent the black background from dominating interpretation. Native PIDG remains ahead; the transferred mechanical renderer improves on its previous coarse initialization.}\end{center}']
(PAPER/'sections/pidg-rendering-results.tex').write_text('\n'.join(lines)+'\n')
a=r['means']['320'];b=old['means']['320'];n=native['320']
p=PAPER/'main.tex';s=p.read_text();start=s.index('\\paragraph{Connecting appearance to mechanics.}') if '\\paragraph{Connecting appearance to mechanics.}' in s else s.index('\\paragraph{Uncertainty and additional flow experiments.}');end=s.index('\\paragraph{Scope and reproducibility.}',start)
s=s[:start]+f'''\\paragraph{{Connecting appearance to mechanics.}}
On PIDG's public Balls Reaction scene, transferring fitted Gaussian appearance to our equilibrium adapter and alternating boundary/appearance refinement raises held-image PSNR from {b['psnr_db']:.2f} to {a['psnr_db']:.2f}\\,dB and foreground PSNR from {b['foreground_psnr_db']:.2f} to {a['foreground_psnr_db']:.2f}\\,dB (38 images, $320\\times180$). Image gradients reach boundary targets through the mechanics adjoint. Native PIDG remains stronger ({n['psnr_db']:.2f}\\,dB); this reconstruction-assisted study does not establish new-action prediction or hidden-view accuracy. The supplement reports all resolutions, camera sweeps and separate flow/calibration failures.\n\n'''+s[end:];p.write_text(s)
files=list((ROOT/'scripts/pidg_appearance_transfer').glob('*.py'))+[OUT/'selection.json',OUT/'results.json',run/selection['checkpoint']]
audit={'passed':True,'checks':['38 distinct test images at each of three resolutions','Finite metrics and means recomputed from rows','101/11/38 disjoint fitting/validation/test IDs','Checkpoint SHA matches pre-evaluation seal','Selected highest completed validation foreground score','21 predeclared camera views; only zero yaw has RGB reference'],'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'test_reuse_disclosed':True}
(OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(report['transferred'],indent=2))
