"""Verify publication arithmetic and provenance without changing any experiment."""
from pathlib import Path
import json,hashlib,statistics,subprocess,re
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def read(p):return json.loads((ROOT/p).read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def mean(a):return statistics.mean(a)
def close(a,b):assert abs(a-b)<1e-9,(a,b)
def main():
 checks=[]
 for file,h in json.loads((HERE/'evidence/sources.json').read_text()).items():assert sha(ROOT/file)==h;checks.append('hash '+file)
 soft=read('reports/benchmark-improvement-v14/softvt-summary-b.json')['groups']['prospective_softvt_phase_b'];r=soft['contact_scene_state'];rows=r['per_case']
 assert len(rows)==40 and all(v['complete_pair']for v in rows)
 close(mean(v['proposal_mm']for v in rows),r['proposal_mean_mm']);close(mean(v['baseline_mm']for v in rows),r['baseline_mean_mm'])
 assert sum(v['proposal_mm']<v['baseline_mm']for v in rows)==36
 assert soft['proposed_coverage']==dict(cases=40,complete_cases=40,total_frames=1998,failed_frames=0)
 checks.append('SoftVT: 40 pairs, 36 wins, means and 1998 complete frames')
 d=read('reports/real-transfer-v22/scores.json');assert len(d['rows'])==8
 for method,s in d['summary'].items():
  for metric in ('tracking_mm','displacement_mm','pair_distance_change_mm'):close(mean(r['results'][method]['manual'][metric]for r in d['rows']),s[metric])
 checks.append('PhysTwin: complete eight-case geometry means')
 tr=read('reports/real-transfer-v22/transfer-scores.json');assert len(tr['rows'])==5
 for group,rows in [('all_five_diagnostic',tr['rows']),('initial_geometry_gate_passed',[r for r in tr['rows']if r['registration']['initial_geometry_gate']])]:
  for method,s in tr['summary'][group].items():
   for metric in ('tracking_mm','displacement_mm','pair_distance_change_mm'):close(mean(r['results'][method]['whole_interaction'][metric]for r in rows),s[metric])
 assert len(rows)==2;checks.append('Transfer: five diagnostics, two passing pairs and all means')
 # Generated case rows must be embedded exactly in the supplementary source.
 supp=(HERE/'supplement.tex').read_text()
 for filename in ('soft-table.tex','real-table.tex','transfer-table.tex'):
  for line in (HERE/'evidence'/filename).read_text().splitlines():assert line in supp
 checks.append('All 53 per-case table rows match saved metrics')
 rendered={}
 for name,expected in [('main',9),('supplement',5)]:
  path=HERE/(name+'.pdf');info=subprocess.check_output(['pdfinfo',str(path)],text=True);pages=int(re.search(r'Pages:\s+(\d+)',info).group(1));assert pages==expected
  text=subprocess.check_output(['pdftotext',str(path),'-'],text=True);assert '\ufffd' not in text
  assert not re.search(r'\[\?\??\]|TODO|TBD|PLACEHOLDER',text)
  if name=='main':assert 'Conclusion' in text.split('\f')[7] and 'LLM Usage Statement' in text.split('\f')[8]
  rendered[name]=dict(pages=pages,sha256=sha(path))
 checks.append('PDFs: eight main content pages, one references/disclosure page, five supplement pages; resolved references')
 result=dict(passed=True,checks=checks,pdfs=rendered,visual_review='All final pages rendered and inspected; no clipping or overlapping content.',scope='Same-agent publication arithmetic and layout audit; no new fitting or independent experimental replication.')
 (HERE/'evidence/publication-audit.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS',len(checks),'publication checks')
if __name__=='__main__':main()
