"""Independent arithmetic/provenance check of native mechanics diagnostics."""
import json,hashlib
from pathlib import Path
import numpy as np
from scipy.special import softmax
R=Path('reports/gap-crossmethod-v02');U=Path('runs/gap-crossmethod-v02/mphygs-native-contact');d=json.loads((R/'mphygs-second-interaction.json').read_text());bank=np.load(U/'second-interaction-target-bank.npz');q=bank['targets'];mu=bank['calibration_means'];checks=0
for r in d['rows']:
 w=np.array([.5,.5]) if r['stage']=='before' else softmax(-.5*((r['observation']-mu)/.00002)**2)
 j=int(w.argmax());e=np.sqrt(np.mean(np.sum((q-q[j])**2,axis=-1),axis=(1,2)));risk=float(w@(e>.0001))
 for a,b in [(r['error'],e[r['world']]),(r['risk'],risk),(r['same_evidence_MAP_error'],e[r['world']]),(r['brier'],(risk-float(e[r['world']]>.0001))**2)]:assert np.isclose(a,b,rtol=1e-12,atol=1e-15);checks+=1
for name in ['mphygs-native-smoke.json','mphygs-native-contact.json']:
 doc=json.loads((R/name).read_text())
 for p,h in doc['source_hashes'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
out={'passed':True,'checks':checks,'source_hashes_verified':True,'scope_correction':{'incorrect_old_text':'native full pipeline normally supplies density / beyond known-density native fit assumptions','corrected':'Current official material_field.py predicts density from learned logits, and predicts E through stiffness times density. Density override in test_parallel_photo_opt.py is optional. Unknown-density symmetry is relevant to its native parameterization, but this experiment does not evaluate trained material-field fitting or native rendering.','evidence':['mphygs/material_field.py lines71 and97–104','scripts/test_parallel_photo_opt.py optional --density defaultNone and material-field outputs lines536+'],'frozen_protocol_original_preserved':True},'limitations':['Finite synthetic two-world forward reference includes truth by construction.','No measured ground-truth rendering comparison or native trained inference.','Same-evidence MAP prediction has identical accuracy.','Local timing excludes initialization/JIT and is not a speed benchmark.','Scalar coordinate RMS in smoke/contact report differs from Euclidean-point RMS in second-interaction target.']}
(R/'mphygs-independent-audit.json').write_text(json.dumps(out,indent=2));print(out)
