"""Prospectively frozen read-only whole-element diagnostic on saved fields."""
import argparse, hashlib, json, os, time
from pathlib import Path
import numpy as np
from experiments.campaign_p2_geometry import P2Geometry
from experiments.campaign_p2_jacobian_certificate import whole_p2_jacobian_bounds

ROOT = Path(__file__).resolve().parents[1]
OP = ROOT/'runs/operator-dev/p2-bernstein-saved-v01'
OUT = ROOT/'runs/gaussian-campaign-v01/p2-bernstein-saved-v01'
SOURCES = ['experiments/campaign_p2_jacobian_certificate.py',
           'experiments/run_p2_saved_jacobian_certificate.py',
           'tests/test_campaign_p2_jacobian_certificate.py',
           'experiments/campaign_p2_geometry.py',
           'experiments/campaign_static_p2_block.py',
           'experiments/campaign_block_surface_controls.py',
           'experiments/boundary_block.py']

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p, d):
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists(): raise FileExistsError('Preserve prior diagnostic')
    p.write_text(json.dumps(d, indent=2, allow_nan=False)+'\n')

def prepare():
    previous = ROOT/'runs/gaussian-campaign-v01/p2-graded-static-v02/summary.json'
    d = json.loads(previous.read_text())
    if d.get('completed') is not True: raise ValueError('Completed reference required')
    cases = []
    for r in d['meshes']:
        if r['physics_passed'] is not True: raise ValueError('Original physical gates required')
        cases.append(dict(cells=r['cells'], capture=r['capture'],
                          old_norm_bound=r['diagnostics']['global_affine_F_positive_J_sufficient_lower_bound']))
    p = dict(schema='C38-saved-P2-Bernstein-v01', no_new_physics_solve=True,
             source_sha256={n:sha(ROOT/n) for n in SOURCES},
             previous=dict(path=str(previous.relative_to(ROOT)), sha256=sha(previous)),
             cases=cases, CPU_threads=2, chunk_elements=4096,
             interpretation='Positive bound certifies whole computed affine-F polynomial; nonpositive is inconclusive; old diagnostic unchanged',
             limits=['Supplied FP64 reference gradients, not interval inversion',
                     'No spatial/time/quadrature admission from this diagnostic'])
    write(OP/'protocol.json', p)
    print(sha(OP/'protocol.json'))

def run(digest):
    path=OP/'protocol.json'
    if sha(path)!=digest: raise ValueError('Protocol changed')
    p=json.loads(path.read_text())
    for n,h in p['source_sha256'].items():
        if sha(ROOT/n)!=h: raise ValueError('Source changed')
    for r in [p['previous']]+[c['capture'] for c in p['cases']]:
        if sha(ROOT/r['path'])!=r['sha256']: raise ValueError('Saved evidence changed')
    start=time.monotonic();rows=[]
    for c in p['cases']:
        m=P2Geometry(tuple(c['cells']), mapping='clamp_graded')
        with np.load(ROOT/c['capture']['path'], allow_pickle=False) as z:
            np.testing.assert_array_equal(m.rest,z['rest_m'])
            np.testing.assert_array_equal(m.elements,z['elements'])
            b=whole_p2_jacobian_bounds(m,z['positions_m'],chunk_size=p['chunk_elements'])
        rows.append(dict(**c,minimum_Bernstein_lower_bound=b['minimum_lower_bound'],
                         maximum_Bernstein_upper_bound=float(b['upper'].max()),
                         all_elements_positive=b['all_elements_positive'],
                         inconclusive_elements=int(np.sum(b['lower']<=0)),
                         minimum_raw_coefficient=float(b['minimum_coefficient'].min()),
                         scope=b['scope']))
    result=dict(completed=True,passed=all(r['all_elements_positive'] for r in rows),
                protocol_sha256=digest,meshes=rows,elapsed_s=time.monotonic()-start,
                no_new_physics_solve=True,old_diagnostics_unchanged=True)
    write(OUT/'summary.json',result)
    print(json.dumps(result))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','run']);a.add_argument('--sha');a=a.parse_args()
    if a.action=='prepare':prepare()
    else:run(a.sha)
