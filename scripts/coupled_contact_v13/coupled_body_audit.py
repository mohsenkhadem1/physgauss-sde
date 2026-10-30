"""Independent dynamic-sphere development audit, separate from soft-node metrics."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[2];REPORT=ROOT/'reports/coupled-surface-v13'

def compare(path_a,path_b,limit):
    paths=[ROOT/path_a/'trajectory.npz',ROOT/path_b/'trajectory.npz'];a,b=[dict(np.load(p)) for p in paths]
    finite=all(np.isfinite(r['centers']).all() and np.isfinite(r['times']).all() for r in (a,b))
    same=a['centers'].shape==b['centers'].shape and a['times'].shape==b['times'].shape and np.allclose(a['times'],b['times'],rtol=0,atol=1e-10)
    error=maximum=None
    if finite and same:
        delta=a['centers'][1:,-1]-b['centers'][1:,-1]
        error=float(np.sqrt(np.mean(delta**2)));maximum=float(np.max(np.linalg.norm(delta,axis=1)))
    return dict(paths=[str(p.relative_to(ROOT)) for p in paths],sha256=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths],all_saved_centers_finite=finite,matched_grids=same,coordinate_rms_m=error,max_euclidean_discrepancy_m=maximum,limit_m=limit,passed=bool(error is not None and error<=limit))

def main():
    arg=argparse.ArgumentParser();arg.add_argument('--extra-pairs',help='JSON list of {type,young,friction,path_a,path_b}; predeclared comparisons only');args=arg.parse_args()
    protocol=json.loads((REPORT/'coupled-body-qualification-protocol.json').read_text())
    manifest=json.loads((ROOT/'reports/coupled-contact-v13/clipped-development-qualification.json').read_text())
    runs={tuple(r['spec']):r for r in manifest['runs'] if r.get('success')};spacing=manifest['accepted_sphere_spacing_m'];rows=[];pending=[]
    for e,mu in [(6000.,.35),(3000.,.1)]:
        keys=[('moving_sphere',e,mu,5e-5,h,1) for h in (spacing,spacing/2)]
        if all(k in runs for k in keys):rows.append(dict(type='sphere_quadrature',young=e,friction=mu,**compare(runs[keys[0]]['path'],runs[keys[1]]['path'],protocol['gates']['sphere_quadrature_center_coordinate_rms_m'])))
        else:pending.append(dict(type='sphere_quadrature',young=e,friction=mu))
        keys=[('moving_sphere',e,mu,dt,spacing,1) for dt in (5e-5,2.5e-5)]
        if all(k in runs for k in keys):rows.append(dict(type='temporal',young=e,friction=mu,**compare(runs[keys[0]]['path'],runs[keys[1]]['path'],protocol['gates']['temporal_center_coordinate_rms_m'])))
        else:pending.append(dict(type='temporal',young=e,friction=mu))
        pending.append(dict(type='spatial',young=e,friction=mu))
    if args.extra_pairs:
        for pair in json.loads(Path(args.extra_pairs).read_text()):
            tag={k:pair[k] for k in ('type','young','friction')};limit=protocol['gates'][pair['type']+'_center_coordinate_rms_m']
            rows.append(dict(**tag,**compare(pair['path_a'],pair['path_b'],limit)))
            pending=[r for r in pending if r!=tag]
    report=dict(scope='Dynamic payload sphere center; allsavedcentersfinite; postinitial coordinateRMS acrosssame savedtimes. Two developmentmaterials only; no fresh reservedoutputs.',protocol_sha256=hashlib.sha256((REPORT/'coupled-body-qualification-protocol.json').read_bytes()).hexdigest(),comparisons=rows,pending=pending,all_available_pass=all(r['passed'] for r in rows),complete=not pending,all_gates_pass=not pending and all(r['passed'] for r in rows),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (REPORT/'coupled-body-qualification-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
