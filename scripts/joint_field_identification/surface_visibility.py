"""Independent visibility audit of declared surface-marker observations.

Orthographic rays are tested against actual deformed FEM boundary triangles.
Known material correspondences are supplied; this does not validate RGB tracking,
segmentation, illumination or a real camera's resolution. No failed case is removed.
"""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
import torch
from .mechanics import LinearGaussianBody
from . import surface

ROOT=surface.ROOT;REPORT=ROOT/'reports/joint-field-surface-v04'
SURFACE_TOL=1e-8;RAY_SELF_TOL=5e-8;BARY_TOL=1e-8


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def triangle_membership(point,triangles,tol=SURFACE_TOL):
    a=triangles[:,0];e1=triangles[:,1]-a;e2=triangles[:,2]-a;v=point-a
    normal=np.cross(e1,e2);norm=np.linalg.norm(normal,axis=1);distance=np.abs(np.einsum('fi,fi->f',v,normal))/norm
    d00=np.einsum('fi,fi->f',e1,e1);d01=np.einsum('fi,fi->f',e1,e2);d11=np.einsum('fi,fi->f',e2,e2);d20=np.einsum('fi,fi->f',v,e1);d21=np.einsum('fi,fi->f',v,e2);den=d00*d11-d01*d01
    u=(d11*d20-d01*d21)/den;w=(d00*d21-d01*d20)/den
    return (distance<=tol)&(u>=-BARY_TOL)&(w>=-BARY_TOL)&(u+w<=1+BARY_TOL),distance


def ray_hits(point,direction,triangles):
    # Double-sided Moller–Trumbore test. Positive ray direction points toward
    # camera. Intersections within50nm are self-hits, not foreground occluders.
    e1=triangles[:,1]-triangles[:,0];e2=triangles[:,2]-triangles[:,0];p=np.cross(direction,e2);det=np.einsum('fi,fi->f',e1,p)
    good=np.abs(det)>1e-14;inv=np.divide(1.,det,out=np.zeros_like(det),where=good);s=point-triangles[:,0];u=np.einsum('fi,fi->f',s,p)*inv;q=np.cross(s,e1);v=(q@direction)*inv;t=np.einsum('fi,fi->f',e2,q)*inv
    hit=good&(u>=-BARY_TOL)&(v>=-BARY_TOL)&(u+v<=1+BARY_TOL)&(t>RAY_SELF_TOL)
    return np.flatnonzero(hit),t[hit]


class Visibility:
    def __init__(self,shape,cells):
        self.body=LinearGaussianBody(shape,cells);self.points=surface.points();self.N=self.body.interpolation(self.points).numpy();self.rest=self.body.points.numpy();self.faces=self.body.surface_triangles.numpy();tri=self.rest[self.faces]
        normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);self.sign=np.sign(np.einsum('fi,fi->f',normal,self.body.surface_normals.numpy()))
        self.incident=[np.flatnonzero(triangle_membership(p,tri)[0]) for p in self.points]
        self.membership=[dict(point=i,coordinate_m=p.tolist(),surface_triangle_count=len(self.incident[i])) for i,p in enumerate(self.points)]

    def check(self,displacement,requests):
        xyz=self.rest+displacement;tri=xyz[self.faces];points=self.points+self.N@displacement
        normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])*self.sign[:,None];normal/=np.linalg.norm(normal,axis=1)[:,None]
        checks=[]
        for request in requests:
            index=request['point'];c=np.asarray(request['camera_direction'],float);c/=np.linalg.norm(c);incident=self.incident[index]
            if not len(incident):
                checks.append(dict(**request,visible=False,reason='reference_point_not_on_surface'));continue
            membership,distance=triangle_membership(points[index],tri[incident]);surface_error=float(distance.min());facing=float(np.max(normal[incident]@c));hits,depth=ray_hits(points[index],c,tri)
            visible=bool(membership.any() and facing>1e-10 and len(hits)==0)
            checks.append(dict(**request,visible=visible,on_deformed_surface=bool(membership.any()),surface_plane_error_m=surface_error,max_incident_facing_cosine=facing,occluding_triangle_count=len(hits),nearest_occlusion_m=float(depth.min()) if len(depth) else None))
        return checks


def all_time_requests():
    requests=[dict(name=f'initial_front_p{p}',point=p,camera_direction=[0.,-1.,0.]) for p in [1,5,9,13]]
    seen={(r['point'],tuple(r['camera_direction'])) for r in requests}
    for a in surface.action_specs():
        key=(a['point'],tuple(a['camera_direction']))
        if key not in seen:requests.append(dict(name=a['name'],point=a['point'],camera_direction=a['camera_direction']));seen.add(key)
    return requests


def static_audit():
    camera=[]
    for a in surface.action_specs():
        c=np.asarray(a['camera_direction']);e=np.asarray(a['projection']);vertical=np.array([0.,0.,1.]);horizontal=np.cross(vertical,c);horizontal/=np.linalg.norm(horizontal)
        valid=bool(abs(c@e)<1e-12 and abs(np.linalg.norm(c)-1)<1e-12 and abs(np.linalg.norm(e)-1)<1e-12 and abs(c@vertical)<1e-12 and abs(horizontal@vertical)<1e-12)
        camera.append(dict(name=a['name'],projection=e.tolist(),camera_direction=c.tolist(),orthonormal_camera_basis_valid=valid))
    shapes=[]
    for shape in ['block','step']:
        v=Visibility(shape,(8,4,4));checks=v.check(np.zeros_like(v.rest),all_time_requests());shapes.append(dict(shape=shape,point_membership=v.membership,rest_visibility=checks))
    # Independent negative control: a back-surface marker must be occluded
    # and back-facing when viewed from the front of an opaque block.
    v=Visibility('block',(8,4,4));negative=v.check(np.zeros_like(v.rest),[dict(name='back_marker_front_camera',point=3,camera_direction=[0.,-1.,0.])])[0]
    assert not negative['visible'] and negative['occluding_triangle_count']>0
    positive=v.check(np.zeros_like(v.rest),[dict(name='front_marker_front_camera',point=1,camera_direction=[0.,-1.,0.])])[0]
    assert positive['visible']
    passed=all(r['orthonormal_camera_basis_valid'] for r in camera) and all(p['surface_triangle_count']>0 for s in shapes for p in s['point_membership']) and all(r['visible'] for s in shapes for r in s['rest_visibility'])
    report=dict(passed=passed,cameras=camera,shapes=shapes,negative_control=negative,positive_control=positive,tolerances=dict(surface_m=SURFACE_TOL,self_hit_m=RAY_SELF_TOL,barycentric=BARY_TOL),scope='Reference-surface/camera geometry checks; actual deformed truths must be audited separately.')
    REPORT.mkdir(exist_ok=True,parents=True);(REPORT/'visibility-static.json').write_text(json.dumps(report,indent=2));return report


def audit_truth(path,shape,population,output):
    truth=dict(np.load(path));cells=(12,6,6) if population=='offgrid-fine' else (8,4,4);v=Visibility(shape,cells);requests=all_time_requests();rows=[];start=time.monotonic()
    for case,theta in enumerate(truth['theta']):
        with torch.no_grad():
            solved=v.body.solve(theta);unit=solved['unit_response'][0].numpy();probes=solved['probe_response'][0].numpy()
        replay=np.einsum('ncl,tl->tnc',unit,truth['force'][case].reshape(8,2));allchecks=[]
        for t,d in enumerate(replay):
            allchecks.extend([dict(state='original_load',time_index=t,**r) for r in v.check(d,requests)])
        for a in surface.action_specs():
            if a['kind']!='probe':continue
            q=a['probe'];request=dict(name=a['name'],point=a['point'],camera_direction=a['camera_direction'])
            allchecks.extend([dict(state='calibrated_probe',probe_index=q,**r) for r in v.check(surface.PROBE_FORCE*probes[...,q],[request])])
        failures=[r for r in allchecks if not r['visible']]
        rows.append(dict(case=case,checks=len(allchecks),failed_checks=len(failures),failures=failures,max_surface_plane_error_m=max(r.get('surface_plane_error_m',0) for r in allchecks),minimum_facing_cosine=min(r.get('max_incident_facing_cosine',-1) for r in allchecks)))
    report=dict(shape=shape,population=population,cases=len(rows),cells=list(cells),checks=sum(r['checks'] for r in rows),failed_checks=sum(r['failed_checks'] for r in rows),failed_cases=sum(r['failed_checks']>0 for r in rows),rows=rows,seconds=time.monotonic()-start,truth_path=str(path.relative_to(ROOT)),truth_sha256=sha(path),source_sha256={str(Path(__file__).relative_to(ROOT)):sha(__file__),'scripts/joint_field_identification/surface.py':sha(Path(__file__).with_name('surface.py')),'scripts/joint_field_identification/mechanics.py':sha(Path(__file__).with_name('mechanics.py'))},scope='All actual truth histories at all8times and all6candidate known-probe states. All failures retained; no data or policy changes. Supplied material correspondences, orthographic cameras, opaque FEM surface; not a test of RGB feature tracking.')
    output.parent.mkdir(exist_ok=True,parents=True);output.write_text(json.dumps(report,indent=2));print(shape,population,report['cases'],report['failed_checks'],'visibility failures',round(report['seconds'],2),'s',flush=True);return report


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--static',action='store_true');ap.add_argument('--hierarchy-root',type=Path);ap.add_argument('--hierarchy-report',type=Path);args=ap.parse_args();
    if args.hierarchy_root:args.hierarchy_root=args.hierarchy_root.resolve()
    if args.hierarchy_report:args.hierarchy_report=args.hierarchy_report.resolve()
    torch.set_num_threads(1);static=static_audit();print('Static visibility',static['passed'],flush=True)
    if args.static:return
    results=[];missing=[]
    for shape in ['block','step']:
        for population in ['matched','continuous-coarse','offgrid-fine','force-prior']:
            path=ROOT/f'runs/joint-field-surface-v04/{shape}/{population}-truth.npz'
            if not path.exists():missing.append(str(path.relative_to(ROOT)));continue
            output=REPORT/shape/f'{population}-visibility.json'
            if output.exists():
                saved=json.loads(output.read_text())
                if saved['truth_sha256']==sha(path) and saved['source_sha256'].get(str(Path(__file__).relative_to(ROOT)))==sha(__file__):results.append(saved);continue
            results.append(audit_truth(path,shape,population,output))
    if args.hierarchy_root:
        for shape in ['block','step']:
            path=args.hierarchy_root/shape/'truth.npz'
            if not path.exists():missing.append(str(path));continue
            out=(args.hierarchy_report or REPORT)/shape/'hierarchy-visibility.json'
            results.append(audit_truth(path,shape,'hierarchy',out))
    summary=dict(static_passed=static['passed'],cases=sum(r['cases'] for r in results),checks=sum(r['checks'] for r in results),failed_cases=sum(r['failed_cases'] for r in results),failed_checks=sum(r['failed_checks'] for r in results),missing=missing,complete=not missing and sum(r['cases'] for r in results)==320+(128 if args.hierarchy_root else 0),passed=static['passed'] and not missing and sum(r['failed_checks'] for r in results)==0,expected_v04_cases=320,expected_hierarchy_cases=128 if args.hierarchy_root else None,scope='Visibility checks only; no filtering or reranking. Known material correspondences do not constitute raw RGB inference.')
    (REPORT/'visibility-summary.json').write_text(json.dumps(summary,indent=2));print(summary,flush=True)

if __name__=='__main__':main()
