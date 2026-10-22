"""Independent manufactured geometry/metric and completed-metadata controls.

No nonlinear campaign array, actual score, observation or physics solve is read.
"""
import copy,json,hashlib
from pathlib import Path
import numpy as np
import pytest
from experiments import score_campaign_c38_static_linear as m
from experiments.campaign_static_p2_block import enrich


def points():return np.array([[x,y,z] for x in (-1.,1.) for y in (-1.,1.) for z in (-1.,1.)])
def cloud(X,P,T,w=None):
    if w is None:w=np.ones(len(X))
    return lambda:iter([(X,P,T,w)])


def test_rigid_fit_exact_and_corotated_deformation_metrics_have_analytic_values():
    X=points();a=.4;R=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1.]])
    fits,metrics=m.bulk_rigid_metrics(cloud(X,X@R+[.3,.2,-.4],X@R+[.3,.2,-.4]))
    assert metrics['linear_coordinate_RMS_m']<1e-16
    assert metrics['target_nonrigid_coordinate_RMS_m']<5e-16
    assert metrics['corotated_deformation_error_coordinate_RMS_m']<1e-16
    np.testing.assert_allclose(fits['target_rotation'],R,atol=1e-15)
    _,d=m.bulk_rigid_metrics(cloud(X,1.01*X+[.3,.2,-.4],1.02*X+[.3,.2,-.4]))
    assert d['target_nonrigid_coordinate_RMS_m']==pytest.approx(.02,abs=5e-16)
    assert d['corotated_deformation_error_coordinate_RMS_m']==pytest.approx(.01,abs=5e-16)
    assert d['target_rigid_coordinate_RMS_m']==pytest.approx(.02,abs=5e-16)


def test_bulk_metric_rigid_frame_invariance_and_positive_weights_only():
    X=points();P=X*np.array([1.01,1.02,.99])+[.02,0,0];T=X*np.array([1.02,1.03,1.])+[0,.01,0]
    w=np.arange(1,9,dtype=float);_,before=m.bulk_rigid_metrics(cloud(X,P,T,w))
    a=.7;R=np.array([[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]]);shift=[.2,-.4,.8]
    _,after=m.bulk_rigid_metrics(cloud(X@R+shift,P@R+shift,T@R+shift,w))
    for key in before:
        if key.endswith('_RMS_m'):assert after[key]==pytest.approx(before[key],abs=2e-15)
    bad=w.copy();bad[0]=-1
    with pytest.raises(ValueError,match='strictly positive'):m.bulk_rigid_metrics(cloud(X,P,T,bad))


def tetra():
    corners=np.array([[0.,0,0],[.2,0,0],[0,.1,0],[0,0,.08]])
    X,e=enrich(corners,np.array([[0,1,2,3]]));return corners,X,e


def test_positive_keast_volume_metric_matches_independent_uniform_simplex_moment():
    corners,X,e=tetra();P=1.01*X+[.01,0,0];T=1.02*X+[.01,0,0]
    chunks=m.positive_volume_chunks(X,e,P,T,element_chunk=1)
    assert all(np.all(w>0) for *_,w in chunks())
    _,result=m.bulk_rigid_metrics(chunks)
    centered=corners-corners.mean(0);cov=centered.T@centered/20
    expected=.01*np.sqrt(np.trace(cov)/3)
    assert result['corotated_deformation_error_coordinate_RMS_m']==pytest.approx(expected,rel=1e-12,abs=1e-17)
    assert result['reference_volume_m3']==pytest.approx(.2*.1*.08/6,rel=1e-14)


def data():
    _,X,e=tetra();basis=dict(rest_m=X,elements=e,tets=e[:,:4],faces=np.array([[0,1,2]]),constraint_mask=np.zeros_like(X,dtype=bool))
    prediction=dict(positions_m=X+[.001,0,0],material_probes_m=m.linear.QUERY_POINTS.copy(),probe_positions_m=m.linear.QUERY_POINTS+[.001,0,0])
    target=dict(**basis,positions_m=X+[.002,0,0],material_probes_m=m.linear.QUERY_POINTS.copy(),probe_positions_m=m.linear.QUERY_POINTS+[.002,0,0])
    return basis,prediction,target


def test_exact_geometry_query_binding_and_independent_q63_error():
    basis,prediction,target=data();r=m.compare_arrays(basis,prediction,target)
    assert r['q63']['linear']['coordinate_RMSE_m']==pytest.approx(.001/np.sqrt(3))
    assert r['q63']['linear']['maximum_point_L2_error_m']==pytest.approx(.001)
    assert len(r['q63']['linear']['per_point_L2_error_m'])==63
    assert r['q63']['rest_no_motion']['maximum_point_L2_error_m']==pytest.approx(.002)
    bad=copy.deepcopy(target);bad['constraint_mask'][0,0]=True
    with pytest.raises(ValueError,match='geometry and support'):m.compare_arrays(basis,prediction,bad)
    bad=copy.deepcopy(target);bad['material_probes_m'][0,0]+=1e-12
    with pytest.raises(ValueError,match='common63'):m.compare_arrays(basis,prediction,bad)


def fixture_metadata(tmp,monkeypatch):
    original=m.ROOT
    for name in sorted(set(m.linear.source_closure())|{m.SOURCE}|set(m.META.values())):
        p=tmp/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((original/name).read_bytes())
    lp=json.loads((original/m.META['linear_protocol']).read_text());ls=json.loads((original/m.META['linear_summary']).read_text())
    for row in ls['models']:
        name=lp['output']+'/'+row['job']['id']+'/result.json';p=tmp/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((original/name).read_bytes())
    monkeypatch.setattr(m,'ROOT',tmp)


def test_prepare_temp_write_boundary_exact28_join_no_array_reads(tmp_path,monkeypatch):
    fixture_metadata(tmp_path,monkeypatch)
    monkeypatch.setattr(np,'load',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('Arrays read during preparation')))
    protocol=tmp_path/'opdev'/'protocol.json';out=tmp_path/'new-scores';original_open=Path.open;writes=[]
    def guarded(self,mode='r',*a,**kw):
        if any(c in mode for c in 'wax+'):
            assert self.resolve().is_relative_to(tmp_path);writes.append(self)
        return original_open(self,mode,*a,**kw)
    monkeypatch.setattr(Path,'open',guarded)
    digest=m.prepare(protocol,out);p=m.validate(protocol,digest)
    assert len(p['pairs'])==28 and writes==[protocol] and not out.exists()
    with pytest.raises(ValueError,match='Fresh'):m.prepare(protocol,out)


def test_physical_join_ignores_historical_kind_but_rejects_E_support_F_changes(tmp_path,monkeypatch):
    fixture_metadata(tmp_path,monkeypatch);path=tmp_path/m.META['nonlinear_summary'];s=json.loads(path.read_text())
    s['rows'][0]['case']['kind']='historical-only';path.write_text(json.dumps(s));assert len(m.metadata_plan())==28
    s['rows'][0]['case']['young_Pa']*=1.01;path.write_text(json.dumps(s))
    with pytest.raises(ValueError,match='Physical ID/E/support/F'):m.metadata_plan()


def test_tamper_and_duplicate_case_fail_before_array_load(tmp_path,monkeypatch):
    p=tmp_path/'capture.npz';p.write_bytes(b'original');r={'path':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()};monkeypatch.setattr(m,'ROOT',tmp_path)
    assert m._checked(r)==p
    p.write_bytes(b'tampered')
    with pytest.raises(ValueError,match='Changed'):m._checked(r)


def test_duplicate_or_missing_engineering_cases_cannot_be_filtered(tmp_path,monkeypatch):
    fixture_metadata(tmp_path,monkeypatch);path=tmp_path/m.META['nonlinear_summary'];s=json.loads(path.read_text())
    s['rows'][1]['case']=copy.deepcopy(s['rows'][0]['case']);path.write_text(json.dumps(s))
    with pytest.raises(ValueError,match='Exact14 case joins'):m.metadata_plan()
