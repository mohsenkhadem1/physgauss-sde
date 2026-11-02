import copy
import numpy as np
import pytest
import torch
from scripts.compositional_neural_mechanics.feature_pack import build_features,default_probes,load_labelled_record,batch_records,sha,load_frozen_laws
from scripts.compositional_neural_mechanics.scale_separated_material import ScaleSeparatedMaterialLaw
from scripts.compositional_neural_mechanics.interfaces import SharedInterfaceLaw

def fixture():
 torch.manual_seed(51)
 scene=dict(name='synthetic-train',geometry=dict(dimensions=(.2,.08,.06),taper=.1,arch=.005),bulk_latent=[.1,.2,-1.],gradient=.2,factors={'heterogeneity':1},interface_latents=[[np.log(3000),.4,0.],[np.log(5000),.7,1.]],centers=[[.5,.25],[.5,.7]],widths=[[.3,.35],[.25,.3]],gravity=[0,0,0],density_kg_m3=125.)
 bulk=ScaleSeparatedMaterialLaw().double();interface=SharedInterfaceLaw().double()
 for law in (bulk,interface):
  law.eval()
  for parameter in law.parameters():parameter.requires_grad_(False)
 nodes=np.array([[0,.2,.3],[.5,.4,.6],[1,.8,.7]])
 pose=np.zeros((2,6));pose[1,0]=.005
 return scene,bulk,interface,nodes,pose

def test_features_permutation_no_ids_and_numeric_fingerprints():
 s,b,i,n,p=fixture();a=build_features(s,p,n,n[:2],b,i)
 assert a['nodefeatures'].shape==(3,81)and a['globalfeatures'].shape==(88,)and a['queryfeatures'].shape==(2,9)
 changed=copy.deepcopy(s);changed['name']='never-a-network-feature';changed['future_truth']=np.ones(100)*999
 c=build_features(changed,p,n[::-1],n[:2][::-1],b,i)
 np.testing.assert_allclose(c['nodefeatures'],a['nodefeatures'][::-1]);np.testing.assert_allclose(c['queryfeatures'],a['queryfeatures'][::-1]);np.testing.assert_array_equal(c['globalfeatures'],a['globalfeatures'])
 # Exact bulk scale is carried numerically in law fingerprint, not encoded by a hash.
 scaled=copy.deepcopy(s);scaled['bulk_latent'][0]+=np.log(2)
 d=build_features(scaled,p,n,n[:2],b,i)
 np.testing.assert_allclose(d['nodefeatures'][:,11:],2*a['nodefeatures'][:,11:],atol=1e-9,rtol=1e-10)
 # Queries arbitrary and their reference positions derive from the same geometry.
 e=build_features(s,p,n,np.array([[.3,.5,.7]]),b,i);assert e['queryfeatures'].shape==(1,9)


def test_committed_label_loader_rejects_test_and_modified_artifacts(tmp_path):
 s,b,i,n,p=fixture();labels=tmp_path/'labels.npz';np.savez(labels,material_query=n[:2],displacement=np.zeros((2,3)),wrenches=np.zeros((2,6)))
 kwargs=dict(expected_sha256=sha(labels),node_material=n,bulk_law=b,interface_law=i,probes=default_probes(),checkpoint_sha256={'bulk':'synthetic','interface':'synthetic'},allowed_scenes_by_split={'train':[s['name']],'validation':[]})
 metadata=dict(scene_id=s['name'],split='train',prescribed_pose=p.tolist())
 record=load_labelled_record(s,metadata,labels,**kwargs);pack,contract=batch_records([record,record]);assert pack['nodefeatures'].shape==(2,3,81)
 with pytest.raises(ValueError,match='outside'):load_labelled_record(s,{**metadata,'split':'test'},tmp_path/'nonexistent.npz',**kwargs)
 labels.write_bytes(b'changed')
 with pytest.raises(ValueError,match='artifact changed'):load_labelled_record(s,metadata,labels,**kwargs)


def test_checkpoint_versions_are_explicit_and_verified(tmp_path):
 _,b,i,_,_=fixture();bp=tmp_path/'bulk.pt';ip=tmp_path/'interface.pt'
 torch.save({'state_dict':b.state_dict()},bp);torch.save(dict(state_dict=i.state_dict(),reference_length=i.reference_length,bases=6,width=16),ip)
 expected={'bulk':sha(bp),'interface':sha(ip)};bb,ii=load_frozen_laws(bp,ip,expected)
 assert not any(x.requires_grad for x in bb.parameters())and not any(x.requires_grad for x in ii.parameters())
 with pytest.raises(ValueError,match='checkpoint changed'):load_frozen_laws(bp,ip,{**expected,'interface':'wrong'})
