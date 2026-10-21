"""Read-only NumPy endpoint/immutability checks; no solver/objective calls."""
from pathlib import Path
import json,hashlib,numpy as np
from experiments.campaign_rigid_inertia import auxiliary_mass_points
ROOT=Path(__file__).resolve().parents[3]
base=ROOT/'runs/gaussian-campaign-v01/authored-force-envelope-v01/stage1/cells/t0000__twist_couple__temporal__hz120'
out=ROOT/'runs/gaussian-campaign-v01/authored-solver-step-damped-v01'
read=lambda p:json.loads(p.read_text())
with np.load(base/'private-provenance/static-reference.npz') as a:rest=a['rest_points_m'];tets=a['tetrahedra']
with np.load(out/'state.npz') as a:state={k:a[k].copy() for k in a.files}
physical=read(base/'private-provenance/physical-inputs.json');q=state['positions_m'];R=state['handle_rotations'];centers=state['handle_centers_m'];spec=physical['spec']
dm=np.swapaxes(rest[tets[:,1:]]-rest[tets[:,:1]],1,2);g=np.linalg.inv(dm);g=np.concatenate([-g.sum(1,keepdims=True),g],axis=1)
F=np.eye(3)+np.einsum('eic,eid->ecd',q[tets]-rest[tets],g);J=np.linalg.det(F)
patch_error=aux_error=0.
for j,h in enumerate(physical['hardware']):
 patch=h['ordered_patch_nodes'];origin=h['reference_origin_node']
 expected=centers[j]+(rest[patch]-rest[origin])@R[j].T
 patch_error=max(patch_error,float(abs(expected-q[patch]).max()))
 offsets,_=auxiliary_mass_points(h['mass_kg'],h['full_COM_inertia_kg_m2'])
 expected=centers[j]+(offsets+np.array(h['COM_offset_m']))@R[j].T
 aux_error=max(aux_error,float(abs(expected-state['auxiliary_positions_m'][6*j:6*j+6]).max()))
fixture=float(abs(q[spec['fixture_nodes']]-rest[spec['fixture_nodes']]).max());proper=max(float(abs(np.swapaxes(R,-1,-2)@R-np.eye(3)).max()),float(abs(np.linalg.det(R)-1).max()))
protocol=read(ROOT/'runs/operator-dev/authored-solver-step-damped-v01/protocol.json');unchanged=all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in protocol['source_and_input_sha256'].items())
result={'role':'Independent posthoc endpoint and frozen-input verification, no model/step/gradient call','all_state_arrays_finite':all(np.isfinite(v).all() for v in state.values()),'all_state_arrays_float64':all(v.dtype==np.float64 for v in state.values()),'minimum_J':float(J.min()),'maximum_J':float(J.max()),'rigid_patch_pose_error_m':patch_error,'auxiliary_pose_error_m':aux_error,'fixture_error_m':fixture,'SO3_error':proper,'all_frozen_D_and_ABC_source_inputs_unchanged':unchanged,'own_PID_missing':not Path('/proc',str(read(out/'execution.json')['pid'])).exists(),'ABC_PID_missing':not Path('/proc','2098418').exists(),'original_failed_cell_retained':(base/'failure.json').exists() and read(base/'execution.json')['completed'] is False,'production_qualification':False}
assert result['all_state_arrays_finite'] and result['all_state_arrays_float64'] and result['minimum_J']>=.1
assert max(patch_error,aux_error,fixture,proper)<=1e-12 and unchanged and result['own_PID_missing'] and result['ABC_PID_missing']
target=ROOT/'runs/operator-dev/authored-solver-step-damped-v01/posthoc-verification.json';assert not target.exists();target.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
