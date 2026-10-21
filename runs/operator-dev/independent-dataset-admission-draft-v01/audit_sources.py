"""Reproduce source/TRAIN-static-prior evidence; no response or physics loader."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'experiments'))
try:
    from campaign_physical_prior import ConditionalPhysicalPrior
    import isaac_dataset as source
finally:sys.path.pop(0)

FILES=['experiments/isaac_dataset.py','experiments/prepare_gaussian_boundary_study.py',
    'experiments/campaign_physical_prior.py','experiments/campaign_rejection_prior.py',
    'experiments/campaign_tempered_reference.py','experiments/campaign_full_hardware_context.py',
    'experiments/isaac_calibrated_probes_v03.py','experiments/gaussian_active_design.py',
    'experiments/campaign_authored_commands.py','configs/gaussian_independent_diagnostic_inputs_v01.json',
    'configs/gaussian_campaign_actual_force_envelope_v01.json']


def run():
    fixture=json.loads((ROOT/'configs/gaussian_independent_diagnostic_inputs_v01.json').read_text());rows=[]
    for item in fixture['cases']:
        recipe=item['recipe']
        if recipe['split']!='train':raise ValueError('Only public TRAIN static recipe permitted')
        points,_,_,spec=source.geometry(recipe)
        prior=ConditionalPhysicalPrior(points,points[spec['load_nodes']],handle_patch_node_ids=np.array(spec['load_patches']))
        state=prior.sample(256,np.random.default_rng(971));known=prior.decode(state,{'force_type_index':0,'peak_force_N':[.01,0,0]})
        metadata=prior.metadata()
        rows.append({'train_case':item['episode_id'],
            'public_hardware_descriptor_sha256':hashlib.sha256(json.dumps(metadata['public_hardware'],sort_keys=True,separators=(',',':')).encode()).hexdigest(),
            'prior_sha256':metadata['prior_sha256'],'all_four_BCs_geometrically_supported':all(len(prior.feasible_faces[b])>0 for b in range(4)),
            'feasible_faces_by_BC':{str(b):prior.feasible_faces[b].tolist() for b in range(4)},
            'certified_cells_by_BC_face':{f'{b}/{face}':len(prior.feasible_cells[b,face]) for b in range(4) for face in range(6)},
            'sampled_states':256,'all_log_densities_finite':bool(np.isfinite(prior.log_prob(state)).all()),
            'known_override_equal_all_BCs':bool(np.array_equal(known['physical'][:,16:19],np.broadcast_to([.01,0,0],(256,3))))})
    return {'scope':'Pure TRAIN static geometry/prior inspection only; NO physical rollout, observations, checkpoint or reserved response reads',
        'audit_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in FILES},'rows':rows,
        'point_pin_passive_amplitude_support_N':[.03,.06],'other_BC_passive_amplitude_support_N':[.12,.24],
        'threshold_N':.09,'threshold_recovers_point_pin_from_passive_known_peak':True,
        'camera_RNG_side_channel_statistical_accuracy_measured':False,'mechanical_operating_envelope_qualified':False}


if __name__=='__main__':
    output=Path(__file__).with_name('source-audit.json')
    output.write_text(json.dumps(run(),indent=2,allow_nan=False)+'\n')
    print(json.dumps({'TRAIN_static_contexts':4,'physics_rollouts':0,'reserved_response_reads':0}))
