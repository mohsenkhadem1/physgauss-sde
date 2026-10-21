"""TRAIN geometry-only audit of information omitted by handle-center encoding."""
import argparse
from collections import defaultdict
import copy
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def attachment_descriptor(points,node,patch,mass):
    # IDs and offsets share one canonical ordering. Constraint order itself is
    # irrelevant, but an ID-sorted context must permute its offsets together.
    points=np.asarray(points,dtype=float);patch=np.sort(np.asarray(patch,dtype=int))
    offsets=points[patch]-points[node]
    # Exactly the v0.3 worker's authored diagonal inertia, including sphere floor.
    inertia=mass*np.mean(np.sum(offsets**2,axis=1)[:,None]-offsets**2,axis=0)
    diagonal=np.maximum(inertia,.4*mass*.003**2)
    return {'center_m':points[node].tolist(),'patch_nodes':sorted(patch.tolist()),
        'reference_local_offsets_m':offsets.tolist(),'patch_centroid_offset_m':offsets.mean(0).tolist(),
        'authored_diagonal_inertia_kg_m2':diagonal.tolist(),'handle_mass_kg':mass}


def run(config_path,output):
    config=json.loads(Path(config_path).read_text());dataset=Path(config['dataset'])
    cases={c['episode_id']:c for c in json.loads((dataset/'plan.json').read_text())['episodes']}
    ids=[f't{i:04d}' for i in range(12)]+config['new_train_control_ids']
    if len(set(ids))!=len(ids) or any(cases[e]['split']!='train' for e in ids):
        raise ValueError('Explicit fitting-TRAIN role required before private geometry access')
    hashes={str(p):sha(p) for p in [Path(config_path),dataset/'plan.json']}
    rows=[];groups=defaultdict(list);pairs=defaultdict(list);loaded={}
    for eid in ids:
        episode=dataset/'episodes'/eid
        reference=episode/'reference.npz';spec_path=episode/'truth/boundaries.json'
        with np.load(reference) as raw:points=raw['rest_points_m'].copy()
        spec=json.loads(spec_path.read_text());hashes.update({str(p):sha(p) for p in [reference,spec_path]})
        loaded[eid]=(points,spec)
        entries=[]
        for h,(node,patch) in enumerate(zip(spec['load_nodes'],spec['load_patches'])):
            d=attachment_descriptor(points,node,patch,spec['handle_mass_kg_each'])
            d.update(episode_id=eid,handle_index=h,node_index=node)
            entries.append(d);groups[(h,tuple(d['center_m']))].append(d);rows.append(d)
        pairs[tuple(tuple(e['center_m']) for e in entries)].append(entries)
    ambiguities=[]
    for (h,center),items in groups.items():
        variants={tuple(d['patch_nodes']) for d in items}
        if len(variants)<=1:continue
        inertia=np.array([d['authored_diagonal_inertia_kg_m2'] for d in items])
        ambiguities.append({'handle_index':h,'encoded_center_m':list(center),
            'TRAIN_instances':len(items),'distinct_patch_sets':len(variants),
            'maximum_same_axis_inertia_ratio':float((inertia.max(0)/inertia.min(0)).max()),
            'examples':items})
    # An explicit, geometry-valid alternative demonstrates information loss,
    # without claiming any measured response changes under the alternative.
    counterexamples=[]
    try:from gaussian_operator import physical_vector
    except ImportError:from experiments.gaussian_operator import physical_vector
    for group in ambiguities:
        for original in group['examples']:
            eid,h=original['episode_id'],original['handle_index'];points,spec=loaded[eid]
            blocked=set(spec['fixture_nodes'])|set(spec['load_patches'][1-h])
            normal=np.asarray(spec['reference_load_normals'][h]);axis=int(np.argmax(abs(normal)))
            for alternative in group['examples']:
                patch=alternative['patch_nodes']
                if patch==original['patch_nodes'] or blocked.intersection(patch):continue
                if not np.allclose(points[patch,axis],points[spec['load_nodes'][h],axis],atol=1e-12):continue
                modified=copy.deepcopy(spec);modified['load_patches'][h]=patch
                a,b=physical_vector(cases[eid],spec,points),physical_vector(cases[eid],modified,points)
                if not np.array_equal(a,b):raise AssertionError('Unexpected encoded hardware dependence')
                counterexamples.append({'original_episode_id':eid,'alternative_patch_observed_in':alternative['episode_id'],
                    'handle_index':h,'same_full25D_vector':True,'physical_vector':a.tolist(),
                    'original_hardware':original,'alternative_hardware':alternative,
                    'qualification':'Geometry-only counterfactual: same case/material/support/force and both handle centers, change only one surface attachment patch to a disjoint nine-node set observed in another TRAIN scene. No counterfactual response simulated or compared.'})
                break
            if counterexamples:break
        if counterexamples:break
    pair_variants=[]
    for center,items in pairs.items():
        signatures={tuple(tuple(h['patch_nodes']) for h in pair) for pair in items}
        if len(signatures)>1:pair_variants.append({'both_centers_m':[list(x) for x in center],'instances':len(items),'distinct_patch_pairs':len(signatures)})
    source=Path(__file__);physical=Path(__file__).with_name('gaussian_operator.py');worker=Path(__file__).with_name('isaac_calibrated_probes_v03.py')
    result={'train_layouts':len(ids),'handle_instances':len(rows),'distinct_center_index_groups':len(groups),
        'same_encoded_center_groups_with_multiple_attachment_sets':len(ambiguities),
        'instances_in_ambiguous_center_groups':sum(x['TRAIN_instances'] for x in ambiguities),
        'naturally_repeated_both_centers_with_different_patches':pair_variants,
        'ambiguities':ambiguities,'geometry_only_same25D_counterexamples':counterexamples,
        'input_sha256':hashes,'source_sha256':{p.name:sha(p) for p in [source,physical,worker]},
        'response_labels_opened':False,'reserved_geometry_or_responses_opened':False,
        'qualification':'Patch/inertia omission is demonstrated, not proven to cause under-response or the within-layout superposition defect. Different natural scenes may have different materials/supports. Known full reference patches should condition every model/policy equally, or be explicit physical nuisance variables.'}
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    with Path(output).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/gaussian_campaign_v01.json');p.add_argument('--output',required=True)
    a=p.parse_args();r=run(a.config,a.output);print(json.dumps({k:r[k] for k in ['train_layouts','handle_instances','same_encoded_center_groups_with_multiple_attachment_sets','instances_in_ambiguous_center_groups']}))
