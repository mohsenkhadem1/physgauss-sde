"""Bind development hypothesis identities, poses, wrench origins and renders.

This validates provenance before a separate immutable protocol seal; it is not
an access-control boundary and does not read any reserved truth.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.physical_refinement_v15.banks import BASE,ROOT,sha,checked_npz,CAMERAS
from scripts.physical_refinement_v15.quadrature import geometry_cases
from scripts.physical_refinement_v15.study import development_actions


def audit(count=17):
    evidence={};checks=0
    def require(condition,message):
        nonlocal checks
        if not condition:raise ValueError(message)
        checks+=1
    def record(path):
        evidence[str(path.relative_to(ROOT))]=sha(path)
        return json.loads(path.read_text())
    actions=development_actions()
    for case in geometry_cases(count):
        directory=BASE/'hypotheses'/case['name']/'n4'
        manifest=record(directory/'manifest.json')
        spec=manifest['spec']
        for key,value in case.items():
            if key=='name':continue
            actual=spec['case'][key]
            require(actual==value if isinstance(value,str) else np.isclose(actual,value,rtol=0,atol=1e-12),
                    f'Case mismatch {case["name"]}/{key}')
        require(spec['n']==4 and spec['order']==3,'Unexpected prediction discretization')
        require(np.array_equal(spec['case']['gravity'],[0.,0.,0.]) and spec['case']['field_amplitude_J']==0,
                'Common scale reuse requires zero external loads')
        require(np.array_equal(spec['poses_m_rad'],list(actions.values())),'Action list mismatch')
        for name,digest in spec['source_sha256'].items():
            require(sha(ROOT/name)==digest,'Mechanics source changed: '+name)
        for index,(action,pose) in enumerate(actions.items()):
            folder=directory/f'pose-{index:03d}'
            complete=record(folder/'complete.json')
            require(complete['signature']==manifest['signature'],'Pose/manifest identity mismatch')
            require(complete['pose_index']==index and np.array_equal(complete['pose_m_rad'],pose),'Pose mismatch')
            require(complete['minimum_J']>0 and complete['residual_N']<=1e-9 and complete['reaction_balance_N']<=1e-7,
                    'Unqualified mechanical state')
            data,hashes=checked_npz(folder,'reaction-wrench.npz');evidence.update(hashes)
            with data:
                expected=np.array([[0.,0.,0.],[.24,0.,0.]])+pose[:,:3]
                require(np.allclose(data['origins_world_m'],expected,rtol=0,atol=1e-12),'Wrong torque origin')
            rendered=BASE/'render-bank'/case['name']/action
            receipt=record(rendered/'complete.json')
            require(receipt['case']==case['name'] and receipt['action']==action and receipt['pose_index']==index,
                    'Render identity mismatch')
            expected_input={str((folder/name).relative_to(ROOT)) for name in ('complete.json','sensor-geometry.npz')}
            require(set(receipt['input_sha256'])==expected_input,'Render uses unexpected physical state')
            for name,digest in receipt['input_sha256'].items():
                require(sha(ROOT/name)==digest,'Render input changed: '+name)
            for camera in CAMERAS:
                data,hashes=checked_npz(rendered,camera+'.npz');evidence.update(hashes);data.close()
    return dict(scope='Development hypothesis semantic provenance; not a sealed reserved evaluation',
                cases=len(geometry_cases(count)),checks=checks,passed=True,evidence_sha256=evidence)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--gradient-count',type=int,default=17)
    args=parser.parse_args();result=audit(args.gradient_count)
    out=ROOT/'reports/physical-refinement-v15'/f'bank-semantic-audit-g{args.gradient_count}.json'
    if out.exists():raise ValueError('Preserve previous audit')
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:value for key,value in result.items() if key!='evidence_sha256'}))
