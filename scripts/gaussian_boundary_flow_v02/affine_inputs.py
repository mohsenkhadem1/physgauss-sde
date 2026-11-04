"""Version immutable predictor packages with an initial-only affine observation map."""
import argparse
from pathlib import Path
from scripts.gaussian_boundary_flow_v02.export_inputs import load_package,write_package,sha
from scripts.gaussian_boundary_flow_v02.affine_observation import affine_observation_matrix


def export_affine(source,destination):
    source=Path(source);data=load_package(source)
    matrix,diagnostic=affine_observation_matrix(data['nodes'],data['observed_initial'])
    data['observation_matrix']=matrix
    data['observation_map_revision']={'source_package':str(source.resolve()),
        'source_manifest_sha256':sha(source/'manifest.json'),
        'rule':'Initial-geometry affine-complete local map; no motion or future targets used',
        'maximum_row_l1':diagnostic['maximum_row_l1'],
        'maximum_row_l2':diagnostic['maximum_row_l2'],
        'negative_weights_allowed':True}
    return write_package(data,destination)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();m=export_affine(a.source,a.output)
    print(m['name'],m['predictor_digest'])
