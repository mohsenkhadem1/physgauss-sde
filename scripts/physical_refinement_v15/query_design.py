"""Public acquisition menu; no actual observation or truth reader."""
from scripts.physical_refinement_v15.banks import CAMERAS,query


def menu(initial_camera):
    if initial_camera not in CAMERAS:raise ValueError('Unknown camera')
    return [(f'view_{camera}','initial',camera) for camera in CAMERAS if camera!=initial_camera]+[
        (name,name,'left') for name in ('probe_left_twist','probe_right_twist','probe_left_bend','probe_right_bend')]


def prediction_queries(expanded,initial_camera,track):
    if track not in ('V','VF'):raise ValueError('Unknown measurement track')
    initial=query(expanded,'initial',initial_camera,name='initial')
    queries=[query(expanded,action,camera,include_wrench=track=='VF' and name.startswith('probe'),name=name)
             for name,action,camera in menu(initial_camera)]
    return initial,queries
